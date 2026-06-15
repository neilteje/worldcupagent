from typing import Any
import hashlib
from datetime import datetime, timezone
import json
from agents.base import AgentStrategy, ExecutionPlan, AgentRecommendation
from agents.contracts import (
    FixtureDataSnapshot,
    AgentDataView,
    AgentForecast,
    MarketContext,
    TradeCandidate,
)
from harness.profiles import get_profile
import config
from models.forecast_contracts import MatchForecast # the legacy forecast
from betting.policy import select_picks, suppress_blitz_draw_picks

class BlitzLegacyDataView(AgentDataView):
    # Specialized subclass to hold legacy objects
    legacy_forecast: MatchForecast | None = None

class BlitzStrategy(AgentStrategy):
    name = "blitz"

    def __init__(self):
        self.profile = get_profile(self.name)

    def build_data_view(
        self,
        snapshot: FixtureDataSnapshot,
        prior_agent_forecast: AgentForecast | None = None,
    ) -> AgentDataView:
        
        view_state = {
            "agent": self.name,
            "fixture_id": snapshot.fixture_id,
            "window": snapshot.window,
        }
        view_hash = hashlib.sha256(json.dumps(view_state, sort_keys=True).encode()).hexdigest()

        view = BlitzLegacyDataView(
            agent_name=self.name,
            fixture_id=snapshot.fixture_id,
            window=snapshot.window,
            as_of_timestamp=snapshot.as_of_timestamp,
            football_features=snapshot.football_context,
            live_features=snapshot.live_context,
            external_model_predictions={},
            evidence=tuple(),
            market_features=None, 
            data_coverage={"overall": 1.0},
            prohibited_fields_removed=tuple(),
            warnings=tuple(),
            data_view_hash=view_hash,
        )
        
        # We will populate view.legacy_forecast externally in the cycle.py adapter
        return view

    def build_forecast(
        self,
        view: AgentDataView,
    ) -> AgentForecast:
        forecast_id = hashlib.sha256(f"{view.data_view_hash}_blitz_v1".encode()).hexdigest()

        return AgentForecast(
            agent_name=self.name,
            fixture_id=view.fixture_id,
            window=view.window,
            as_of_timestamp=view.as_of_timestamp,
            home_probability=0.45,
            draw_probability=0.25,
            away_probability=0.30,
            home_lower_bound=0.40,
            draw_lower_bound=0.20,
            away_lower_bound=0.25,
            home_upper_bound=0.50,
            draw_upper_bound=0.30,
            away_upper_bound=0.35,
            confidence=0.8,
            data_coverage_score=view.data_coverage.get("overall", 1.0),
            forecast_type="legacy",
            model_version="legacy_v1",
            components={},
            evidence_ids=tuple(),
            warnings=tuple(),
            data_view_hash=view.data_view_hash,
            forecast_id=forecast_id,
        )

    def generate_candidates(
        self,
        forecast: AgentForecast,
        view: AgentDataView,
        market: MarketContext | None,
    ) -> list[TradeCandidate]:
        if not market or not isinstance(view, BlitzLegacyDataView) or not view.legacy_forecast:
            return []

        # mock moneyline using market_midpoint for legacy compatibility
        moneyline = None
        if market and market.midpoint:
            moneyline = {
                "market_source": "polymarket",
                "outcomes": {
                    "home": {"current_mid_yes": market.midpoint.get("home")},
                    "draw": {"current_mid_yes": market.midpoint.get("draw")},
                    "away": {"current_mid_yes": market.midpoint.get("away")}
                }
            }

        probabilities = {
            "home": view.legacy_forecast.home_probability,
            "draw": view.legacy_forecast.draw_probability,
            "away": view.legacy_forecast.away_probability
        }
        
        home_code = view.football_features.get("home_code", "home") if view.football_features else "home"
        away_code = view.football_features.get("away_code", "away") if view.football_features else "away"

        self.skip_reasons: list[str] = []
        legacy_picks = select_picks(
            profile=self.profile,
            probabilities=probabilities,
            moneyline=moneyline,
            home_code=home_code,
            away_code=away_code,
            bankroll=100.0,
            window=view.window,
            confidence_num=view.legacy_forecast.confidence,
            skip_reasons=self.skip_reasons,
        )

        # Draw removal runs AFTER BLITZ's existing candidate selection (spec §11),
        # via the canonical suppressor — it never promotes a replacement and
        # records ``blitz_draw_disabled`` per removed draw.
        kept = suppress_blitz_draw_picks(self.profile, legacy_picks, self.skip_reasons)
        self.draws_removed = 0

        candidates = []
        for pick in kept:
            candidates.append(
                TradeCandidate(
                    agent_name=self.name,
                    fixture_id=forecast.fixture_id,
                    outcome=pick.slot,
                    probability_mean=pick.our_prob,
                    probability_lower_bound=pick.our_prob,  # Legacy uses point estimate
                    probability_upper_bound=pick.our_prob,
                    market_midpoint=pick.fair_prob,
                    best_ask=pick.limit_price,
                    expected_fill_price=pick.entry_price,
                    gross_edge=pick.edge_vs_fair,
                    conservative_edge=pick.edge_vs_fair,
                    expected_value_after_costs=pick.ev_per_dollar,
                    signal_type="blitz_legacy",
                    signals=tuple(),
                    candidate_created_at=datetime.now(timezone.utc),
                    candidate_expires_at=None,
                    forecast_id=forecast.forecast_id,
                    correlation_key=f"{self.name}_{forecast.fixture_id}_{pick.slot}",
                )
            )
        # Stash the sized picks so the legacy execution path can use exact stakes.
        self.legacy_picks = kept
        return candidates

    def generate_recommendations(
        self,
        candidates: list[TradeCandidate],
        forecast: AgentForecast,
        view: AgentDataView,
        market: MarketContext | None,
        bankroll: float,
    ) -> list[AgentRecommendation]:
        recs = []
        # Convert back to legacy recommendations
        for cand in candidates:
            # We preserve the legacy path, so we don't go through the central portfolio coordinator
            # But we still return recommendations if needed by the runner.
            pass
        return recs

    def create_execution_plan(
        self,
        recommendations: list[AgentRecommendation],
    ) -> ExecutionPlan:
        return ExecutionPlan()
