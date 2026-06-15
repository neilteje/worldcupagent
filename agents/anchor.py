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
from models.independent_forecast import build_independent_forecast
from agents.monk import _coverage_from_features
from agents._reco import reco_from_candidate
from agents._conviction import build_council_forecast, conviction_candidates

class AnchorStrategy(AgentStrategy):
    name = "anchor"

    def __init__(self):
        self.profile = get_profile(self.name)

    def build_data_view(
        self,
        snapshot: FixtureDataSnapshot,
        prior_agent_forecast: AgentForecast | None = None,
    ) -> AgentDataView:
        # ANCHOR needs the independent football foundation from MONK
        # but is allowed to see the market features when creating candidates.
        
        coverage = _coverage_from_features(snapshot.football_context)
        view_state = {
            "agent": self.name,
            "fixture_id": snapshot.fixture_id,
            "window": snapshot.window,
        }
        view_hash = hashlib.sha256(json.dumps(view_state, sort_keys=True, default=str).encode()).hexdigest()

        return AgentDataView(
            agent_name=self.name,
            fixture_id=snapshot.fixture_id,
            window=snapshot.window,
            as_of_timestamp=snapshot.as_of_timestamp,
            football_features=snapshot.football_context,
            live_features=snapshot.live_context,
            external_model_predictions={},
            evidence=tuple(),
            market_features=None, # Market features provided in Candidate Generation
            data_coverage=coverage,
            prohibited_fields_removed=tuple(),
            warnings=tuple(),
            data_view_hash=view_hash,
        )

    def build_forecast(
        self,
        view: AgentDataView,
    ) -> AgentForecast:
        # Conviction: ANCHOR bets off the SHARED goated council forecast when
        # present (live path); falls back to its independent model offline.
        council = build_council_forecast(view, self.name, "anchor_v2_conviction")
        if council is not None:
            return council
        coverage = float(view.data_coverage.get("overall", 1.0))
        ind = build_independent_forecast(
            view.football_features, data_coverage_score=coverage,
        )  # uses walk-forward-tuned params (models.independent_forecast.TUNED)
        p = ind["probabilities"]
        lo = ind["lower_bounds"]
        hi = ind["upper_bounds"]
        confidence = round(min(0.99, max(p.values())), 4)
        forecast_id = hashlib.sha256(f"{view.data_view_hash}_anchor_v1".encode()).hexdigest()

        return AgentForecast(
            agent_name=self.name,
            fixture_id=view.fixture_id,
            window=view.window,
            as_of_timestamp=view.as_of_timestamp,
            home_probability=p["home"],
            draw_probability=p["draw"],
            away_probability=p["away"],
            home_lower_bound=lo["home"],
            draw_lower_bound=lo["draw"],
            away_lower_bound=lo["away"],
            home_upper_bound=hi["home"],
            draw_upper_bound=hi["draw"],
            away_upper_bound=hi["away"],
            confidence=confidence,
            data_coverage_score=coverage,
            forecast_type="independent_deterministic",
            model_version="anchor_v1",
            components=ind.get("components", {}),
            evidence_ids=tuple(),
            warnings=ind.get("warnings", tuple()),
            data_view_hash=view.data_view_hash,
            forecast_id=forecast_id,
        )

    def generate_candidates(
        self,
        forecast: AgentForecast,
        view: AgentDataView,
        market: MarketContext | None,
    ) -> list[TradeCandidate]:
        # Conviction: ANCHOR backs the best-EV council outcome (favorite
        # included) clearing its disciplined edge bar and the 12% floor.
        return conviction_candidates(forecast, view, market, self.profile, self.name)

    def generate_recommendations(
        self,
        candidates: list[TradeCandidate],
        forecast: AgentForecast,
        view: AgentDataView,
        market: MarketContext | None,
        bankroll: float,
    ) -> list[AgentRecommendation]:
        # ANCHOR selects at most ONE disciplined trade per fixture: its best EV.
        ranked = sorted(candidates, key=lambda c: c.expected_value_after_costs or -1, reverse=True)
        return [
            reco_from_candidate(self.name, cand, view, forecast, bankroll, self.profile)
            for cand in ranked[: self.profile.max_bets_per_window]
        ]

    def create_execution_plan(
        self,
        recommendations: list[AgentRecommendation],
    ) -> ExecutionPlan:
        return ExecutionPlan()
