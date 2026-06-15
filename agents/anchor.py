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
        # ANCHOR starts from the SAME frozen independent foundation as MONK
        # (spec §9). Market prices are only introduced later, in candidate
        # generation — never here.
        coverage = float(view.data_coverage.get("overall", 1.0))
        ind = build_independent_forecast(
            view.football_features, data_coverage_score=coverage,
            w_elo=0.5, w_poisson=0.5,
        )
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
        if not market:
            return []

        candidates = []
        for outcome in ("home", "draw", "away"):
            prob = getattr(forecast, f"{outcome}_probability")
            lower = getattr(forecast, f"{outcome}_lower_bound")
            
            fill_price = market.expected_fill_price.get(outcome) or market.best_ask.get(outcome) or market.midpoint.get(outcome)
            
            if not fill_price:
                continue

            fee_buffer = config.FEE_BUFFER
            slippage_buffer = config.SLIPPAGE_BUFFER
            model_risk_buffer = config.MODEL_RISK_BUFFER

            conservative_edge = lower - fill_price - fee_buffer - slippage_buffer - model_risk_buffer
            ev_after_costs = prob - fill_price - fee_buffer - slippage_buffer

            # ANCHOR searches ALL three outcomes (spec §9). Candidate gen only
            # enforces the executable price band and a positive raw edge; the
            # disciplined conservative-edge / EV gates are applied downstream in
            # build_recommendation so a weak edge produces an auditable
            # abstention rather than a silent drop.
            gross_edge = prob - fill_price
            if gross_edge > 0 and config.ANCHOR_MIN_ENTRY_PRICE <= fill_price <= config.ANCHOR_MAX_ENTRY_PRICE:
                    candidates.append(
                        TradeCandidate(
                            agent_name=self.name,
                            fixture_id=forecast.fixture_id,
                            outcome=outcome,
                            probability_mean=prob,
                            probability_lower_bound=lower,
                            probability_upper_bound=getattr(forecast, f"{outcome}_upper_bound"),
                            market_midpoint=market.midpoint.get(outcome),
                            best_ask=market.best_ask.get(outcome),
                            expected_fill_price=fill_price,
                            gross_edge=prob - fill_price,
                            conservative_edge=conservative_edge,
                            expected_value_after_costs=ev_after_costs,
                            signal_type="disciplined_value",
                            signals=tuple(),
                            candidate_created_at=datetime.now(timezone.utc),
                            candidate_expires_at=None,
                            forecast_id=forecast.forecast_id,
                            correlation_key=f"{self.name}_{forecast.fixture_id}_{outcome}",
                        )
                    )
        return candidates

    def generate_recommendations(
        self,
        candidates: list[TradeCandidate],
        forecast: AgentForecast,
        view: AgentDataView,
        market: MarketContext | None,
        bankroll: float,
    ) -> list[AgentRecommendation]:
        # ANCHOR selects at most ONE disciplined trade per fixture: its best raw
        # edge. The conservative-edge / cost / EV gates run inside
        # build_recommendation, so an undersized edge becomes an abstention.
        ranked = sorted(candidates, key=lambda c: c.gross_edge or -1, reverse=True)
        return [
            reco_from_candidate(self.name, cand, view, forecast, bankroll, self.profile)
            for cand in ranked[: self.profile.max_bets_per_window]
        ]

    def create_execution_plan(
        self,
        recommendations: list[AgentRecommendation],
    ) -> ExecutionPlan:
        return ExecutionPlan()
