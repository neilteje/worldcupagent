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
    DirectionalSignal,
)
from harness.profiles import get_profile
import config
from agents.monk import _coverage_from_features
from agents._reco import reco_from_candidate
from agents._conviction import build_council_forecast, skew_candidates
from models.calibration import normalize_probs
from models.poisson_model import poisson_1x2

_BAND_HALF = 0.06  # offline-fallback band, widened by coverage


class HunterStrategy(AgentStrategy):
    name = "hunter"

    def __init__(self):
        self.profile = get_profile(self.name)

    def build_data_view(
        self,
        snapshot: FixtureDataSnapshot,
        prior_agent_forecast: AgentForecast | None = None,
    ) -> AgentDataView:
        
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
            market_features=None,
            data_coverage=coverage,
            prohibited_fields_removed=tuple(),
            warnings=tuple(),
            data_view_hash=view_hash,
        )

    def build_forecast(
        self,
        view: AgentDataView,
    ) -> AgentForecast:
        # Conviction: HUNTER bets off the SHARED goated council forecast when
        # present (live path); falls back to a Poisson belief offline.
        council = build_council_forecast(view, self.name, "hunter_v2_conviction")
        if council is not None:
            return council

        forecast_id = hashlib.sha256(f"{view.data_view_hash}_hunter_v1".encode()).hexdigest()
        coverage = float(view.data_coverage.get("overall", 1.0))
        ff = view.football_features or {}
        det = ff.get("deterministic_model") or {}
        eg = det.get("expected_goals") or ff.get("expected_goals") or {}
        lam_h = float(eg.get("lambda_home", 1.3) or 1.3)
        lam_a = float(eg.get("lambda_away", 1.3) or 1.3)
        pm = poisson_1x2(lam_h, lam_a)
        p = normalize_probs({"home": pm.get("home", 0.34), "draw": pm.get("draw", 0.33),
                             "away": pm.get("away", 0.33)})
        half = _BAND_HALF + (1.0 - coverage) * 0.10
        lo = {k: max(0.0, p[k] - half) for k in p}
        hi = {k: min(1.0, p[k] + half) for k in p}
        confidence = round(min(0.99, max(p.values())), 4)

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
            forecast_type="poisson_fallback",
            model_version="hunter_v1",
            components={"lambda_home": lam_h, "lambda_away": lam_a},
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
        return skew_candidates(forecast, view, market, self.profile, self.name)

    def generate_recommendations(
        self,
        candidates: list[TradeCandidate],
        forecast: AgentForecast,
        view: AgentDataView,
        market: MarketContext | None,
        bankroll: float,
    ) -> list[AgentRecommendation]:
        limit = min(
            self.profile.max_bets_per_window,
            max(1, int(config.HUNTER_MAX_TAIL_POSITIONS_PER_FIXTURE)),
        )
        ranked = sorted(candidates, key=lambda c: c.expected_value_after_costs or -1, reverse=True)
        return [
            reco_from_candidate(self.name, cand, view, forecast, bankroll, self.profile)
            for cand in ranked[:limit]
        ]

    def create_execution_plan(
        self,
        recommendations: list[AgentRecommendation],
    ) -> ExecutionPlan:
        return ExecutionPlan()
