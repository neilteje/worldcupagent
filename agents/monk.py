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
import config
from harness.profiles import get_profile
from models.independent_forecast import build_independent_forecast
from agents._reco import reco_from_candidate
from agents._conviction import build_council_forecast, build_engine_forecast, conviction_candidates


def _coverage_from_features(football_features: dict) -> dict:
    """Feature-level coverage (spec §18): presence of each forecast-relevant
    football source. ``overall`` is the mean of the tracked features."""
    ff = football_features or {}
    det = ff.get("deterministic_model") or {}
    feats = {
        "deterministic_model": 1.0 if det.get("home_state") and det.get("away_state") else 0.0,
        "sportmonks": 1.0 if ff.get("sportmonks_digest") else 0.0,
        "supabase": 1.0 if ff.get("supabase_digest") else 0.0,
        "bzzoiro": 1.0 if ff.get("bzzoiro_digest") else 0.0,
    }
    # The frozen independent snapshot already carries an authoritative coverage
    # score (built upstream from the full feature set). Prefer it when present so
    # downstream gates see the real coverage, not a re-derivation from digests
    # that may not be re-attached to the agent view.
    snap = ff.get("independent_forecast") or {}
    snap_cov = snap.get("data_coverage_score")
    if snap_cov is not None:
        feats["overall"] = round(float(snap_cov), 4)
    else:
        feats["overall"] = round(sum(feats.values()) / len(feats), 4)
    return feats


class MonkStrategy(AgentStrategy):
    name = "monk"

    def __init__(self):
        self.profile = get_profile(self.name)

    def _scrub_market_fields(self, features: dict) -> tuple[dict, tuple[str, ...]]:
        """MONK must not receive any market-derived fields. Uses the shared,
        recursive, fail-closed scrubber (spec §8)."""
        from reasoning.market_blind import scrub_market_fields
        return scrub_market_fields(features or {})

    def build_data_view(
        self,
        snapshot: FixtureDataSnapshot,
        prior_agent_forecast: AgentForecast | None = None,
    ) -> AgentDataView:
        football_features, removed = self._scrub_market_fields(snapshot.football_context)
        coverage = _coverage_from_features(football_features)

        # Build deterministic stable hash
        view_state = {
            "agent": self.name,
            "fixture_id": snapshot.fixture_id,
            "window": snapshot.window,
            "football_features": football_features,
            # include other view states
        }
        view_hash = hashlib.sha256(json.dumps(view_state, sort_keys=True, default=str).encode()).hexdigest()

        return AgentDataView(
            agent_name=self.name,
            fixture_id=snapshot.fixture_id,
            window=snapshot.window,
            as_of_timestamp=snapshot.as_of_timestamp,
            football_features=football_features,
            live_features=snapshot.live_context,
            external_model_predictions={},
            evidence=tuple(),
            market_features=None, # Market blind
            data_coverage=coverage,
            prohibited_fields_removed=removed,
            warnings=tuple(),
            data_view_hash=view_hash,
        )

    def build_forecast(
        self,
        view: AgentDataView,
    ) -> AgentForecast:
        engine = build_engine_forecast(view, self.name)
        if engine is not None:
            return engine
        # Compatibility fallback for snapshots without probability_engine output.
        council = build_council_forecast(view, self.name, "monk_v2_conviction")
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
        forecast_id = hashlib.sha256(f"{view.data_view_hash}_monk_v1".encode()).hexdigest()

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
            model_version="monk_v1",
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
        # Conviction: back the best-EV council outcome (favorite included) that
        # clears MONK's strict edge bar and the 12% probability floor.
        return conviction_candidates(forecast, view, market, self.profile, self.name)

    def generate_recommendations(
        self,
        candidates: list[TradeCandidate],
        forecast: AgentForecast,
        view: AgentDataView,
        market: MarketContext | None,
        bankroll: float,
    ) -> list[AgentRecommendation]:
        # MONK is prediction-first: at most one trade per fixture, the highest-EV.
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


# Compatibility name: the live and offline paths now share one Blitz clone.
from agents.legacy_blitz import LegacyBlitzStrategy


class MonkStrategy(LegacyBlitzStrategy):
    def __init__(self):
        super().__init__("monk")
