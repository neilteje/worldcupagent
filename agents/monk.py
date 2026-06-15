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
        # MONK's purest forecast: market-blind Elo + Poisson (+ BZZOIRO shadow),
        # NEVER any market term. Bounds widen as data coverage drops.
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
        if not market:
            return []

        candidates = []
        for outcome in ("home", "draw", "away"):
            # MONK evaluates exceptional edges
            prob = getattr(forecast, f"{outcome}_probability")
            midpoint = market.midpoint.get(outcome)
            
            if midpoint is not None and (prob - midpoint) >= self.profile.min_edge_vs_fair:
                candidates.append(
                    TradeCandidate(
                        agent_name=self.name,
                        fixture_id=forecast.fixture_id,
                        outcome=outcome,
                        probability_mean=prob,
                        probability_lower_bound=getattr(forecast, f"{outcome}_lower_bound"),
                        probability_upper_bound=getattr(forecast, f"{outcome}_upper_bound"),
                        market_midpoint=midpoint,
                        best_ask=market.best_ask.get(outcome),
                        expected_fill_price=market.expected_fill_price.get(outcome) or midpoint,
                        gross_edge=prob - midpoint,
                        conservative_edge=getattr(forecast, f"{outcome}_lower_bound") - midpoint,
                        expected_value_after_costs=prob - midpoint - 0.01, # Simplified
                        signal_type="exceptional_value",
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
        # MONK is prediction-first: at most one exceptional-value trade per fixture.
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
