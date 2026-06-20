"""Shared legacy Blitz/SURGE policy for every live wallet identity."""
from __future__ import annotations

import hashlib
import json

from agents.base import AgentStrategy, ExecutionPlan
from agents.contracts import AgentDataView, AgentForecast, FixtureDataSnapshot, MarketContext, TradeCandidate
from agents._conviction import build_council_forecast, conviction_candidates
from agents._reco import reco_from_candidate
from harness.profiles import get_profile
from models.independent_forecast import build_independent_forecast
from reasoning.market_blind import scrub_market_fields


class LegacyBlitzStrategy(AgentStrategy):
    """One forecast/value policy, parameterized only by wallet identity."""

    def __init__(self, agent_name: str):
        self.name = agent_name.strip().lower()
        self.profile = get_profile(self.name)

    def build_data_view(self, snapshot: FixtureDataSnapshot,
                        prior_agent_forecast: AgentForecast | None = None) -> AgentDataView:
        clean_features, removed = scrub_market_fields(snapshot.football_context)
        clean_features.pop("bzzoiro_digest", None)
        clean_features["evidence_ids"] = [
            item for item in (clean_features.get("evidence_ids") or [])
            if "bzzoiro" not in str(item).lower()
        ]
        common = {
            "fixture_id": snapshot.fixture_id, "window": snapshot.window,
            "snapshot_id": snapshot.snapshot_id, "snapshot_hash": snapshot.snapshot_hash,
            "football_context": clean_features, "live_context": snapshot.live_context,
        }
        view_hash = hashlib.sha256(json.dumps(common, sort_keys=True, default=str).encode()).hexdigest()
        return AgentDataView(
            agent_name=self.name, fixture_id=snapshot.fixture_id, window=snapshot.window,
            as_of_timestamp=snapshot.as_of_timestamp, football_features=clean_features,
            live_features=snapshot.live_context, external_model_predictions={}, evidence=tuple(),
            market_features=None, data_coverage=_coverage(clean_features),
            prohibited_fields_removed=removed, warnings=tuple(), data_view_hash=view_hash,
        )

    def build_forecast(self, view: AgentDataView) -> AgentForecast:
        council = build_council_forecast(view, self.name, "legacy_blitz_shared_v1")
        if council is not None:
            return council
        coverage = float(view.data_coverage.get("overall", 0.0) or 0.0)
        out = build_independent_forecast(view.football_features, data_coverage_score=coverage)
        p, lo, hi = out["probabilities"], out["lower_bounds"], out["upper_bounds"]
        forecast_id = hashlib.sha256(
            f"{view.data_view_hash}:legacy_blitz_fallback_v1".encode()
        ).hexdigest()
        return AgentForecast(
            agent_name=self.name, fixture_id=view.fixture_id, window=view.window,
            as_of_timestamp=view.as_of_timestamp,
            home_probability=p["home"], draw_probability=p["draw"], away_probability=p["away"],
            home_lower_bound=lo["home"], draw_lower_bound=lo["draw"], away_lower_bound=lo["away"],
            home_upper_bound=hi["home"], draw_upper_bound=hi["draw"], away_upper_bound=hi["away"],
            confidence=round(max(p.values()), 4), data_coverage_score=coverage,
            forecast_type="legacy_blitz_fallback", model_version="legacy_blitz_fallback_v1",
            components=out.get("components", {}),
            evidence_ids=tuple((view.football_features or {}).get("evidence_ids") or ()),
            warnings=tuple(out.get("warnings") or ()), data_view_hash=view.data_view_hash,
            forecast_id=forecast_id,
        )

    def generate_candidates(self, forecast: AgentForecast, view: AgentDataView,
                            market: MarketContext | None) -> list[TradeCandidate]:
        if forecast.confidence < self.profile.min_confidence:
            return []
        if view.window == "PRE_MATCH" and not self.profile.trade_prematch:
            return []
        if view.window == "HT" and not self.profile.trade_halftime:
            return []
        return conviction_candidates(forecast, view, market, self.profile, self.name)

    def generate_recommendations(self, candidates, forecast, view, market, bankroll):
        ranked = sorted(
            candidates,
            key=lambda c: c.expected_value_after_costs
            if c.expected_value_after_costs is not None else float("-inf"),
            reverse=True,
        )
        return [reco_from_candidate(self.name, candidate, view, forecast, bankroll, self.profile)
                for candidate in ranked[:self.profile.max_bets_per_window]]

    def create_execution_plan(self, recommendations) -> ExecutionPlan:
        return ExecutionPlan(recommendations=tuple(recommendations))


def _coverage(features: dict | None) -> dict:
    ff = features or {}
    present = (bool(ff.get("sportmonks_digest")), bool(ff.get("supabase_digest")),
               bool(ff.get("bzzoiro_digest")), bool(ff.get("evidence_ids")))
    return {"overall": sum(present) / len(present)}


__all__ = ["LegacyBlitzStrategy"]
