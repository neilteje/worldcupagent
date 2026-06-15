from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from agents.base import AgentStrategy, ExecutionPlan, AgentRecommendation
from agents.contracts import (
    AgentDataView,
    AgentForecast,
    DirectionalSignal,
    FixtureDataSnapshot,
    MarketContext,
    TradeCandidate,
)
from agents._conviction import build_council_forecast, conviction_candidates
from agents._reco import reco_from_candidate
from harness.profiles import get_profile

VALID_EVENT_TRIGGERS = {
    "confirmed_lineup_surprise",
    "late_injury_or_rotation",
    "stale_market_after_news",
    "cross_market_divergence",
    "dead_rubber_motivation",
    "halftime_xg_reversion",
    "red_card",
    "material_live_state_change",
}


def _parse_dt(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            out = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return out if out.tzinfo else out.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def valid_blitz_signals(view: AgentDataView, *, now: datetime | None = None) -> dict[str, list[DirectionalSignal]]:
    now = now or datetime.now(timezone.utc)
    out: dict[str, list[DirectionalSignal]] = {}
    for raw in (view.football_features or {}).get("event_signals") or ():
        trigger = str(raw.get("event_type") or raw.get("trigger") or "")
        observed = _parse_dt(raw.get("observed_at"))
        expires = _parse_dt(raw.get("expires_at"))
        outcome = str(raw.get("outcome") or raw.get("direction") or "")
        if trigger not in VALID_EVENT_TRIGGERS or not observed or not expires or expires <= now:
            continue
        signal = DirectionalSignal(
            signal_id=str(raw.get("signal_id") or raw.get("evidence_id") or hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()),
            fixture_id=view.fixture_id,
            outcome=outcome,
            source=str(raw.get("source") or "event_signal"),
            source_group=trigger,
            direction=outcome,
            strength=max(0.0, min(1.0, float(raw.get("strength", 0.5) or 0.5))),
            confidence=max(0.0, min(1.0, float(raw.get("confidence", 0.5) or 0.5))),
            observed_at=observed,
            expires_at=expires,
            evidence_hash=hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest(),
            summary=str(raw.get("summary") or trigger),
        )
        out.setdefault(outcome, []).append(signal)
    return out


class BlitzStrategy(AgentStrategy):
    name = "blitz"

    def __init__(self):
        self.profile = get_profile(self.name)

    def build_data_view(
        self,
        snapshot: FixtureDataSnapshot,
        prior_agent_forecast: AgentForecast | None = None,
    ) -> AgentDataView:
        coverage = float(((snapshot.football_context or {}).get("independent_forecast") or {}).get("data_coverage_score", 0.0))
        view_state = {
            "agent": self.name,
            "fixture_id": snapshot.fixture_id,
            "window": snapshot.window,
            "event_signals": (snapshot.football_context or {}).get("event_signals") or [],
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
            data_coverage={"overall": coverage},
            prohibited_fields_removed=tuple(),
            warnings=tuple(),
            data_view_hash=view_hash,
        )

    def build_forecast(self, view: AgentDataView) -> AgentForecast:
        council = build_council_forecast(view, self.name, "blitz_v1_event")
        if council is not None:
            return council
        ff = view.football_features or {}
        snap = ff.get("independent_forecast") or {}
        probs = snap.get("probabilities_by_code") or {}
        home = ff.get("home_code", "home")
        away = ff.get("away_code", "away")
        p = {
            "home": float(probs.get(home, 0.40)),
            "draw": float(probs.get("draw", 0.28)),
            "away": float(probs.get(away, 0.32)),
        }
        total = sum(p.values()) or 1.0
        p = {k: v / total for k, v in p.items()}
        forecast_id = hashlib.sha256(f"{view.data_view_hash}_blitz_v1_event".encode()).hexdigest()
        return AgentForecast(
            agent_name=self.name,
            fixture_id=view.fixture_id,
            window=view.window,
            as_of_timestamp=view.as_of_timestamp,
            home_probability=p["home"], draw_probability=p["draw"], away_probability=p["away"],
            home_lower_bound=max(0.0, p["home"] - 0.10),
            draw_lower_bound=max(0.0, p["draw"] - 0.10),
            away_lower_bound=max(0.0, p["away"] - 0.10),
            home_upper_bound=min(1.0, p["home"] + 0.10),
            draw_upper_bound=min(1.0, p["draw"] + 0.10),
            away_upper_bound=min(1.0, p["away"] + 0.10),
            confidence=float((ff.get("council_forecast") or {}).get("confidence", 0.5)),
            data_coverage_score=float(view.data_coverage.get("overall", 0.0)),
            forecast_type="event_triggered_common",
            model_version="blitz_v1_event",
            components={"source": "shared_forecast_contract"},
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
        signals = valid_blitz_signals(view, now=view.as_of_timestamp)
        if not signals:
            return []
        candidates = conviction_candidates(forecast, view, market, self.profile, self.name, signals)
        valid_outcomes = set(signals)
        return [
            TradeCandidate(
                agent_name=c.agent_name,
                fixture_id=c.fixture_id,
                outcome=c.outcome,
                probability_mean=c.probability_mean,
                probability_lower_bound=c.probability_lower_bound,
                probability_upper_bound=c.probability_upper_bound,
                market_midpoint=c.market_midpoint,
                best_ask=c.best_ask,
                expected_fill_price=c.expected_fill_price,
                gross_edge=c.gross_edge,
                conservative_edge=c.conservative_edge,
                expected_value_after_costs=c.expected_value_after_costs,
                signal_type="event_trigger",
                signals=c.signals,
                candidate_created_at=c.candidate_created_at,
                candidate_expires_at=min((s.expires_at for s in c.signals if s.expires_at), default=None),
                forecast_id=c.forecast_id,
                correlation_key=f"{forecast.fixture_id}:{c.outcome}:event:{','.join(sorted(s.signal_id for s in c.signals))}",
            )
            for c in candidates
            if c.outcome in valid_outcomes
        ]

    def generate_recommendations(
        self,
        candidates: list[TradeCandidate],
        forecast: AgentForecast,
        view: AgentDataView,
        market: MarketContext | None,
        bankroll: float,
    ) -> list[AgentRecommendation]:
        ranked = sorted(candidates, key=lambda c: c.expected_value_after_costs or -1, reverse=True)
        return [
            reco_from_candidate(self.name, cand, view, forecast, bankroll, self.profile)
            for cand in ranked[: self.profile.max_bets_per_window]
        ]

    def create_execution_plan(self, recommendations: list[AgentRecommendation]) -> ExecutionPlan:
        return ExecutionPlan(recommendations=tuple(recommendations))


__all__ = ["BlitzStrategy", "VALID_EVENT_TRIGGERS", "valid_blitz_signals"]
