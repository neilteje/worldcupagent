from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

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

_TRIGGER_KEYWORDS = (
    ("red card", "red_card"),
    ("lineup", "confirmed_lineup_surprise"),
    ("starting xi", "confirmed_lineup_surprise"),
    ("injury", "late_injury_or_rotation"),
    ("rotation", "late_injury_or_rotation"),
    ("stale market", "stale_market_after_news"),
    ("cross-market", "cross_market_divergence"),
    ("divergence", "cross_market_divergence"),
    ("dead rubber", "dead_rubber_motivation"),
    ("motivation", "dead_rubber_motivation"),
    ("xg reversion", "halftime_xg_reversion"),
    ("live state", "material_live_state_change"),
)


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
    for raw in coerce_event_signals((view.football_features or {}).get("event_signals") or (),
                                    fixture_id=view.fixture_id,
                                    observed_at=now,
                                    home_code=(view.football_features or {}).get("home_code"),
                                    away_code=(view.football_features or {}).get("away_code")):
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


def _trigger_from_text(text: str) -> str | None:
    lower = text.lower()
    for needle, trigger in _TRIGGER_KEYWORDS:
        if needle in lower:
            return trigger
    return None


def coerce_event_signals(raw_flags, *, fixture_id: str, observed_at: datetime,
                         home_code: str | None = None,
                         away_code: str | None = None) -> list[dict]:
    """Convert Scout/live flags into timestamped BLITZ event signals."""
    out = []
    for raw in raw_flags or ():
        if not isinstance(raw, dict):
            continue
        observed = _parse_dt(raw.get("observed_at")) or observed_at
        expires = _parse_dt(raw.get("expires_at"))
        signal_text = " ".join(str(raw.get(k) or "") for k in ("signal", "summary", "rationale", "event_type", "trigger"))
        trigger = str(raw.get("event_type") or raw.get("trigger") or "") or (_trigger_from_text(signal_text) or "")
        if trigger not in VALID_EVENT_TRIGGERS:
            continue
        if expires is None:
            expires = observed + timedelta(minutes=30 if trigger in {"red_card", "material_live_state_change", "halftime_xg_reversion"} else 90)
        direction = str(raw.get("outcome") or raw.get("direction") or "").lower()
        team = str(raw.get("team") or "").upper()
        direction_code = str(raw.get("direction") or raw.get("outcome") or "").upper()
        outcome = "draw" if direction == "draw" else (
            "home" if "home" in direction or direction_code == str(home_code or "").upper() or team == str(home_code or "").upper() else (
                "away" if "away" in direction or direction_code == str(away_code or "").upper() or team == str(away_code or "").upper() else ""
            )
        )
        if outcome not in {"home", "draw", "away"}:
            continue
        out.append({
            **raw,
            "signal_id": raw.get("signal_id") or raw.get("evidence_id") or hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest(),
            "fixture_id": raw.get("fixture_id") or fixture_id,
            "event_type": trigger,
            "outcome": outcome,
            "observed_at": observed,
            "expires_at": expires,
            "summary": raw.get("summary") or raw.get("signal") or trigger,
        })
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
        candidates = conviction_candidates(forecast, view, market, self.profile, self.name, signals)
        valid_outcomes = set(signals) if signals else {"home", "draw", "away"}
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
                signal_type="event_trigger" if signals else "blitz_value",
                signals=c.signals,
                candidate_created_at=c.candidate_created_at,
                candidate_expires_at=min((s.expires_at for s in c.signals if s.expires_at), default=None),
                forecast_id=c.forecast_id,
                correlation_key=(
                    f"{forecast.fixture_id}:{c.outcome}:event:{','.join(sorted(s.signal_id for s in c.signals))}"
                    if signals else f"{forecast.fixture_id}:{c.outcome}:blitz_value:{forecast.forecast_id}"
                ),
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


__all__ = ["BlitzStrategy", "VALID_EVENT_TRIGGERS", "coerce_event_signals", "valid_blitz_signals"]
