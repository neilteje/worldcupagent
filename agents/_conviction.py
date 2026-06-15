"""Shared conviction-betting logic for MONK / ANCHOR / HUNTER.

User directive: all four agents trade off the SAME goated council forecast (the
one grounded by BZZOIRO + web + Reddit + Grok) and back the genuine best-EV
outcome — favorites included. They differ ONLY by risk (edge bar, Kelly, stake
cap, how many bets per window), never by which side they are allowed to take.

Two hard rules kill the old pathologies:
  • A probability floor (``config.CONVICTION_MIN_PROB``) — never back an outcome
    the council gives less than ~12% real chance, no matter how juicy the price.
    This ends "10x odds on the shittiest team" payout-chasing.
  • Candidates must be +EV vs the executable price AND clear the profile's
    edge-vs-fair bar; we never divert to a reflexive draw without real value.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import config
from agents.contracts import AgentForecast, AgentDataView, MarketContext, TradeCandidate
from models.calibration import normalize_probs

_SLOTS = ("home", "draw", "away")


def council_probs(football_features: dict | None) -> dict | None:
    """Slot-keyed council probabilities ({home,draw,away}) if the shared council
    forecast is present on the data view, else None."""
    cf = (football_features or {}).get("council_forecast") or {}
    probs = cf.get("probabilities") or {}
    vals = {k: probs.get(k) for k in _SLOTS}
    if any(not isinstance(v, (int, float)) for v in vals.values()):
        return None
    return normalize_probs({k: max(0.0, float(v)) for k, v in vals.items()})


def build_council_forecast(
    view: AgentDataView, agent_name: str, model_version: str,
) -> AgentForecast | None:
    """Build an AgentForecast straight from the shared council belief.

    Returns None when no council forecast is attached, so each agent can fall
    back to its own independent model (keeps offline/unit paths working)."""
    p = council_probs(view.football_features)
    if p is None:
        return None

    cf = (view.football_features or {}).get("council_forecast") or {}
    confidence = float(cf.get("confidence") or round(max(p.values()), 4))
    coverage = float(view.data_coverage.get("overall", 1.0))

    # Band widens as confidence/coverage fall (used for the conservative-edge gate).
    half = 0.05 + (1.0 - confidence) * 0.10 + (1.0 - coverage) * 0.08
    lo = {k: max(0.0, p[k] - half) for k in _SLOTS}
    hi = {k: min(1.0, p[k] + half) for k in _SLOTS}

    forecast_id = hashlib.sha256(
        f"{view.data_view_hash}_{model_version}".encode()).hexdigest()
    return AgentForecast(
        agent_name=agent_name,
        fixture_id=view.fixture_id,
        window=view.window,
        as_of_timestamp=view.as_of_timestamp,
        home_probability=p["home"], draw_probability=p["draw"], away_probability=p["away"],
        home_lower_bound=lo["home"], draw_lower_bound=lo["draw"], away_lower_bound=lo["away"],
        home_upper_bound=hi["home"], draw_upper_bound=hi["draw"], away_upper_bound=hi["away"],
        confidence=round(confidence, 4),
        data_coverage_score=coverage,
        forecast_type="council_conviction",
        model_version=model_version,
        components={"source": "council", "confidence": confidence},
        evidence_ids=tuple(),
        warnings=tuple(),
        data_view_hash=view.data_view_hash,
        forecast_id=forecast_id,
    )


def conviction_candidates(
    forecast: AgentForecast,
    view: AgentDataView,
    market: MarketContext | None,
    profile,
    agent_name: str,
    signals_by_outcome: dict | None = None,
) -> list[TradeCandidate]:
    """Best-EV candidates across ALL three outcomes, conviction-style.

    Eligibility (every rule must pass):
      1. council probability ≥ CONVICTION_MIN_PROB   (no longshot payout-chasing)
      2. a usable executable price exists
      3. positive EV vs that price                    (we are paid to be right)
      4. edge vs the de-vigged fair price ≥ profile bar (risk differentiation)
    Ranked by EV after costs, so the agent backs what it most expects to win.
    """
    if not market:
        return []

    floor = config.CONVICTION_MIN_PROB
    fee, slip, mrisk = config.FEE_BUFFER, config.SLIPPAGE_BUFFER, config.MODEL_RISK_BUFFER
    signals_by_outcome = signals_by_outcome or {}
    out: list[TradeCandidate] = []

    for outcome in _SLOTS:
        prob = getattr(forecast, f"{outcome}_probability")
        if prob < floor:
            continue
        lower = getattr(forecast, f"{outcome}_lower_bound")
        fill = (market.expected_fill_price.get(outcome)
                or market.best_ask.get(outcome)
                or market.midpoint.get(outcome))
        if not fill:
            continue

        gross_edge = prob - float(fill)
        if gross_edge <= 0:
            continue

        fair = market.midpoint.get(outcome) or float(fill)
        if (prob - fair) < profile.min_edge_vs_fair:
            continue

        ev_after_costs = prob - float(fill) - fee - slip
        conservative_edge = lower - float(fill) - fee - slip - mrisk

        out.append(TradeCandidate(
            agent_name=agent_name,
            fixture_id=forecast.fixture_id,
            outcome=outcome,
            probability_mean=prob,
            probability_lower_bound=lower,
            probability_upper_bound=getattr(forecast, f"{outcome}_upper_bound"),
            market_midpoint=market.midpoint.get(outcome),
            best_ask=market.best_ask.get(outcome),
            expected_fill_price=float(fill),
            gross_edge=gross_edge,
            conservative_edge=conservative_edge,
            expected_value_after_costs=ev_after_costs,
            signal_type="council_conviction",
            signals=tuple(signals_by_outcome.get(outcome, ())),
            candidate_created_at=datetime.now(timezone.utc),
            candidate_expires_at=None,
            forecast_id=forecast.forecast_id,
            correlation_key=f"{agent_name}_{forecast.fixture_id}_{outcome}",
        ))

    return sorted(out, key=lambda c: c.expected_value_after_costs or -1, reverse=True)
