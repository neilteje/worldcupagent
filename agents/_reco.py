"""Shared helpers for the coordinated agents (MONK / ANCHOR / HUNTER).

Each agent owns its candidate-generation logic but funnels selected candidates
through the SAME audited, gated recommendation path (``betting.recommendation.
build_recommendation``). The conservative-edge belief comes from the frozen
independent snapshot carried on the data view; the market price is used only to
measure edge, never to form the belief.
"""
from __future__ import annotations

from betting.policy import SizedPick
from betting.kelly import kelly_fraction
from betting.recommendation import build_recommendation
from models.forecast_contracts import AgentRecommendation


def code_for(view, outcome: str) -> str:
    ff = view.football_features or {}
    home = ff.get("home_code", "home")
    away = ff.get("away_code", "away")
    return {"home": home, "away": away, "draw": "draw"}.get(outcome, outcome)


def shared_snapshot(view, forecast) -> dict:
    """Belief snapshot (mean + lower/upper bands by code) for the conservative-edge gate.

    Conviction mode: when the shared council forecast is attached, the belief IS
    the council's (already baked into ``forecast``), so we build the snapshot from
    the agent forecast's own council-derived band — NOT the market-blind
    independent snapshot. Offline (no council) we prefer the frozen independent
    snapshot, falling back to the forecast band."""
    ff = view.football_features or {}
    council_active = bool((ff.get("council_forecast") or {}).get("probabilities"))
    if not council_active:
        snap = ff.get("independent_forecast")
        if snap and snap.get("probabilities_by_code"):
            return snap
    home = ff.get("home_code", "home")
    away = ff.get("away_code", "away")
    return {
        "probabilities_by_code": {
            home: forecast.home_probability, "draw": forecast.draw_probability,
            away: forecast.away_probability,
        },
        "lower_bounds_by_code": {
            home: forecast.home_lower_bound, "draw": forecast.draw_lower_bound,
            away: forecast.away_lower_bound,
        },
        "upper_bounds_by_code": {
            home: forecast.home_upper_bound, "draw": forecast.draw_upper_bound,
            away: forecast.away_upper_bound,
        },
    }


def reco_from_candidate(agent_name, cand, view, forecast, bankroll, profile) -> AgentRecommendation:
    code = code_for(view, cand.outcome)
    entry = cand.expected_fill_price or cand.market_midpoint or 0.5
    raw_kelly = max(0.0, kelly_fraction(cand.probability_mean, float(entry)))
    stake = round(min(
        bankroll * raw_kelly * profile.kelly_fraction,
        bankroll * profile.stake_cap_fraction,
        profile.max_bet_usd,
        bankroll,
    ), 2)
    if 0 < stake < 1.0 and profile.floor_to_min_order and bankroll >= 1.0:
        stake = 1.0
    pick = SizedPick(
        slot=cand.outcome, code=code, stake_usd=stake,
        entry_price=float(entry),
        limit_price=round(min(0.99, float(entry) + 0.02), 2),
        our_prob=cand.probability_mean, fair_prob=cand.market_midpoint,
        edge_vs_fair=cand.gross_edge or 0.0,
        ev_per_dollar=cand.expected_value_after_costs or 0.0,
        kelly_usd=stake,
    )
    snap = shared_snapshot(view, forecast)
    fid = (view.football_features or {}).get("forecast_snapshot_id") or forecast.forecast_id
    evidence_ids = [s.signal_id for s in cand.signals]
    if not evidence_ids:
        evidence_ids = list(getattr(forecast, "evidence_ids", ()) or [])
    if not evidence_ids:
        evidence_ids = list(((view.football_features or {}).get("evidence_ids") or []))
    return build_recommendation(
        agent_name, pick,
        fixture_id=cand.fixture_id, bankroll=bankroll,
        forecast_snapshot=snap, forecast_id=fid,
        evidence_ids=evidence_ids,
        data_coverage_score=forecast.data_coverage_score,
        confidence=forecast.confidence,
    )
