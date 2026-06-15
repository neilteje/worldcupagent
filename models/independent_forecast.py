"""Shared market-blind independent forecast builder.

MONK and ANCHOR both start from the SAME frozen football foundation (spec §8/§9:
"ANCHOR should start with the same frozen football foundation as MONK"). This
module turns the deterministic engine's per-team pre-match states into a 1X2
distribution with **market weight forced to zero** — no Polymarket, Kalshi,
bookmaker, or odds term ever enters. The BZZOIRO external model stays shadow-only
unless ``config.BZZOIRO_MODEL_SHADOW_ONLY`` is disabled and a weight is set.

The output also carries calibrated, coverage-scaled uncertainty bounds: the band
widens as data coverage drops (spec §18 — no single fixed interval width).

Reuses the production deterministic core (``models.deterministic_v2.predict_v2``,
``models.team_strength``, ``models.poisson_model``) rather than re-deriving it.
"""
from __future__ import annotations

from dataclasses import replace

import config
from models.calibration import OUTCOMES, normalize_probs
from models.deterministic_v2 import EnsembleConfig, predict_v2

# How wide the uncertainty band is at full coverage vs. zero coverage. The band
# half-width scales linearly between these as data_coverage_score moves 1 -> 0.
_MIN_HALF_WIDTH = 0.04   # full coverage, confident
_MAX_HALF_WIDTH = 0.18   # no coverage, very uncertain


def _coverage_scaled_bounds(probs: dict, coverage: float) -> tuple[dict, dict]:
    """Return (lower, upper) bands that widen as coverage drops."""
    cov = max(0.0, min(1.0, float(coverage)))
    half = _MIN_HALF_WIDTH + (1.0 - cov) * (_MAX_HALF_WIDTH - _MIN_HALF_WIDTH)
    lower = {k: max(0.0, probs[k] - half) for k in OUTCOMES}
    upper = {k: min(1.0, probs[k] + half) for k in OUTCOMES}
    return lower, upper


def _states_from_features(football_features: dict | None) -> tuple[dict | None, dict | None]:
    det = (football_features or {}).get("deterministic_model") or {}
    home_state = det.get("home_state")
    away_state = det.get("away_state")
    return home_state, away_state


def _snapshot_probs_hda(football_features: dict | None, home_code: str, away_code: str) -> dict | None:
    """Fallback: pull an independent snapshot's probabilities (keyed by code) and
    remap to home/draw/away. Used when no deterministic team-states are present
    (e.g. unit tests / degraded data)."""
    snap = (football_features or {}).get("independent_forecast") or {}
    by_code = snap.get("probabilities_by_code")
    if not by_code:
        return None
    return normalize_probs({
        "home": by_code.get(home_code),
        "draw": by_code.get("draw"),
        "away": by_code.get(away_code),
    })


def build_independent_forecast(
    football_features: dict | None,
    *,
    data_coverage_score: float = 1.0,
    w_elo: float = 0.5,
    w_poisson: float = 0.5,
    include_bzzoiro_shadow: bool = True,
    is_knockout: bool = False,
    match_week: int = 0,
) -> dict:
    """Build a market-blind 1X2 forecast + coverage-scaled bounds.

    Returns ``{probabilities, lower_bounds, upper_bounds, components, weights,
    expected_goals, forecast_type, warnings}`` with keys home/draw/away. Never
    consults any market field; ``use_market=False`` is hard-wired.
    """
    warnings: list[str] = []
    home_state, away_state = _states_from_features(football_features)

    if not home_state or not away_state:
        # Degraded path: fall back to an independent snapshot if present, else a
        # neutral prior. Bounds widen because coverage is effectively unknown.
        home_code = (football_features or {}).get("home_code", "home")
        away_code = (football_features or {}).get("away_code", "away")
        probs = _snapshot_probs_hda(football_features, home_code, away_code)
        if probs is None:
            probs = {"home": 0.40, "draw": 0.26, "away": 0.34}
            warnings.append("no_team_states_or_snapshot_neutral_prior")
        else:
            warnings.append("no_team_states_used_independent_snapshot")
        coverage = min(float(data_coverage_score), 0.5)
        lower, upper = _coverage_scaled_bounds(probs, coverage)
        return {
            "probabilities": probs,
            "lower_bounds": lower,
            "upper_bounds": upper,
            "components": {},
            "weights": {},
            "expected_goals": None,
            "forecast_type": "independent_deterministic",
            "warnings": tuple(warnings),
        }

    # BZZOIRO external model is shadow-only by default (spec §17): never blended
    # into the production forecast unless explicitly enabled with a weight.
    bz_probs = None
    w_bz = 0.0
    if include_bzzoiro_shadow and not config.BZZOIRO_MODEL_SHADOW_ONLY and config.BZZOIRO_MODEL_WEIGHT > 0:
        det = (football_features or {}).get("deterministic_model") or {}
        bz_probs = (det.get("components") or {}).get("bzzoiro")
        w_bz = config.BZZOIRO_MODEL_WEIGHT
    else:
        warnings.append("bzzoiro_model_shadow_only")

    cfg = EnsembleConfig(
        w_elo=w_elo,
        w_poisson=w_poisson,
        w_market=0.0,          # MARKET-BLIND: hard zero
        w_bzzoiro=w_bz,
        use_market=False,      # never form the independent forecast from market
        use_bzzoiro=bool(bz_probs),
    )

    out = predict_v2(
        home_state, away_state,
        market_probs=None,         # never pass market
        bzzoiro_probs=bz_probs,
        cfg=cfg,
        neutral=True,
        is_knockout=is_knockout,
        match_week=match_week,
    )
    probs = normalize_probs(out["probabilities"])
    lower, upper = _coverage_scaled_bounds(probs, data_coverage_score)
    return {
        "probabilities": probs,
        "lower_bounds": lower,
        "upper_bounds": upper,
        "components": out.get("components", {}),
        "weights": out.get("weights", {}),
        "expected_goals": out.get("expected_goals"),
        "forecast_type": "independent_deterministic",
        "warnings": tuple(warnings),
    }


__all__ = ["build_independent_forecast"]
