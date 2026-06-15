"""Deterministic halftime forecasting model (spec §20).

Produces a full 1X2 distribution at HT by:
  1. scaling pre-match expected goals to the remaining minutes,
  2. applying calibrated, CAPPED live multipliers (red cards, live-xG rate,
     score-state pressure),
  3. convolving the CURRENT score with the remaining-goal Poisson/Dixon-Coles
     distribution.

The LLM never produces the numeric update (spec §20). When live data are
insufficient the pre-match forecast is preserved, bounds widen, and the warning
``insufficient_live_evidence`` is emitted so coordinated agents abstain from new
HT risk.
"""
from __future__ import annotations

from models.calibration import OUTCOMES, normalize_probs
from models.live_state import LiveMatchState
from models.poisson_model import poisson_1x2, score_matrix

INSUFFICIENT_LIVE_EVIDENCE = "insufficient_live_evidence"

# Multiplier caps so no single live signal can dominate the structural model.
_RED_CARD_DOWN = 0.65
_RED_CARD_UP = 1.30
_XG_CAP_LOW = 0.70
_XG_CAP_HIGH = 1.40


def _bounds(probs: dict, half: float) -> tuple[dict, dict]:
    lo = {k: max(0.0, probs[k] - half) for k in OUTCOMES}
    hi = {k: min(1.0, probs[k] + half) for k in OUTCOMES}
    return lo, hi


class DeterministicHalftimeModel:
    """Deterministic Halftime Forecasting Model."""

    def __init__(self, min_coverage: float = 0.5):
        self.min_coverage = min_coverage

    def _live_multipliers(self, live: LiveMatchState, rem_lam_h: float, rem_lam_a: float) -> tuple[float, float]:
        mh = mw = 1.0
        # Red cards: the down-a-man side scores less, concedes more.
        if live.home_red_cards > live.away_red_cards:
            mh *= _RED_CARD_DOWN
            mw *= _RED_CARD_UP
        elif live.away_red_cards > live.home_red_cards:
            mw *= _RED_CARD_DOWN
            mh *= _RED_CARD_UP

        # Live-xG rate vs the pre-match remaining-rate expectation: a side
        # generating chances above expectation gets a bounded boost.
        minutes_played = max(1, live.current_minute)
        if live.home_live_xg is not None and rem_lam_h > 0:
            rate = (live.home_live_xg / minutes_played) * 45.0   # per-half rate
            ratio = rate / max(0.1, rem_lam_h)
            mh *= max(_XG_CAP_LOW, min(_XG_CAP_HIGH, ratio if ratio > 0 else 1.0)) ** 0.5
        if live.away_live_xg is not None and rem_lam_a > 0:
            rate = (live.away_live_xg / minutes_played) * 45.0
            ratio = rate / max(0.1, rem_lam_a)
            mw *= max(_XG_CAP_LOW, min(_XG_CAP_HIGH, ratio if ratio > 0 else 1.0)) ** 0.5

        mh = max(_XG_CAP_LOW, min(_XG_CAP_HIGH, mh))
        mw = max(_XG_CAP_LOW, min(_XG_CAP_HIGH, mw))
        return mh, mw

    def update_forecast(
        self,
        prematch_home_lambda: float,
        prematch_away_lambda: float,
        live_state: LiveMatchState,
        prematch_probs: dict | None = None,
    ) -> dict:
        """Return a full 1X2 distribution with bounds and coverage metadata."""
        coverage = float((live_state.data_coverage or {}).get("overall", 0.0))

        # ── Insufficient live data: preserve pre-match, widen bounds ──────────
        if coverage < self.min_coverage:
            base = normalize_probs(
                prematch_probs or poisson_1x2(prematch_home_lambda, prematch_away_lambda)
            )
            lo, hi = _bounds(base, 0.20)
            return {
                "probabilities": base,
                "lower_bounds": lo,
                "upper_bounds": hi,
                "home_win": base["home"], "draw": base["draw"], "away_win": base["away"],
                "insufficient_evidence": True,
                "warnings": (INSUFFICIENT_LIVE_EVIDENCE,),
                "data_coverage": coverage,
                "remaining_minutes": max(0, 90 - live_state.current_minute),
            }

        remaining_minutes = max(0, 90 - live_state.current_minute)
        frac = remaining_minutes / 90.0
        mh, mw = self._live_multipliers(live_state, prematch_home_lambda * frac,
                                        prematch_away_lambda * frac)
        rem_lam_h = max(0.01, prematch_home_lambda * frac * mh)
        rem_lam_a = max(0.01, prematch_away_lambda * frac * mw)

        # Convolve current score with the remaining-goal distribution.
        matrix = score_matrix(rem_lam_h, rem_lam_a)
        home = draw = away = 0.0
        for gh, row in enumerate(matrix):
            for ga, p in enumerate(row):
                hf = live_state.home_score + gh
                af = live_state.away_score + ga
                if hf > af:
                    home += p
                elif hf == af:
                    draw += p
                else:
                    away += p
        probs = normalize_probs({"home": home, "draw": draw, "away": away})

        # Bounds tighten as more of the match is complete (less uncertainty left).
        half = 0.05 + 0.10 * frac
        lo, hi = _bounds(probs, half)
        return {
            "probabilities": probs,
            "lower_bounds": lo,
            "upper_bounds": hi,
            "home_win": probs["home"], "draw": probs["draw"], "away_win": probs["away"],
            "insufficient_evidence": False,
            "warnings": tuple(),
            "data_coverage": coverage,
            "remaining_minutes": remaining_minutes,
            "remaining_lambda_home": round(rem_lam_h, 4),
            "remaining_lambda_away": round(rem_lam_a, 4),
            "live_multiplier_home": round(mh, 4),
            "live_multiplier_away": round(mw, 4),
        }
