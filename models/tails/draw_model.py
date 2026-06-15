"""Specialized draw model for HUNTER (spec §10).

Draw probability is grounded in the Poisson/Dixon-Coles score model — NOT a fixed
multiplicative bump. Context features (defensive quality, failure-to-score, group
incentives, knockout 90-minute dynamics, tactical pace) apply small, capped tilts
on top of the structural draw mass.
"""
from __future__ import annotations

from models.poisson_model import poisson_1x2

# Hard cap on how far context can move the structural draw probability, so one
# soft feature can never dominate the score model.
_MAX_CONTEXT_TILT = 0.10


class DrawModel:
    def __init__(self, max_context_tilt: float = _MAX_CONTEXT_TILT):
        self.max_context_tilt = max_context_tilt

    def structural_draw(self, lam_home: float, lam_away: float) -> float:
        """Draw mass straight from the Dixon-Coles corrected score matrix."""
        return poisson_1x2(lam_home, lam_away)["draw"]

    def _context_tilt(self, context: dict) -> float:
        ctx = context or {}
        tilt = 0.0
        # Two strong defenses / low expected total -> more draws.
        total = ctx.get("expected_total_goals")
        if total is not None:
            if total < 2.2:
                tilt += 0.04
            elif total > 3.0:
                tilt -= 0.04
        # Both teams frequently fail to score.
        fts = ctx.get("both_fail_to_score_rate")
        if fts is not None and fts > 0.30:
            tilt += 0.03
        # Group stage where a draw advances/suffices both sides.
        if ctx.get("draw_is_enough"):
            tilt += 0.05
        # Knockout: a level-after-90 still counts as a "draw" outcome at the
        # 90-minute market line, so the 90' draw is underpriced.
        if ctx.get("is_knockout"):
            tilt += 0.03
        # High tactical pace / open game -> fewer draws.
        if ctx.get("high_pace"):
            tilt -= 0.03
        return max(-self.max_context_tilt, min(self.max_context_tilt, tilt))

    def probability(self, lam_home: float, lam_away: float, context: dict | None = None) -> float:
        base = self.structural_draw(lam_home, lam_away)
        p = base + self._context_tilt(context or {})
        return max(0.02, min(0.70, p))

    # Back-compat shim for the old call site.
    def calculate_probability(self, base_poisson_draw: float, context: dict | None = None) -> float:
        p = float(base_poisson_draw) + self._context_tilt(context or {})
        return max(0.02, min(0.70, p))
