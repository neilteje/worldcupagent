"""Context adjustments (spec §15).

Small, capped multiplicative tilts to an expected-goal rate for opponent quality,
rest, and travel. Each effect is bounded so no single context feature dominates
the structural form/Elo signal.
"""
from __future__ import annotations


class ContextAdjustments:
    def __init__(self, max_tilt: float = 0.20):
        self.max_tilt = max_tilt

    def apply_adjustments(self, base_xg: float, opponent_quality: float = 0.0,
                          rest_days: float = 3.0) -> float:
        """Adjust ``base_xg`` for opponent strength (higher quality opponent ->
        fewer expected goals) and short rest (fatigue -> fewer). ``opponent_quality``
        is a scaled rating in roughly [-2.5, 2.5]."""
        tilt = 0.0
        tilt -= 0.06 * max(-2.5, min(2.5, opponent_quality))   # tougher opponent suppresses xG
        if rest_days < 3.0:
            tilt -= 0.05 * (3.0 - rest_days)                   # fatigue
        tilt = max(-self.max_tilt, min(self.max_tilt, tilt))
        return max(0.05, base_xg * (1.0 + tilt))
