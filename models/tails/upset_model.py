"""Specialized underdog-upset model for HUNTER (spec §10).

The underdog's win probability comes from the Poisson/Dixon-Coles score model
(so it reflects the actual goal distribution, not a fixed bump), then context
features — favorite rotation, goalkeeper absence, rest/travel mismatch, set-piece
edge, favorite overperformance vs xG, underdog defensive resilience — apply small
capped tilts. Positive skew, not merely cheap price, is the product.
"""
from __future__ import annotations

from models.poisson_model import poisson_1x2

_MAX_CONTEXT_TILT = 0.10


class UnderdogUpsetModel:
    def __init__(self, max_context_tilt: float = _MAX_CONTEXT_TILT):
        self.max_context_tilt = max_context_tilt

    def structural_upset(self, lam_home: float, lam_away: float, underdog: str) -> float:
        """Win mass for the underdog side from the score matrix.

        ``underdog`` is "home" or "away" — whichever side is the dog.
        """
        p = poisson_1x2(lam_home, lam_away)
        return p["away"] if underdog == "away" else p["home"]

    def _context_tilt(self, context: dict) -> float:
        ctx = context or {}
        tilt = 0.0
        if ctx.get("favorite_rotation"):          # favorite resting starters
            tilt += 0.04
        if ctx.get("favorite_keeper_absent"):     # first-choice GK out
            tilt += 0.04
        if ctx.get("underdog_rest_edge"):         # dog fresher / less travel
            tilt += 0.02
        if ctx.get("underdog_set_piece_edge"):
            tilt += 0.02
        if ctx.get("favorite_overperforming_xg"): # favorite due regression
            tilt += 0.03
        if ctx.get("underdog_defensive_resilience"):
            tilt += 0.02
        return max(-self.max_context_tilt, min(self.max_context_tilt, tilt))

    def probability(self, lam_home: float, lam_away: float, underdog: str,
                    context: dict | None = None) -> float:
        base = self.structural_upset(lam_home, lam_away, underdog)
        p = base + self._context_tilt(context or {})
        return max(0.02, min(0.80, p))

    # Back-compat shim.
    def calculate_probability(self, base_poisson_upset: float, context: dict | None = None) -> float:
        p = float(base_poisson_upset) + self._context_tilt(context or {})
        return max(0.02, min(0.80, p))
