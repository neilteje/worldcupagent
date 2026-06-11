"""
De-vigging (overround removal) for bookmaker 1X2 odds.

The naive method — normalize 1/odds so they sum to 1 — distributes the
bookmaker's margin proportionally and therefore preserves the favorite-longshot
bias (favorites end up over-priced, longshots under-priced). The **power method**
solves for k such that sum(p_i**k) = 1, which deflates favorites more than
longshots and produces better-calibrated probabilities. This is a standard,
principled correction — it uses no match outcomes, so it is not fit to 2022.
"""
from __future__ import annotations

from models.calibration import OUTCOMES, normalize_probs


def _raw_implied(odds: dict[str, float]) -> dict[str, float]:
    return {k: 1.0 / max(float(odds[k]), 1.01) for k in OUTCOMES}


def basic_devig(odds: dict[str, float]) -> dict[str, float]:
    """Proportional normalization of 1/odds (preserves favorite-longshot bias)."""
    return normalize_probs(_raw_implied(odds))


def power_devig(odds: dict[str, float], *, iterations: int = 60) -> dict[str, float]:
    """Power method: find k with sum(p_i**k)=1; deflates favorites, corrects bias."""
    raw = _raw_implied(odds)
    vals = list(raw.values())
    # Overround > 1 -> need k > 1 to shrink the sum to 1 (and vice versa).
    lo, hi = 0.3, 4.0
    for _ in range(iterations):
        k = (lo + hi) / 2.0
        total = sum(v ** k for v in vals)
        if total > 1.0:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2.0
    total = sum(v ** k for v in vals) or 1.0
    return normalize_probs({o: raw[o] ** k / total for o in OUTCOMES})


def devig(odds: dict[str, float] | None, method: str = "power") -> dict[str, float] | None:
    if not odds:
        return None
    try:
        return power_devig(odds) if method == "power" else basic_devig(odds)
    except Exception:
        return None


__all__ = ["basic_devig", "power_devig", "devig"]
