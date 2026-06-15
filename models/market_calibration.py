"""Single deterministic market-calibration step."""
from __future__ import annotations

from models.calibration import OUTCOMES, normalize_probs


def normalize_market_probabilities(market: dict | None) -> dict[str, float] | None:
    if not market:
        return None
    vals = {}
    for k in OUTCOMES:
        v = market.get(k)
        if not isinstance(v, (int, float)) or v < 0:
            return None
        vals[k] = float(v)
    if sum(vals.values()) <= 0:
        return None
    return normalize_probs(vals)


def clamp_market_weight(value: float | int | None) -> float:
    try:
        return max(0.0, min(0.60, float(value)))
    except (TypeError, ValueError):
        return 0.0


def apply_market_calibration(pre_market: dict[str, float],
                             market: dict[str, float] | None,
                             weight: float | int | None) -> dict[str, float]:
    """Mix market into a pre-market belief exactly once."""
    base = normalize_probs(pre_market)
    mkt = normalize_market_probabilities(market)
    w = clamp_market_weight(weight) if mkt else 0.0
    if w <= 0.0:
        return base
    return normalize_probs({k: base[k] * (1.0 - w) + mkt[k] * w for k in OUTCOMES})


def information_edge(pre_market_probability: float, expected_fill_price: float) -> float:
    return float(pre_market_probability) - float(expected_fill_price)


__all__ = [
    "apply_market_calibration",
    "clamp_market_weight",
    "information_edge",
    "normalize_market_probabilities",
]
