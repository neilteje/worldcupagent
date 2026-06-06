from __future__ import annotations

def bet_size(edge_tier: str, confidence: float, max_order_usd: float, consensus_modifier: float = 1.0, allow_soft: bool = False) -> float:
    if edge_tier == "none": return 0.0
    if edge_tier == "soft": base = 0.50 if allow_soft and confidence > .75 else 0.0
    elif edge_tier == "medium": base = 1.0 + max(0.0, confidence-.55)*5.0
    else: base = 2.5 + max(0.0, confidence-.55)*4.0
    return round(max(0.0, min(max_order_usd, base * consensus_modifier)), 2)

def limit_price(outcome: str, market_mid: float, model_probability: float) -> float:
    tolerance = 0.006 if outcome == "draw" else 0.010
    cap = model_probability - (0.025 if outcome == "draw" else 0.015)
    return round(max(0.01, min(0.99, market_mid + tolerance, cap)), 4)
