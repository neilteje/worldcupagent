from __future__ import annotations
from models.calibration import normalize_probs


def apply_draw_model(probs: dict[str, float], *, total_projected_xg: float | None = None, strength_gap: float | None = None, market_draw: float | None = None, bookmaker_draw: float | None = None, ht_score: tuple[int, int] | None = None, ht_total_xg: float | None = None, red_cards: tuple[int, int] | None = None) -> dict:
    p = normalize_probs(probs)
    delta = 0.0
    reasons: list[str] = []
    if total_projected_xg is not None:
        if total_projected_xg < 2.15:
            delta += 0.035; reasons.append("low projected xG")
        elif total_projected_xg > 3.05:
            delta -= 0.030; reasons.append("high projected xG")
    if strength_gap is not None:
        if strength_gap < 0.12:
            delta += 0.025; reasons.append("small strength gap")
        elif strength_gap > 0.32:
            delta -= 0.025; reasons.append("large favorite gap")
    if ht_score is not None:
        hg, ag = ht_score
        if hg == ag:
            delta += 0.045; reasons.append("level score state")
            if ht_total_xg is not None and ht_total_xg < 0.70:
                delta += 0.045; reasons.append("0-0/level low HT xG")
        else:
            delta -= 0.025; reasons.append("non-level score state")
    if market_draw is not None and bookmaker_draw is not None and min(market_draw, bookmaker_draw) > 0:
        ref = (market_draw + bookmaker_draw) / 2
        if p["draw"] < ref - 0.06:
            delta += min(0.025, (ref - p["draw"]) / 3); reasons.append("draw below market/book baseline")
    if red_cards and sum(red_cards) > 0 and ht_score and ht_score[0] == ht_score[1]:
        delta -= 0.015; reasons.append("level game with red card volatility")
    delta = max(-0.06, min(0.08, delta))
    if abs(delta) < 1e-9:
        return {"probabilities": p, "delta": {"home": 0.0, "draw": 0.0, "away": 0.0}, "reason": "No draw-model adjustment."}
    take_home = delta * (p["home"] / max(p["home"] + p["away"], 1e-9))
    take_away = delta - take_home
    adjusted = normalize_probs({"home": p["home"] - take_home, "draw": p["draw"] + delta, "away": p["away"] - take_away})
    return {"probabilities": adjusted, "delta": {k: adjusted[k] - p[k] for k in p}, "reason": "; ".join(reasons)}


def draw_sanity_flags(probs: dict[str, float], *, normal_soccer_fixture: bool = True, reason: str = "") -> list[str]:
    p = normalize_probs(probs)
    if normal_soccer_fixture and p["draw"] < 0.15 and "large favorite" not in reason.lower() and "red card" not in reason.lower():
        return ["draw_probability_requires_reason"]
    return []
