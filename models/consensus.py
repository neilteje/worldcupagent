from __future__ import annotations
from models.calibration import OUTCOMES, normalize_probs


def _usable(probs: dict[str, float] | None) -> bool:
    return bool(probs) and sum(float(probs.get(k, 0) or 0) for k in OUTCOMES) > 0


def _pick(probs: dict[str, float] | None) -> str | None:
    if not _usable(probs):
        return None
    p = normalize_probs(probs)
    return max(OUTCOMES, key=lambda k: p[k])


def _agree(a: dict[str, float], b: dict[str, float]) -> bool:
    ap, bp = _pick(a), _pick(b)
    if ap is None or bp is None or ap != bp:
        return False
    an, bn = normalize_probs(a), normalize_probs(b)
    return abs(an[ap] - bn[bp]) <= 0.08


def consensus_triangle(model_probs: dict[str, float], bookmaker_probs: dict[str, float] | None, market_probs: dict[str, float] | None) -> dict:
    model = normalize_probs(model_probs)
    market = normalize_probs(market_probs) if _usable(market_probs) else None
    bookmaker = normalize_probs(bookmaker_probs) if _usable(bookmaker_probs) else None
    mp, kp, bp = _pick(model), _pick(market), _pick(bookmaker)
    if bookmaker is None:
        return {"case": "missing_bookmaker", "model_pick": mp, "bookmaker_pick": None, "market_pick": kp, "agreement_score": 0.35, "confidence_modifier": -0.05, "bet_size_modifier": 0.75, "reason": "Bookmaker probabilities are missing or unusable."}
    mb = _agree(model, bookmaker)
    mk = bool(market) and _agree(model, market)
    bk = bool(market) and _agree(bookmaker, market)
    if mp == bp == kp:
        case, cm, bm, score = "all_agree", 0.05, 1.0, 1.0
    elif mb and kp != mp:
        case, cm, bm, score = "model_bookmaker_vs_polymarket", 0.08, 1.25, 0.75
    elif mk and bp != mp:
        case, cm, bm, score = "model_polymarket_vs_bookmaker", 0.0, 0.75, 0.62
    elif bk and mp != bp:
        case, cm, bm, score = "bookmaker_polymarket_vs_model", -0.15, 0.25, 0.35
    else:
        case, cm, bm, score = "all_disagree", -0.20, 0.10, 0.15
    return {"case": case, "model_pick": mp, "bookmaker_pick": bp, "market_pick": kp, "agreement_score": score, "confidence_modifier": cm, "bet_size_modifier": bm, "reason": f"model={mp}, bookmaker={bp}, market={kp}; case={case}."}
