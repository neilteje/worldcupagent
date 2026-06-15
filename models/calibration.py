from __future__ import annotations
import math
OUTCOMES = ("home", "draw", "away")


def normalize_probs(probs: dict[str, float] | None) -> dict[str, float]:
    vals = {k: float((probs or {}).get(k, 0.0) or 0.0) for k in OUTCOMES}
    vals = {k: (v if math.isfinite(v) and v > 0 else 0.0) for k, v in vals.items()}
    total = sum(vals.values())
    if total <= 0:
        return {"home": 1/3, "draw": 1/3, "away": 1/3}
    return {k: v / total for k, v in vals.items()}


def clamp_probs(probs: dict[str, float], min_prob: float = 0.02, max_prob: float = 0.92) -> dict[str, float]:
    return normalize_probs({k: max(min_prob, min(max_prob, float(probs.get(k, 0)))) for k in OUTCOMES})


def temperature_scale(probs: dict[str, float], temp: float = 1.05) -> dict[str, float]:
    p = clamp_probs(probs, 1e-6, 0.999999)
    if temp <= 0:
        temp = 1.0
    logits = {k: math.log(v) / temp for k, v in p.items()}
    m = max(logits.values())
    exps = {k: math.exp(v - m) for k, v in logits.items()}
    return normalize_probs(exps)


def shrink_toward_market(probs: dict[str, float], market_probs: dict[str, float] | None, strength: float = 0.08) -> dict[str, float]:
    if not market_probs:
        return normalize_probs(probs)
    model = normalize_probs(probs)
    market = normalize_probs(market_probs)
    s = max(0.0, min(1.0, strength))
    return normalize_probs({k: model[k] * (1 - s) + market[k] * s for k in OUTCOMES})


def calibrate_probs(
    probs: dict[str, float],
    *,
    market_probs: dict[str, float] | None = None,
    temperature: float = 1.06,
    shrink_strength: float = 0.06,
    draw_floor: float = 0.16,
    max_prob: float = 0.86,
    draw_floor_reason: str | None = None,
) -> dict:
    risk_flags: list[str] = []
    calibrated = normalize_probs(probs)
    calibrated = clamp_probs(calibrated, min_prob=0.025, max_prob=max_prob)
    calibrated = temperature_scale(calibrated, temperature)
    if market_probs:
        calibrated = shrink_toward_market(calibrated, market_probs, shrink_strength)
    if max(calibrated.values()) > max_prob:
        risk_flags.append("prediction_overconfidence_clamped")
        calibrated = clamp_probs(calibrated, min_prob=0.025, max_prob=max_prob)
    if calibrated["draw"] < draw_floor and not _clear_low_draw_reason(draw_floor_reason):
        risk_flags.append("draw_floor_applied")
        shortfall = draw_floor - calibrated["draw"]
        home_away = calibrated["home"] + calibrated["away"]
        calibrated = normalize_probs(
            {
                "home": calibrated["home"] - shortfall * calibrated["home"] / max(home_away, 1e-9),
                "draw": draw_floor,
                "away": calibrated["away"] - shortfall * calibrated["away"] / max(home_away, 1e-9),
            }
        )
    return {"probabilities": normalize_probs(calibrated), "risk_flags": list(dict.fromkeys(risk_flags)), "temperature": temperature, "shrink_strength": shrink_strength, "draw_floor": draw_floor}


def valid_probs(probs: dict[str, float], lo: float = 0.02, hi: float = 0.92, tol: float = 0.015) -> bool:
    if set(OUTCOMES) - set(probs):
        return False
    vals = [float(probs[k]) for k in OUTCOMES]
    return all(math.isfinite(v) and lo <= v <= hi for v in vals) and abs(sum(vals) - 1.0) <= tol


def _clear_low_draw_reason(reason: str | None) -> bool:
    text = (reason or "").lower()
    return any(marker in text for marker in ("large favorite", "red card volatility", "must win late", "extreme mismatch"))
