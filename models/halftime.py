from __future__ import annotations
from models.calibration import clamp_probs, normalize_probs, temperature_scale


def _num(d: dict, *keys: str, default: float | None = 0.0) -> float | None:
    for k in keys:
        if k in d and d[k] is not None:
            try: return float(d[k])
            except (TypeError, ValueError): pass
    return default


def evaluate_halftime(pre_match_probs: dict[str, float], market_probs_ht: dict[str, float] | None = None, live_checkpoint: dict | None = None, score: dict | None = None, xg: dict | None = None, shots: dict | None = None, shots_on_target: dict | None = None, cards: dict | None = None, substitutions: dict | None = None) -> dict:
    live = live_checkpoint or {}
    score = score or live.get("score") or live
    xg = xg if xg is not None else (live.get("xg") if isinstance(live.get("xg"), dict) else live)
    shots = shots or live.get("shots") or live
    shots_on_target = shots_on_target or live.get("shots_on_target") or live
    cards = cards or live.get("cards") or live
    hg, ag = int(_num(score, "home_goals", "home_score", default=0) or 0), int(_num(score, "away_goals", "away_score", default=0) or 0)
    hxg = _num(xg or {}, "home_xg", "home_expected_goals", default=None)
    axg = _num(xg or {}, "away_xg", "away_expected_goals", default=None)
    has_xg = hxg is not None and axg is not None
    hxg = float(hxg or 0.0); axg = float(axg or 0.0)
    hs, aws = _num(shots, "home_shots", default=0) or 0, _num(shots, "away_shots", default=0) or 0
    hsot, asot = _num(shots_on_target, "home_sot", "home_shots_on_target", default=0) or 0, _num(shots_on_target, "away_sot", "away_shots_on_target", default=0) or 0
    hr, ar = int(_num(cards, "home_red", "home_red_cards", default=0) or 0), int(_num(cards, "away_red", "away_red_cards", default=0) or 0)
    p = normalize_probs(pre_match_probs).copy()
    gd = hg - ag
    if gd == 1: p["home"] += .18; p["draw"] -= .08; p["away"] -= .10
    elif gd >= 2: p["home"] += .35; p["draw"] -= .15; p["away"] -= .20
    elif gd == -1: p["away"] += .18; p["draw"] -= .08; p["home"] -= .10
    elif gd <= -2: p["away"] += .35; p["draw"] -= .15; p["home"] -= .20
    else: p["draw"] += .10; p["home"] -= .05; p["away"] -= .05
    xgd, total_xg = hxg - axg, hxg + axg
    label = "data_insufficient" if not has_xg else "volatile_match"
    if has_xg:
        if gd > 0 and xgd < -0.7: p["home"] -= .12; p["away"] += .08; p["draw"] += .04; label = "lucky_lead"
        elif gd < 0 and xgd > 0.7: p["away"] -= .12; p["home"] += .08; p["draw"] += .04; label = "lucky_lead"
        elif gd == 0 and abs(xgd) > 0.8:
            dom = "home" if xgd > 0 else "away"; p[dom] += .08; p["draw"] -= .04; label = "dominant_draw"
        elif gd == 0 and total_xg < 0.7: p["draw"] += .08; p["home"] -= .04; p["away"] -= .04; label = "dead_match"
        elif gd != 0 and ((gd > 0 and xgd > 0.35) or (gd < 0 and xgd < -0.35)): label = "deserved_lead"
        if total_xg > 1.8: p["draw"] -= .04; label = "volatile_match" if label not in {"lucky_lead", "deserved_lead"} else label
    card_adjustment = {"home": 0.0, "draw": 0.0, "away": 0.0}
    if hr: p["home"] -= .12; p["draw"] += .04; p["away"] += .08; card_adjustment.update({"home": -.12, "draw": .04, "away": .08})
    if ar: p["away"] -= .12; p["draw"] += .04; p["home"] += .08; card_adjustment.update({"away": -.12, "draw": card_adjustment["draw"]+.04, "home": card_adjustment["home"]+.08})
    if (hr or ar) and label not in {"lucky_lead", "deserved_lead"}: label = "red_card_distortion"
    if gd > 0 and hr: p["away"] += .04; p["home"] -= .04; label = "red_card_distortion"
    if gd < 0 and ar: p["home"] += .04; p["away"] -= .04; label = "red_card_distortion"
    ht_probs = temperature_scale(clamp_probs(p), 1.08)
    pressure = (hs - aws) * .04 + (hsot - asot) * .10
    confidence = 0.58 + (0.12 if has_xg else -0.10) + (0.04 if abs(gd) >= 1 else 0) - (0.05 if hr or ar else 0)
    return {"ht_label": label, "scoreline_luck": {"home_luck": hg - hxg, "away_luck": ag - axg}, "performance_signal": {"xg_delta": xgd, "shot_pressure_delta": pressure, "card_adjustment": card_adjustment}, "ht_probs": ht_probs, "confidence": max(.2, min(.85, confidence)), "reason": f"HT score {hg}-{ag}, xG {hxg:.2f}-{axg:.2f}, label={label}."}
