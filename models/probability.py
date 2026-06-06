from __future__ import annotations
from models.calibration import OUTCOMES, clamp_probs, normalize_probs, shrink_toward_market, temperature_scale
from models.lineup_delta import apply_lineup_delta


def blend_probabilities(sources: dict[str, dict[str, float] | None], weights: dict[str, float]) -> dict[str, float]:
    usable = {n: normalize_probs(p) for n, p in sources.items() if p and sum(float(p.get(k,0) or 0) for k in OUTCOMES) > 0}
    if not usable:
        return normalize_probs(None)
    total_w = sum(weights.get(n, 0.0) for n in usable) or len(usable)
    return normalize_probs({k: sum(usable[n][k] * (weights.get(n, 1.0) / total_w) for n in usable) for k in OUTCOMES})


def pre_match_model(sportmonks_probs=None, bookmaker_probs=None, market_probs=None, supabase_priors=None, lineup_delta=None, data_completeness: float = 1.0) -> dict:
    base = blend_probabilities({"bookmaker": bookmaker_probs, "sportmonks": sportmonks_probs, "market": market_probs, "supabase": supabase_priors}, {"bookmaker": .30, "sportmonks": .25, "market": .20, "supabase": .15})
    if lineup_delta:
        base = apply_lineup_delta(base, lineup_delta)
    strength = 0.05 if data_completeness >= .7 else 0.12
    final = temperature_scale(shrink_toward_market(clamp_probs(base), market_probs, strength), 1.05)
    return {"probabilities": final, "confidence": max(.35, min(.82, .50 + .25*data_completeness)), "uncertainty": max(.18, min(.60, 1-data_completeness))}


def halftime_model(pre_match_probs, halftime_output, market_probs=None, bookmaker_probs=None) -> dict:
    blended = blend_probabilities({"halftime": halftime_output.get("ht_probs"), "market": market_probs, "prematch": pre_match_probs, "bookmaker": bookmaker_probs}, {"halftime": .55, "market": .20, "prematch": .15, "bookmaker": .10})
    final = temperature_scale(clamp_probs(blended), 1.08)
    conf = float(halftime_output.get("confidence", .55))
    return {"probabilities": final, "confidence": max(.3, min(.85, conf)), "uncertainty": max(.15, min(.55, 1-conf))}
