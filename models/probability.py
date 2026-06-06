from __future__ import annotations
from models.calibration import OUTCOMES, clamp_probs, normalize_probs, shrink_toward_market, temperature_scale
from models.lineup_delta import apply_lineup_delta
from models.probability_blender import DEFAULT_HALFTIME_WEIGHTS, DEFAULT_PREMATCH_WEIGHTS, deterministic_blend
from models.signal_scoring import score_signal


def blend_probabilities(sources: dict[str, dict[str, float] | None], weights: dict[str, float]) -> dict[str, float]:
    usable = {n: normalize_probs(p) for n, p in sources.items() if p and sum(float(p.get(k,0) or 0) for k in OUTCOMES) > 0}
    if not usable:
        return normalize_probs(None)
    total_w = sum(weights.get(n, 0.0) for n in usable) or len(usable)
    return normalize_probs({k: sum(usable[n][k] * (weights.get(n, 1.0) / total_w) for n in usable) for k in OUTCOMES})


def pre_match_model(sportmonks_probs=None, bookmaker_probs=None, market_probs=None, supabase_priors=None, lineup_delta=None, data_completeness: float = 1.0, structured_signals: list[dict] | None = None) -> dict:
    signals = list(structured_signals or [])
    if lineup_delta:
        signals.append(score_signal("lineup_delta_probability", "lineup", "lineup", lineup_delta, source_quality=.9, freshness=.85, corroboration=.8, reason="Official lineup delta."))
    strength = 0.05 if data_completeness >= .7 else 0.12
    blended = deterministic_blend(
        {"bookmaker": bookmaker_probs, "sportmonks": sportmonks_probs, "polymarket": market_probs, "supabase": supabase_priors},
        weights=DEFAULT_PREMATCH_WEIGHTS,
        market_probs=market_probs,
        signals=signals,
        temperature=1.05,
        market_shrink=strength,
    )
    return {
        **blended,
        "confidence": max(.25, min(.82, blended["confidence"] + .05 * data_completeness)),
        "uncertainty": max(.18, min(.65, blended["uncertainty"] + max(0, .65 - data_completeness) * .15)),
    }


def halftime_model(pre_match_probs, halftime_output, market_probs=None, bookmaker_probs=None, structured_signals: list[dict] | None = None) -> dict:
    blended = deterministic_blend(
        {"halftime": halftime_output.get("ht_probs"), "polymarket": market_probs, "prematch": pre_match_probs, "bookmaker": bookmaker_probs},
        weights=DEFAULT_HALFTIME_WEIGHTS,
        market_probs=market_probs,
        signals=structured_signals or [],
        temperature=1.08,
        market_shrink=0.04,
    )
    conf = float(halftime_output.get("confidence", .55))
    return {**blended, "confidence": max(.3, min(.85, min(blended["confidence"], conf + .12))), "uncertainty": max(.15, min(.65, max(blended["uncertainty"], 1-conf)))}
