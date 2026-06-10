from __future__ import annotations

from models.calibration import OUTCOMES, calibrate_probs, normalize_probs


DEFAULT_PREMATCH_WEIGHTS = {
    "bookmaker": 0.30,
    "sportmonks": 0.25,
    "polymarket": 0.20,
    "supabase": 0.15,
    "lineup": 0.05,
    "draw_model": 0.05,
    "llm_claims": 0.03,
}

DEFAULT_HALFTIME_WEIGHTS = {
    "halftime": 0.45,
    "bookmaker": 0.15,
    "polymarket": 0.15,
    "prematch": 0.12,
    "sportmonks": 0.08,
    "llm_claims": 0.05,
}


def deterministic_blend(
    sources: dict[str, dict[str, float] | None],
    *,
    weights: dict[str, float] | None = None,
    market_probs: dict[str, float] | None = None,
    signals: list[dict] | None = None,
    temperature: float = 1.06,
    market_shrink: float = 0.06,
    draw_floor: float = 0.16,
    llm_signal_cap: float = 0.02,
) -> dict:
    weights = weights or DEFAULT_PREMATCH_WEIGHTS
    usable = {name: normalize_probs(probs) for name, probs in sources.items() if _has_probs(probs)}
    risk_flags: list[str] = []
    if not usable:
        usable = {"uniform_fallback": normalize_probs(None)}
        risk_flags.append("all_probability_sources_missing")
    missing_sources = [name for name in weights if name not in usable]
    if missing_sources:
        risk_flags.append("probability_sources_missing")

    normalized_weights = _redistributed_weights(usable, weights)
    source_contributions = {
        name: {outcome: usable[name][outcome] * normalized_weights[name] for outcome in OUTCOMES}
        for name in usable
    }
    blended = normalize_probs({outcome: sum(c[outcome] for c in source_contributions.values()) for outcome in OUTCOMES})

    signal_result = apply_structured_signals(blended, signals or [], llm_signal_cap=llm_signal_cap)
    signal_adjusted = signal_result["probabilities"]
    if signal_result["risk_flags"]:
        risk_flags.extend(signal_result["risk_flags"])

    calibration = calibrate_probs(signal_adjusted, market_probs=market_probs, temperature=temperature, shrink_strength=market_shrink, draw_floor=draw_floor)
    final_probs = calibration["probabilities"]
    risk_flags.extend(calibration["risk_flags"])

    final_contribution = _attribute_final_contribution(
        base_contributions=source_contributions,
        base_probs=blended,
        final_probs=final_probs,
        signal_influence=signal_result["influence"],
    )
    confidence = _confidence(normalized_weights, missing_sources, signals or [], calibration["risk_flags"])
    uncertainty = max(0.08, min(0.75, 1.0 - confidence + 0.04 * len(missing_sources)))
    return {
        "probabilities": final_probs,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "risk_flags": list(dict.fromkeys(risk_flags)),
        "weights": normalized_weights,
        "missing_sources": missing_sources,
        "source_contribution": final_contribution,
        "steps": [
            {"name": "source_blend", "probabilities": blended, "weights": normalized_weights, "contribution": source_contributions},
            {"name": "structured_signals", "probabilities": signal_adjusted, "aggregate_delta": signal_result["aggregate_delta"], "influence": signal_result["influence"]},
            {"name": "calibration", "probabilities": final_probs, "details": calibration},
        ],
    }


def apply_structured_signals(base_probs: dict[str, float], signals: list[dict], *, llm_signal_cap: float = 0.02) -> dict:
    base = normalize_probs(base_probs)
    aggregate = {k: 0.0 for k in OUTCOMES}
    influence = 0.0
    risk_flags: list[str] = []
    for signal in signals:
        source = str(signal.get("source") or "").lower()
        delta = {k: float((signal.get("probability_delta") or {}).get(k, 0.0) or 0.0) for k in OUTCOMES}
        cap = llm_signal_cap if source in {"web", "news", "reddit", "text", "rumor"} or str(signal.get("name", "")).startswith("claim_") else 0.07
        max_abs = max(abs(v) for v in delta.values()) if delta else 0.0
        if max_abs > cap and max_abs > 0:
            scale = cap / max_abs
            delta = {k: v * scale for k, v in delta.items()}
            risk_flags.append("structured_signal_delta_capped")
        weight = max(0.0, min(1.0, float(signal.get("final_weight", 0.0) or 0.0)))
        influence += max(abs(v) for v in delta.values()) * weight
        for outcome in OUTCOMES:
            aggregate[outcome] += delta[outcome] * weight
    influence = max(0.0, min(0.25, influence))
    adjusted = normalize_probs({k: base[k] + aggregate[k] for k in OUTCOMES})
    return {"probabilities": adjusted, "aggregate_delta": aggregate, "influence": influence, "risk_flags": list(dict.fromkeys(risk_flags))}


def contribution_sums_to_probs(contribution: dict[str, dict[str, float]], probabilities: dict[str, float], *, tolerance: float = 1e-6) -> bool:
    for outcome in OUTCOMES:
        if abs(sum(float(c.get(outcome, 0.0) or 0.0) for c in contribution.values()) - float(probabilities[outcome])) > tolerance:
            return False
    return True


def _redistributed_weights(usable: dict[str, dict[str, float]], weights: dict[str, float]) -> dict[str, float]:
    raw = {name: max(0.0, float(weights.get(name, 0.0) or 0.0)) for name in usable}
    total = sum(raw.values())
    if total <= 0:
        return {name: 1.0 / len(usable) for name in usable}
    return {name: raw[name] / total for name in usable}


def _has_probs(probs: dict[str, float] | None) -> bool:
    return bool(probs) and sum(float((probs or {}).get(k, 0.0) or 0.0) for k in OUTCOMES) > 0


def _attribute_final_contribution(
    *,
    base_contributions: dict[str, dict[str, float]],
    base_probs: dict[str, float],
    final_probs: dict[str, float],
    signal_influence: float,
) -> dict[str, dict[str, float]]:
    signal_share = max(0.0, min(0.20, signal_influence))
    base_share = 1.0 - signal_share
    attributed: dict[str, dict[str, float]] = {}
    for name, contrib in base_contributions.items():
        attributed[name] = {}
        for outcome in OUTCOMES:
            share = contrib[outcome] / max(base_probs[outcome], 1e-9)
            attributed[name][outcome] = final_probs[outcome] * base_share * share
    if signal_share > 0:
        attributed["structured_signals"] = {outcome: final_probs[outcome] * signal_share for outcome in OUTCOMES}
    return attributed


def _confidence(weights: dict[str, float], missing_sources: list[str], signals: list[dict], calibration_flags: list[str]) -> float:
    source_strength = sum(weights.values())
    missing_penalty = min(0.35, 0.06 * len(missing_sources))
    weak_signal_penalty = 0.0
    if signals:
        weak_signal_penalty = min(0.08, 0.02 * sum(1 for s in signals if str(s.get("source", "")).lower() in {"web", "news", "reddit", "text"}))
    calibration_penalty = 0.05 if calibration_flags else 0.0
    return max(0.25, min(0.88, 0.58 + 0.22 * source_strength - missing_penalty - weak_signal_penalty - calibration_penalty))
