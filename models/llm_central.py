from __future__ import annotations

from models.calibration import OUTCOMES, normalize_probs


BLOCKING_FLAGS = {
    "llm_central_missing",
    "llm_central_failed",
    "llm_central_invalid_probabilities",
    "llm_central_veto",
}


def normalize_central_prediction(
    llm_result: dict | None,
    *,
    fallback_probs: dict[str, float],
    fallback_confidence: float,
    fallback_uncertainty: float,
) -> dict:
    if not llm_result:
        return _fallback_payload(
            "llm_central_missing",
            fallback_probs=fallback_probs,
            fallback_confidence=fallback_confidence,
            fallback_uncertainty=fallback_uncertainty,
        )
    if not llm_result.get("ok"):
        return _fallback_payload(
            "llm_central_failed",
            fallback_probs=fallback_probs,
            fallback_confidence=fallback_confidence,
            fallback_uncertainty=fallback_uncertainty,
            llm_result=llm_result,
        )

    parsed = llm_result.get("parsed") or {}
    raw_probs = _extract_probabilities(parsed)
    if not raw_probs:
        return _fallback_payload(
            "llm_central_invalid_probabilities",
            fallback_probs=fallback_probs,
            fallback_confidence=fallback_confidence,
            fallback_uncertainty=fallback_uncertainty,
            llm_result=llm_result,
        )

    risk_flags = []
    recommendation = str(parsed.get("recommendation") or "").strip().upper()
    posture = str(parsed.get("risk_posture") or "").strip().lower()
    if recommendation == "SKIP":
        risk_flags.append("llm_central_recommends_skip")
    elif recommendation == "WATCH":
        risk_flags.append("llm_central_recommends_watch")
    elif recommendation and recommendation != "BET":
        risk_flags.append("llm_central_not_betting")

    if posture == "veto":
        risk_flags.append("llm_central_veto")
    elif posture and posture != "approve":
        risk_flags.append("llm_central_not_betting")

    for flag in parsed.get("additional_risk_flags") or []:
        normalized = str(flag).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized:
            risk_flags.append(normalized[:80])

    return {
        "probabilities": normalize_probs(raw_probs),
        "confidence": _clamp_float(parsed.get("confidence"), fallback_confidence, low=0.30, high=0.90),
        "uncertainty": _clamp_float(parsed.get("uncertainty"), fallback_uncertainty, low=0.05, high=0.65),
        "risk_flags": list(dict.fromkeys(risk_flags)),
        "blocking_risk_flags": [flag for flag in dict.fromkeys(risk_flags) if flag in BLOCKING_FLAGS],
        "supporting_signals": [str(x)[:120] for x in (parsed.get("supporting_signals") or [])[:4]],
        "contradicting_signals": [str(x)[:120] for x in (parsed.get("contradicting_signals") or [])[:4]],
        "rationale": str(parsed.get("rationale") or "")[:800],
        "recommendation": recommendation or "UNKNOWN",
        "risk_posture": posture or "unknown",
        "used_fallback": False,
        "provider_result": llm_result,
    }


def _extract_probabilities(parsed: dict) -> dict[str, float] | None:
    candidates = [
        parsed.get("probabilities"),
        parsed.get("final_probs"),
        {k: parsed.get(k) for k in OUTCOMES},
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        try:
            values = {k: float(candidate.get(k, 0.0) or 0.0) for k in OUTCOMES}
        except Exception:
            continue
        if sum(max(v, 0.0) for v in values.values()) > 0:
            return values
    return None


def _fallback_payload(
    reason: str,
    *,
    fallback_probs: dict[str, float],
    fallback_confidence: float,
    fallback_uncertainty: float,
    llm_result: dict | None = None,
) -> dict:
    return {
        "probabilities": normalize_probs(fallback_probs),
        "confidence": fallback_confidence,
        "uncertainty": fallback_uncertainty,
        "risk_flags": [reason],
        "blocking_risk_flags": [reason],
        "supporting_signals": [],
        "contradicting_signals": [],
        "rationale": str((llm_result or {}).get("reason") or (llm_result or {}).get("errors") or reason)[:800],
        "recommendation": "SKIP",
        "risk_posture": "veto",
        "used_fallback": True,
        "provider_result": llm_result,
    }


def _clamp_float(value, default: float, *, low: float, high: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = float(default)
    return max(low, min(high, numeric))
