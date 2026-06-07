from __future__ import annotations

from typing import Any

from models.calibration import OUTCOMES, normalize_probs


TRUSTED_SOURCE_KINDS = {"official", "sportmonks", "lineup", "bookmaker", "weather", "team", "fifa"}
WEAK_SOURCE_KINDS = {"web", "news", "reddit", "rumor", "sentiment", "social"}


def reconcile_council_output(
    council_output: dict | None,
    *,
    deterministic_reference: dict,
    max_delta: float = 0.045,
) -> dict:
    """
    Validate and compare an external LLM-council payload.

    The council may supply probabilities and evidence, but this function turns
    that into a bounded, auditable adjustment request. No orders are authorized
    here.
    """
    fallback_probs = normalize_probs(deterministic_reference.get("probabilities"))
    if not council_output:
        return {
            "available": False,
            "accepted": False,
            "probabilities": fallback_probs,
            "bounded_delta": {k: 0.0 for k in OUTCOMES},
            "agreement": "missing",
            "evidence_score": 0.0,
            "risk_flags": [],
            "reasons": ["No council output supplied."],
        }

    raw_probs = _extract_probs(council_output)
    if raw_probs is None:
        return {
            "available": True,
            "accepted": False,
            "probabilities": fallback_probs,
            "bounded_delta": {k: 0.0 for k in OUTCOMES},
            "agreement": "invalid",
            "evidence_score": 0.0,
            "risk_flags": ["council_invalid_probabilities"],
            "reasons": ["Council payload did not contain usable home/draw/away probabilities."],
            "raw": _compact(council_output),
        }

    council_probs = normalize_probs(raw_probs)
    evidence_score = _evidence_score(council_output)
    raw_delta = {k: council_probs[k] - fallback_probs[k] for k in OUTCOMES}
    cap = max(0.0, min(max_delta, 0.015 + 0.045 * evidence_score))
    bounded_delta = _cap_delta(raw_delta, cap)
    adjusted = normalize_probs({k: fallback_probs[k] + bounded_delta[k] for k in OUTCOMES})
    agreement = _agreement(fallback_probs, council_probs)
    risk_flags: list[str] = []
    reasons: list[str] = []

    if evidence_score < 0.35:
        risk_flags.append("council_low_evidence")
        reasons.append("Council evidence quality below acceptance threshold.")
    if agreement == "sharp_conflict" and evidence_score < 0.65:
        risk_flags.append("council_unresolved_conflict")
        reasons.append("Council sharply conflicts with deterministic forecast without strong evidence.")
    recommendation = str(council_output.get("recommendation") or "").upper()
    posture = str(council_output.get("risk_posture") or "").lower()
    if recommendation in {"SKIP", "WATCH"}:
        risk_flags.append(f"council_recommends_{recommendation.lower()}")
    if posture == "veto":
        risk_flags.append("council_veto")

    accepted = evidence_score >= 0.35 and "council_unresolved_conflict" not in risk_flags
    if accepted and any(abs(v) > 1e-9 for v in bounded_delta.values()):
        reasons.append(f"Council adjustment accepted with cap {cap:.3f}.")
    elif accepted:
        reasons.append("Council accepted but did not materially move probabilities.")

    return {
        "available": True,
        "accepted": accepted,
        "probabilities": adjusted if accepted else fallback_probs,
        "council_probabilities": council_probs,
        "raw_delta": raw_delta,
        "bounded_delta": bounded_delta if accepted else {k: 0.0 for k in OUTCOMES},
        "agreement": agreement,
        "evidence_score": evidence_score,
        "risk_flags": list(dict.fromkeys(risk_flags)),
        "reasons": reasons,
        "supporting_signals": [str(x)[:160] for x in (council_output.get("supporting_signals") or [])[:5]],
        "challenged_signals": [str(x)[:160] for x in (council_output.get("challenged_signals") or council_output.get("contradicting_signals") or [])[:5]],
        "raw": _compact(council_output),
    }


def _extract_probs(payload: dict[str, Any]) -> dict[str, float] | None:
    candidates = [payload.get("probabilities"), payload.get("final_probs"), {k: payload.get(k) for k in OUTCOMES}]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        try:
            vals = {k: float(candidate.get(k, 0.0) or 0.0) for k in OUTCOMES}
        except Exception:
            continue
        if sum(max(v, 0.0) for v in vals.values()) > 0:
            return vals
    return None


def _evidence_score(payload: dict[str, Any]) -> float:
    evidence = payload.get("evidence") or payload.get("sources") or []
    if not isinstance(evidence, list):
        evidence = []
    score = 0.0
    for item in evidence[:8]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("source_kind") or item.get("kind") or item.get("type") or "").lower()
        confidence = _clamp01(item.get("confidence", 0.5))
        if kind in TRUSTED_SOURCE_KINDS:
            score += 0.24 * confidence
        elif kind in WEAK_SOURCE_KINDS:
            score += 0.08 * confidence
        else:
            score += 0.12 * confidence
    if payload.get("citations") and isinstance(payload.get("citations"), list):
        score += min(0.16, 0.04 * len(payload["citations"]))
    if payload.get("structured_claims"):
        score += 0.10
    return max(0.0, min(1.0, score))


def _cap_delta(delta: dict[str, float], cap: float) -> dict[str, float]:
    max_abs = max(abs(v) for v in delta.values()) if delta else 0.0
    if max_abs <= cap or max_abs <= 0:
        return dict(delta)
    scale = cap / max_abs
    return {k: v * scale for k, v in delta.items()}


def _agreement(det: dict[str, float], council: dict[str, float]) -> str:
    det_pick = max(OUTCOMES, key=lambda k: det[k])
    council_pick = max(OUTCOMES, key=lambda k: council[k])
    max_gap = max(abs(det[k] - council[k]) for k in OUTCOMES)
    if det_pick == council_pick and max_gap <= 0.04:
        return "strong_agreement"
    if det_pick == council_pick:
        return "same_pick_different_confidence"
    if max_gap >= 0.10:
        return "sharp_conflict"
    return "mild_conflict"


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _compact(payload: dict[str, Any]) -> dict:
    keep = {
        "probabilities",
        "final_probs",
        "recommendation",
        "risk_posture",
        "supporting_signals",
        "challenged_signals",
        "contradicting_signals",
        "citations",
        "evidence",
        "rationale",
    }
    return {k: payload.get(k) for k in keep if k in payload}
