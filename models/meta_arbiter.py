from __future__ import annotations

from models.calibration import normalize_probs


def arbitrate_forecast(
    base_model: dict,
    *,
    council_reconciliation: dict | None = None,
    deterministic_reference: dict | None = None,
) -> dict:
    """
    Produce the forecast used by downstream edge/risk logic.

    Deterministic output is the default. Council output can only move the
    forecast if it was validated and bounded by council_reconciliation.
    """
    base_probs = normalize_probs(base_model.get("probabilities"))
    council = council_reconciliation or {}
    risk_flags = list(base_model.get("risk_flags") or [])
    steps = list(base_model.get("steps") or [])

    if council.get("available"):
        risk_flags.extend(council.get("risk_flags") or [])
        steps.append(
            {
                "name": "council_reconciliation",
                "accepted": council.get("accepted"),
                "agreement": council.get("agreement"),
                "evidence_score": council.get("evidence_score"),
                "bounded_delta": council.get("bounded_delta"),
                "reasons": council.get("reasons"),
            }
        )

    if council.get("accepted"):
        probs = normalize_probs(council.get("probabilities"))
        confidence = min(0.88, float(base_model.get("confidence", 0.5) or 0.5) + min(0.04, 0.04 * float(council.get("evidence_score", 0.0) or 0.0)))
        uncertainty = max(0.08, float(base_model.get("uncertainty", 0.45) or 0.45) - min(0.04, 0.03 * float(council.get("evidence_score", 0.0) or 0.0)))
        mode = "deterministic_plus_council"
    else:
        probs = base_probs
        confidence = float(base_model.get("confidence", 0.5) or 0.5)
        uncertainty = float(base_model.get("uncertainty", 0.45) or 0.45)
        mode = "deterministic"

    return {
        **base_model,
        "probabilities": probs,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "risk_flags": list(dict.fromkeys(risk_flags)),
        "steps": steps,
        "arbiter": {
            "mode": mode,
            "base_probabilities": base_probs,
            "deterministic_reference": deterministic_reference,
            "council": council if council.get("available") else None,
        },
    }
