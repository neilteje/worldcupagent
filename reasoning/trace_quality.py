from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from typing import Any


REQUIRED_BEHAVIORS = {"Observing", "Planning", "ToolCalling", "Thinking", "Acting", "Reflecting"}
TRACE_SCHEMA_VERSION = "reasoning_trace_v1"


def reasoning_trace(
    *,
    step_type: str,
    objective: str,
    inputs_used: list[str] | None = None,
    method: str = "",
    assumptions: list[str] | None = None,
    uncertainties: list[str] | None = None,
    checks: list[str] | None = None,
    decision_rule: str = "",
    output_summary: str = "",
    evidence_refs: list[str] | None = None,
    risk_controls: list[str] | None = None,
) -> dict:
    return {
        "trace_schema": TRACE_SCHEMA_VERSION,
        "step_type": step_type,
        "objective": objective,
        "inputs_used": inputs_used or [],
        "method": method,
        "assumptions": assumptions or [],
        "uncertainties": uncertainties or [],
        "checks": checks or [],
        "decision_rule": decision_rule,
        "output_summary": output_summary,
        "evidence_refs": evidence_refs or [],
        "risk_controls": risk_controls or [],
    }


def model_invocation(
    *,
    provider: str,
    model_name: str,
    internal_reasoning: str = "",
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    temperature: float | None = None,
) -> dict:
    payload: dict[str, Any] = {
        "provider": provider,
        "model_name": model_name,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if internal_reasoning:
        payload["internal_reasoning"] = internal_reasoning[:12000]
    return payload


def summarize_reasoning(*, label: str, inputs: dict | None = None, outputs: dict | None = None, rule: str = "", risks: list[str] | None = None) -> str:
    parts = [f"{label}."]
    if inputs:
        parts.append(f"Inputs: {_compact(inputs)}.")
    if rule:
        parts.append(f"Decision rule: {rule}.")
    if outputs:
        parts.append(f"Outputs: {_compact(outputs)}.")
    if risks:
        parts.append(f"Risk controls: {', '.join(risks[:8])}.")
    return " ".join(parts)


def evaluate_trace(records: list[dict]) -> dict:
    behaviors = Counter(str(r.get("behavior")) for r in records)
    ids = {r.get("record_id") for r in records}
    parent_ok = all(pid in ids for r in records for pid in (r.get("parent_ids") or r.get("upstream_record_id") or []))
    prediction_records = [
        r for r in records
        if r.get("behavior") == "Acting" and r.get("action_type") == "prediction"
    ]
    score = 0.0
    components: dict[str, float] = {}
    components["behavior_coverage"] = len(REQUIRED_BEHAVIORS & set(behaviors)) / len(REQUIRED_BEHAVIORS)
    components["dag_validity"] = 1.0 if parent_ok else 0.0
    components["prediction_record"] = 1.0 if prediction_records else 0.0
    components["reasoning_trace_coverage"] = _coverage(records, lambda r: bool(r.get("reasoning_trace")))
    components["model_invocation_coverage"] = _coverage([r for r in records if r.get("behavior") == "Thinking"], lambda r: bool(r.get("model_invocation")))
    components["evidence_reference_coverage"] = _coverage(records, lambda r: bool((r.get("reasoning_trace") or {}).get("evidence_refs")))
    components["risk_control_coverage"] = _coverage(records, lambda r: bool((r.get("reasoning_trace") or {}).get("risk_controls")))
    components["final_reflection"] = 1.0 if any(r.get("behavior") == "Reflecting" and "trace_quality" in r for r in records) else 0.0
    weights = {
        "behavior_coverage": 0.16,
        "dag_validity": 0.16,
        "prediction_record": 0.13,
        "reasoning_trace_coverage": 0.15,
        "model_invocation_coverage": 0.12,
        "evidence_reference_coverage": 0.10,
        "risk_control_coverage": 0.10,
        "final_reflection": 0.08,
    }
    for key, weight in weights.items():
        score += components[key] * weight
    gaps = []
    for behavior in sorted(REQUIRED_BEHAVIORS - set(behaviors)):
        gaps.append(f"missing_behavior:{behavior}")
    if not parent_ok:
        gaps.append("invalid_dag_parent_reference")
    if not prediction_records:
        gaps.append("missing_prediction_acting_record")
    if components["reasoning_trace_coverage"] < 0.85:
        gaps.append("low_reasoning_trace_coverage")
    return {
        "score": round(score, 4),
        "components": {k: round(v, 4) for k, v in components.items()},
        "record_count": len(records),
        "behavior_counts": dict(behaviors),
        "gaps": gaps,
        "trace_hash": _trace_hash(records),
    }


def _coverage(records: list[dict], predicate) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if predicate(r)) / len(records)


def _compact(value: Any, limit: int = 700) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= limit else text[:limit] + f"...[truncated {len(text)} chars]"


def _trace_hash(records: list[dict]) -> str:
    canonical = json.dumps(records, sort_keys=True, default=str)
    return sha256(canonical.encode("utf-8")).hexdigest()[:24]
