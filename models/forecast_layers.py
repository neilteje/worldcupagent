"""Layered forecast pipeline with bounded LLM adjustments."""
from __future__ import annotations

from dataclasses import dataclass

from models.calibration import OUTCOMES, normalize_probs
from models.market_calibration import apply_market_calibration, clamp_market_weight

MAX_OUTCOME_ADJUSTMENT = 0.08
MAX_TOTAL_ADJUSTMENT = 0.12
FORECAST_PIPELINE_VERSION = "forecast_layers_v1"


@dataclass
class ForecastLayers:
    independent_probabilities: dict[str, float]
    evidence_adjusted_probabilities: dict[str, float]
    stressed_probabilities: dict[str, float]
    pre_market_probabilities: dict[str, float]
    market_probabilities: dict[str, float] | None
    scored_probabilities: dict[str, float]
    market_calibration_weight: float
    evidence_ids: tuple[str, ...]
    data_coverage_score: float
    confidence: float
    warnings: tuple[str, ...]


def validate_analyst_adjustments(parsed: dict | None) -> tuple[dict[str, float], tuple[str, ...]]:
    """Malformed Analyst output means zero adjustment."""
    raw = (parsed or {}).get("adjustments")
    if not isinstance(raw, dict):
        return {k: 0.0 for k in OUTCOMES}, ("invalid_analyst_adjustment",)
    vals: dict[str, float] = {}
    try:
        for k in OUTCOMES:
            vals[k] = max(-MAX_OUTCOME_ADJUSTMENT, min(MAX_OUTCOME_ADJUSTMENT, float(raw.get(k, 0.0))))
    except (TypeError, ValueError):
        return {k: 0.0 for k in OUTCOMES}, ("invalid_analyst_adjustment",)
    total = sum(abs(v) for v in vals.values())
    warnings: list[str] = []
    if total > MAX_TOTAL_ADJUSTMENT:
        scale = MAX_TOTAL_ADJUSTMENT / total
        vals = {k: v * scale for k, v in vals.items()}
        warnings.append("analyst_total_adjustment_capped")
    # Keep distribution mass stable; final normalization handles tiny drift.
    mean_shift = sum(vals.values()) / 3.0
    vals = {k: vals[k] - mean_shift for k in OUTCOMES}
    return vals, tuple(warnings)


def apply_evidence_adjustments(independent: dict[str, float],
                               analyst_output: dict | None) -> tuple[dict[str, float], tuple[str, ...]]:
    base = normalize_probs(independent)
    adj, warnings = validate_analyst_adjustments(analyst_output)
    return normalize_probs({k: max(0.001, base[k] + adj[k]) for k in OUTCOMES}), warnings


def validate_and_aggregate_scenarios(base: dict[str, float],
                                     devil_output: dict | None) -> tuple[dict[str, float], tuple[str, ...]]:
    """Aggregate two or three weighted stress scenarios deterministically."""
    base = normalize_probs(base)
    raw = (devil_output or {}).get("scenarios")
    if not isinstance(raw, list) or not (2 <= len(raw) <= 3):
        return base, ("invalid_devil_scenarios",)
    scenarios = []
    for row in sorted(raw, key=lambda s: str((s or {}).get("scenario_id", ""))):
        if not isinstance(row, dict):
            return base, ("invalid_devil_scenarios",)
        try:
            plaus = max(0.0, min(1.0, float(row.get("plausibility"))))
            probs = normalize_probs({k: float((row.get("probabilities") or {}).get(k)) for k in OUTCOMES})
        except (TypeError, ValueError):
            return base, ("invalid_devil_scenarios",)
        scenarios.append((plaus, probs))
    total_stress = min(0.50, sum(p for p, _ in scenarios))
    if total_stress <= 0:
        return base, tuple()
    denom = sum(p for p, _ in scenarios) or 1.0
    stress = normalize_probs({
        k: sum(probs[k] * plaus / denom for plaus, probs in scenarios)
        for k in OUTCOMES
    })
    return normalize_probs({k: base[k] * (1.0 - total_stress) + stress[k] * total_stress for k in OUTCOMES}), tuple()


def build_forecast_layers(independent: dict[str, float], *,
                          analyst_output: dict | None = None,
                          devil_output: dict | None = None,
                          judge_output: dict | None = None,
                          market_probabilities: dict | None = None,
                          evidence_ids: tuple[str, ...] = tuple(),
                          data_coverage_score: float = 0.0,
                          confidence: float = 0.5) -> ForecastLayers:
    warnings: list[str] = ["independent_forecast_market_blind"]
    independent_norm = normalize_probs(independent)
    evidence_adjusted, adj_warnings = apply_evidence_adjustments(independent_norm, analyst_output)
    warnings.extend(adj_warnings)
    stressed, stress_warnings = validate_and_aggregate_scenarios(evidence_adjusted, devil_output)
    warnings.extend(stress_warnings)
    pre_market = stressed
    weight = clamp_market_weight((judge_output or {}).get("recommended_market_weight"))
    scored = apply_market_calibration(pre_market, market_probabilities, weight)
    return ForecastLayers(
        independent_probabilities=independent_norm,
        evidence_adjusted_probabilities=evidence_adjusted,
        stressed_probabilities=stressed,
        pre_market_probabilities=pre_market,
        market_probabilities=normalize_probs(market_probabilities) if market_probabilities else None,
        scored_probabilities=scored,
        market_calibration_weight=weight if market_probabilities else 0.0,
        evidence_ids=tuple(sorted(evidence_ids)),
        data_coverage_score=max(0.0, min(1.0, float(data_coverage_score))),
        confidence=max(0.0, min(1.0, float(confidence))),
        warnings=tuple(warnings),
    )


__all__ = [
    "FORECAST_PIPELINE_VERSION",
    "ForecastLayers",
    "apply_evidence_adjustments",
    "build_forecast_layers",
    "validate_analyst_adjustments",
    "validate_and_aggregate_scenarios",
]
