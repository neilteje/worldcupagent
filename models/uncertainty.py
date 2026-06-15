"""Forecast uncertainty metadata."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbabilityUncertainty:
    lower: dict[str, float]
    upper: dict[str, float]
    uncertainty_source: str
    calibration_sample_size: int
    model_version: str


def build_uncertainty(probabilities: dict[str, float], *,
                      confidence: float,
                      coverage: float,
                      calibration_sample_size: int = 0,
                      model_version: str = "forecast_pipeline_v1") -> ProbabilityUncertainty:
    source = "historical_residuals" if calibration_sample_size > 0 else "fallback"
    half = max(0.04, min(0.22, 0.05 + (1.0 - confidence) * 0.10 + (1.0 - coverage) * 0.10))
    return ProbabilityUncertainty(
        lower={k: max(0.0, float(v) - half) for k, v in probabilities.items()},
        upper={k: min(1.0, float(v) + half) for k, v in probabilities.items()},
        uncertainty_source=source,
        calibration_sample_size=max(0, int(calibration_sample_size)),
        model_version=model_version,
    )


__all__ = ["ProbabilityUncertainty", "build_uncertainty"]
