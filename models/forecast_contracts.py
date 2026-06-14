"""Structured forecast and recommendation contracts.

These contracts are intentionally data-only.  They make forecast snapshots and
agent recommendations auditable without forcing every execution path to adopt a
new strategy implementation at once.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any


OUTCOME_SLOTS = ("home", "draw", "away")


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def stable_hash(payload: Any) -> str:
    """Return a stable sha256 hash for JSON-serializable snapshot data."""
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _require_probability(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1], got {value!r}")
    return value


@dataclass
class MatchForecast:
    fixture_id: str
    as_of_timestamp: datetime

    home_probability: float
    draw_probability: float
    away_probability: float

    home_lower_bound: float
    draw_lower_bound: float
    away_lower_bound: float

    home_upper_bound: float
    draw_upper_bound: float
    away_upper_bound: float

    confidence: float
    data_coverage_score: float

    model_version: str
    feature_snapshot_hash: str

    evidence_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    forecast_id: str | None = None

    def __post_init__(self) -> None:
        self.home_probability = _require_probability("home_probability", self.home_probability)
        self.draw_probability = _require_probability("draw_probability", self.draw_probability)
        self.away_probability = _require_probability("away_probability", self.away_probability)

        total = self.home_probability + self.draw_probability + self.away_probability
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"forecast probabilities must sum to 1.0, got {total:.8f}")

        for slot in OUTCOME_SLOTS:
            lower = _require_probability(f"{slot}_lower_bound", getattr(self, f"{slot}_lower_bound"))
            mean = getattr(self, f"{slot}_probability")
            upper = _require_probability(f"{slot}_upper_bound", getattr(self, f"{slot}_upper_bound"))
            if lower > mean or mean > upper:
                raise ValueError(f"{slot} bounds must satisfy lower <= mean <= upper")
            setattr(self, f"{slot}_lower_bound", lower)
            setattr(self, f"{slot}_upper_bound", upper)

        self.confidence = _require_probability("confidence", self.confidence)
        self.data_coverage_score = _require_probability(
            "data_coverage_score", self.data_coverage_score
        )
        if self.as_of_timestamp.tzinfo is None:
            self.as_of_timestamp = self.as_of_timestamp.replace(tzinfo=timezone.utc)
        if not self.forecast_id:
            self.forecast_id = self.compute_forecast_id()

    @property
    def probabilities(self) -> dict[str, float]:
        return {
            "home": self.home_probability,
            "draw": self.draw_probability,
            "away": self.away_probability,
        }

    @property
    def lower_bounds(self) -> dict[str, float]:
        return {
            "home": self.home_lower_bound,
            "draw": self.draw_lower_bound,
            "away": self.away_lower_bound,
        }

    @property
    def upper_bounds(self) -> dict[str, float]:
        return {
            "home": self.home_upper_bound,
            "draw": self.draw_upper_bound,
            "away": self.away_upper_bound,
        }

    def _identity_payload(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("forecast_id", None)
        return data

    def compute_forecast_id(self) -> str:
        return stable_hash(self._identity_payload())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["as_of_timestamp"] = self.as_of_timestamp.astimezone(timezone.utc).isoformat()
        return data


@dataclass
class AgentRecommendation:
    agent_name: str
    fixture_id: str
    outcome: str | None

    should_trade: bool
    abstain_reason: str | None

    probability_mean: float | None
    probability_lower_bound: float | None
    probability_upper_bound: float | None

    market_midpoint: float | None
    best_ask: float | None
    expected_fill_price: float | None

    gross_edge: float | None
    conservative_edge: float | None
    expected_value_after_costs: float | None

    signal_type: str
    evidence_ids: list[str]
    evidence_summary: str

    confidence: float
    data_coverage_score: float

    recommended_stake: float
    maximum_acceptable_price: float | None

    signal_created_at: datetime
    signal_expires_at: datetime | None

    correlation_key: str | None
    warnings: list[str] = field(default_factory=list)
    forecast_id: str | None = None

    def __post_init__(self) -> None:
        self.confidence = _require_probability("confidence", self.confidence)
        self.data_coverage_score = _require_probability(
            "data_coverage_score", self.data_coverage_score
        )
        self.recommended_stake = max(0.0, float(self.recommended_stake or 0.0))
        if self.signal_created_at.tzinfo is None:
            self.signal_created_at = self.signal_created_at.replace(tzinfo=timezone.utc)
        if self.signal_expires_at and self.signal_expires_at.tzinfo is None:
            self.signal_expires_at = self.signal_expires_at.replace(tzinfo=timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["signal_created_at"] = self.signal_created_at.astimezone(timezone.utc).isoformat()
        if self.signal_expires_at:
            data["signal_expires_at"] = self.signal_expires_at.astimezone(timezone.utc).isoformat()
        return data


def recommendation_correlation_key(
    *,
    fixture_id: str,
    outcome: str | None,
    forecast_id: str | None,
    evidence_ids: list[str],
    signal_type: str,
    catalyst: str | None = None,
    model_version: str | None = None,
) -> str:
    return stable_hash(
        {
            "fixture_id": fixture_id,
            "outcome": outcome,
            "forecast_id": forecast_id,
            "evidence_ids": sorted(evidence_ids or []),
            "signal_type": signal_type,
            "catalyst": catalyst,
            "model_version": model_version,
        }
    )

