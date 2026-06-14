"""Portfolio coordination for MONK, ANCHOR, and HUNTER recommendations.

The live BLITZ path is intentionally not controlled here.  BLITZ recommendations
may be included for aggregate exposure reporting, but allocator rejections are
only applied to non-BLITZ agents.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from models.forecast_contracts import AgentRecommendation


COORDINATED_AGENTS = {"monk", "anchor", "hunter"}
OBSERVED_ONLY_AGENTS = {"blitz"}


@dataclass(frozen=True)
class PortfolioLimits:
    max_fixture_exposure: float = 0.04
    max_outcome_exposure: float = 0.03
    max_ultra_tail_exposure: float = 0.01
    max_daily_drawdown: float = 0.05


@dataclass
class AllocationResult:
    accepted: list[AgentRecommendation] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    observed_only: list[AgentRecommendation] = field(default_factory=list)
    duplicate_recommendations: int = 0

    def to_dict(self) -> dict:
        return {
            "accepted": [r.to_dict() for r in self.accepted],
            "rejected": self.rejected,
            "observed_only": [r.to_dict() for r in self.observed_only],
            "duplicate_recommendations": self.duplicate_recommendations,
        }


def allocate_recommendations(
    recommendations: list[AgentRecommendation],
    *,
    current_exposure: dict[str, float] | None = None,
    limits: PortfolioLimits | None = None,
) -> AllocationResult:
    """Deduplicate and exposure-gate MONK/ANCHOR/HUNTER recommendations."""
    limits = limits or PortfolioLimits()
    exposure = dict(current_exposure or {})
    result = AllocationResult()
    seen: set[str] = set()

    for rec in recommendations:
        agent = rec.agent_name.lower()
        if agent in OBSERVED_ONLY_AGENTS:
            result.observed_only.append(rec)
            continue
        if agent not in COORDINATED_AGENTS:
            result.rejected.append({"recommendation": rec.to_dict(), "reason": "unknown_agent"})
            continue
        if not rec.should_trade:
            result.rejected.append({
                "recommendation": rec.to_dict(),
                "reason": rec.abstain_reason or "agent_abstained",
            })
            continue

        key = rec.correlation_key or f"{rec.fixture_id}:{rec.outcome}:{rec.forecast_id}"
        if key in seen:
            result.duplicate_recommendations += 1
            result.rejected.append({"recommendation": rec.to_dict(), "reason": "duplicate_signal"})
            continue

        fixture_key = f"fixture:{rec.fixture_id}"
        outcome_key = f"outcome:{rec.fixture_id}:{rec.outcome}"
        stake = float(rec.recommended_stake or 0.0)
        if exposure.get(fixture_key, 0.0) + stake > limits.max_fixture_exposure:
            result.rejected.append({"recommendation": rec.to_dict(), "reason": "fixture_exposure_limit"})
            continue
        if exposure.get(outcome_key, 0.0) + stake > limits.max_outcome_exposure:
            result.rejected.append({"recommendation": rec.to_dict(), "reason": "outcome_exposure_limit"})
            continue
        if (rec.expected_fill_price or 1.0) < 0.05:
            tail_key = f"ultra_tail:{rec.fixture_id}"
            if exposure.get(tail_key, 0.0) + stake > limits.max_ultra_tail_exposure:
                result.rejected.append({"recommendation": rec.to_dict(), "reason": "ultra_tail_exposure_limit"})
                continue
            exposure[tail_key] = exposure.get(tail_key, 0.0) + stake

        seen.add(key)
        exposure[fixture_key] = exposure.get(fixture_key, 0.0) + stake
        exposure[outcome_key] = exposure.get(outcome_key, 0.0) + stake
        result.accepted.append(rec)

    return result

