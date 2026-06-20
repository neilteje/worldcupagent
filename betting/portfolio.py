"""Order-invariant portfolio coordination for all agent recommendations."""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from models.forecast_contracts import AgentRecommendation


COORDINATED_AGENTS = {"monk", "anchor", "hunter", "blitz"}
OBSERVED_ONLY_AGENTS: set[str] = set()

def _joint_sort_key(rec: "AgentRecommendation"):
    """Total, deterministic ordering so joint allocation is invariant to the
    order recommendations arrive in (acceptance criterion §22/#24)."""
    edge = rec.conservative_edge if rec.conservative_edge is not None else -1.0
    return (
        -float(edge),
        (rec.agent_name or "").lower(),
        rec.correlation_key or f"{rec.fixture_id}:{rec.outcome}",
    )


@dataclass(frozen=True)
class PortfolioLimits:
    max_fixture_exposure: float = config.MAX_FIXTURE_EXPOSURE
    max_outcome_exposure: float = config.MAX_OUTCOME_EXPOSURE
    max_ultra_tail_exposure: float = config.MAX_ULTRA_TAIL_EXPOSURE
    max_daily_drawdown: float = config.MAX_DAILY_DRAWDOWN


@dataclass
class AllocationResult:
    accepted: list[AgentRecommendation] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    observed_only: list[AgentRecommendation] = field(default_factory=list)
    duplicate_recommendations: int = 0
    exposure: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "accepted": [r.to_dict() for r in self.accepted],
            "rejected": self.rejected,
            "observed_only": [r.to_dict() for r in self.observed_only],
            "duplicate_recommendations": self.duplicate_recommendations,
            "exposure": self.exposure,
        }


def allocate_recommendations(
    recommendations: list[AgentRecommendation],
    *,
    current_exposure: dict[str, float] | None = None,
    limits: PortfolioLimits | None = None,
    seen: set[str] | None = None,
) -> AllocationResult:
    """Deduplicate and exposure-gate MONK/ANCHOR/HUNTER recommendations.

    ``current_exposure`` and ``seen`` may be threaded across successive calls
    (one per coordinated agent within a window) so that the allocator acts as a
    single central book: a signal already backed by an earlier agent, or
    exposure already committed, constrains later agents.  Both are copied, so
    callers carry forward ``result.exposure`` and reuse the same ``seen`` set.
    """
    limits = limits or PortfolioLimits()
    exposure = dict(current_exposure or {})
    result = AllocationResult()
    seen = seen if seen is not None else set()

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

        key = f"{agent}:{rec.correlation_key or f'{rec.fixture_id}:{rec.outcome}:{rec.forecast_id}'}"
        if key in seen:
            result.duplicate_recommendations += 1
            result.rejected.append({"recommendation": rec.to_dict(), "reason": "duplicate_signal"})
            continue

        fixture_key = f"{agent}:fixture:{rec.fixture_id}"
        outcome_key = f"{agent}:outcome:{rec.fixture_id}:{rec.outcome}"
        stake = float(rec.recommended_stake or 0.0)
        if exposure.get(fixture_key, 0.0) + stake > limits.max_fixture_exposure:
            result.rejected.append({"recommendation": rec.to_dict(), "reason": "fixture_exposure_limit"})
            continue
        if exposure.get(outcome_key, 0.0) + stake > limits.max_outcome_exposure:
            result.rejected.append({"recommendation": rec.to_dict(), "reason": "outcome_exposure_limit"})
            continue
        if (rec.expected_fill_price or 1.0) < 0.05:
            tail_key = f"{agent}:ultra_tail:{rec.fixture_id}"
            if exposure.get(tail_key, 0.0) + stake > limits.max_ultra_tail_exposure:
                result.rejected.append({"recommendation": rec.to_dict(), "reason": "ultra_tail_exposure_limit"})
                continue
            exposure[tail_key] = exposure.get(tail_key, 0.0) + stake

        seen.add(key)
        exposure[fixture_key] = exposure.get(fixture_key, 0.0) + stake
        exposure[outcome_key] = exposure.get(outcome_key, 0.0) + stake
        result.accepted.append(rec)

    result.exposure = exposure
    return result


def allocate_jointly(
    recommendations: list[AgentRecommendation],
    *,
    limits: PortfolioLimits | None = None,
) -> AllocationResult:
    """Allocate MONK/ANCHOR/HUNTER recommendations JOINTLY (spec §22).

    Unlike sequential per-agent allocation, this sorts the combined pool by a
    deterministic mandate/edge key before applying dedup + exposure caps, so the
    accepted set is INVARIANT to the order recommendations are supplied in.
    BLITZ recommendations are observed but never gated.
    """
    ordered = sorted(recommendations, key=_joint_sort_key)
    return allocate_recommendations(ordered, limits=limits)


@dataclass
class PortfolioCoordinator:
    """Stateful wrapper that threads one allocation book across coordinated agents.

    Agents in a window run sequentially; each call to :meth:`allocate` carries
    forward the committed exposure and the set of signals already backed, so the
    coordinator behaves as one central allocator even though execution stays
    per-agent.  Counters accumulate for reporting.  BLITZ is never routed here.
    """
    limits: PortfolioLimits = field(default_factory=PortfolioLimits)
    exposure: dict[str, float] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)
    duplicate_recommendations: int = 0
    duplicate_positions_prevented: int = 0
    rejected: list[dict] = field(default_factory=list)

    def allocate(self, recommendations: list[AgentRecommendation]) -> AllocationResult:
        result = allocate_recommendations(
            recommendations,
            current_exposure=self.exposure,
            limits=self.limits,
            seen=self.seen,
        )
        self.exposure = result.exposure
        self.duplicate_recommendations += result.duplicate_recommendations
        self.duplicate_positions_prevented += result.duplicate_recommendations
        self.rejected.extend(result.rejected)
        return result

    def allocate_jointly(self, recommendations: list[AgentRecommendation]) -> AllocationResult:
        """Order-invariant joint allocation over the full coordinated pool.

        Sorts by the deterministic mandate/edge key, then runs allocation
        threading the coordinator's existing book so prior commitments still
        constrain. Use this when all coordinated recommendations for a window
        are collected up front (spec §22)."""
        ordered = sorted(recommendations, key=_joint_sort_key)
        return self.allocate(ordered)

