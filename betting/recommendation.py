"""Build structured AgentRecommendations for the coordinated agents.

MONK, ANCHOR, and HUNTER no longer hand their sized picks straight to the order
path.  Each selected pick is first turned into an :class:`AgentRecommendation`
with an explicit, auditable conservative-edge calculation and a structured
abstain reason when any gate fails.  BLITZ never passes through here — it keeps
its existing direct order path (see ``live/cycle.act_for_agent``).

The probability *belief* used for the conservative edge comes from the
market-blind independent forecast snapshot (``MatchForecast``), never from the
market price.  The expected fill price is the only market input, and it is used
purely to measure edge, not to form the forecast.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import config
from betting.conservative import ConservativeEdgeConfig, calculate_conservative_edge
from betting.policy import MIN_ORDER_USD, SizedPick
from models.forecast_contracts import AgentRecommendation

COORDINATED_AGENTS = ("monk", "anchor", "hunter", "blitz")

# Structured abstain reasons (stable identifiers for ledger + metrics).
REASON_EXPECTED_FILL_UNAVAILABLE = "expected_fill_unavailable"
REASON_LOW_DATA_COVERAGE = "insufficient_data_coverage"
REASON_CONSERVATIVE_EDGE = "conservative_edge_below_threshold"
REASON_EV_AFTER_COSTS = "ev_after_costs_below_threshold"
REASON_BELOW_MIN_STAKE = "stake_below_venue_minimum"
REASON_ULTRA_TAIL_VALIDATION = "ultra_tail_validation_failed"
REASON_INSUFFICIENT_SIGNALS = "insufficient_independent_signals"


@dataclass(frozen=True)
class AgentEdgeThresholds:
    """Per-agent gates applied to a single candidate recommendation."""
    signal_type: str
    min_conservative_edge: float
    min_ev_after_costs: float = 0.0
    min_data_coverage: float = 0.0
    min_independent_signals: int = 0
    ultra_tail_price: float = 0.0          # fills below this are "ultra tail"
    ultra_tail_validation_enabled: bool = False


def thresholds_for(agent_name: str) -> AgentEdgeThresholds:
    """Resolve the edge thresholds for one coordinated agent from config."""
    name = (agent_name or "").strip().lower()
    if name in COORDINATED_AGENTS:
        return AgentEdgeThresholds(
            signal_type="legacy_blitz_value",
            min_conservative_edge=-1.0,
            min_ev_after_costs=0.0,
            min_data_coverage=0.0,
            min_independent_signals=0,
        )
    raise ValueError(f"{agent_name!r} is not a coordinated agent {COORDINATED_AGENTS}")


def _edge_config() -> ConservativeEdgeConfig:
    return ConservativeEdgeConfig(
        fee_buffer=config.FEE_BUFFER,
        slippage_buffer=config.SLIPPAGE_BUFFER,
        model_risk_buffer=config.MODEL_RISK_BUFFER,
    )


def _band_for_code(forecast_snapshot: dict | None, code: str, fallback_mean: float) -> tuple[float, float, float]:
    """Return (mean, lower, upper) belief for ``code`` from the market-blind snapshot.

    Falls back to a symmetric band around ``fallback_mean`` (the council prob)
    when the independent snapshot has no entry for this outcome — e.g. the HT
    window, which does not build a full ``MatchForecast``.
    """
    snap = forecast_snapshot or {}
    means = snap.get("probabilities_by_code") or {}
    lowers = snap.get("lower_bounds_by_code") or {}
    uppers = snap.get("upper_bounds_by_code") or {}
    if code in means:
        mean = float(means[code])
        lower = float(lowers.get(code, max(0.0, mean - 0.08)))
        upper = float(uppers.get(code, min(1.0, mean + 0.08)))
        return mean, lower, upper
    mean = max(0.0, min(1.0, float(fallback_mean)))
    return mean, max(0.0, mean - 0.08), min(1.0, mean + 0.08)


def build_recommendation(
    agent_name: str,
    pick: SizedPick,
    *,
    fixture_id: str,
    bankroll: float,
    forecast_snapshot: dict | None,
    forecast_id: str | None,
    evidence_ids: list[str],
    data_coverage_score: float,
    confidence: float,
    thresholds: AgentEdgeThresholds | None = None,
    edge_config: ConservativeEdgeConfig | None = None,
    now: datetime | None = None,
) -> AgentRecommendation:
    """Convert one sized pick into a structured, gated recommendation.

    The returned recommendation always carries the full conservative-edge
    accounting.  ``should_trade`` is ``False`` with a structured
    ``abstain_reason`` whenever any gate fails; callers must honour it.
    """
    th = thresholds or thresholds_for(agent_name)
    cfg = edge_config or _edge_config()
    created = now or datetime.now(timezone.utc)
    evidence_ids = list(evidence_ids or [])

    code = pick.code
    mean, lower, upper = _band_for_code(forecast_snapshot, code, pick.our_prob)

    expected_fill = float(pick.entry_price) if pick.entry_price else 0.0
    stake_usd = float(pick.stake_usd or 0.0)
    stake_fraction = round(stake_usd / bankroll, 6) if bankroll > 0 else 0.0

    edge = calculate_conservative_edge(
        probability_mean=mean,
        probability_lower_bound=lower,
        expected_fill_price=expected_fill if expected_fill > 0 else 1.0,
        config=cfg,
    )

    # ── Gates (first failure wins; everything else is still populated) ──────
    abstain_reason: str | None = None
    if not (pick.entry_price and float(pick.entry_price) > 0):
        abstain_reason = REASON_EXPECTED_FILL_UNAVAILABLE
    elif data_coverage_score < th.min_data_coverage:
        abstain_reason = REASON_LOW_DATA_COVERAGE
    elif th.min_independent_signals and len(evidence_ids) < th.min_independent_signals:
        abstain_reason = REASON_INSUFFICIENT_SIGNALS
    elif (th.ultra_tail_price and expected_fill < th.ultra_tail_price
          and not th.ultra_tail_validation_enabled):
        abstain_reason = REASON_ULTRA_TAIL_VALIDATION
    elif edge.conservative_edge < th.min_conservative_edge:
        abstain_reason = REASON_CONSERVATIVE_EDGE
    elif edge.expected_value_after_costs < th.min_ev_after_costs:
        abstain_reason = REASON_EV_AFTER_COSTS
    elif stake_usd < MIN_ORDER_USD:
        abstain_reason = REASON_BELOW_MIN_STAKE

    correlation_key = f"{agent_name}:{fixture_id}:{code}:{th.signal_type}:{forecast_id or ''}"

    return AgentRecommendation(
        agent_name=agent_name,
        fixture_id=fixture_id,
        outcome=code,
        should_trade=abstain_reason is None,
        abstain_reason=abstain_reason,
        probability_mean=round(mean, 6),
        probability_lower_bound=round(lower, 6),
        probability_upper_bound=round(upper, 6),
        market_midpoint=round(expected_fill, 6) if expected_fill > 0 else None,
        best_ask=float(pick.limit_price) if pick.limit_price else None,
        expected_fill_price=round(expected_fill, 6) if expected_fill > 0 else None,
        gross_edge=edge.gross_edge,
        conservative_edge=edge.conservative_edge,
        expected_value_after_costs=edge.expected_value_after_costs,
        signal_type=th.signal_type,
        evidence_ids=evidence_ids,
        evidence_summary=f"{len(evidence_ids)} independent evidence id(s)",
        confidence=max(0.0, min(1.0, float(confidence))),
        data_coverage_score=max(0.0, min(1.0, float(data_coverage_score))),
        recommended_stake=stake_fraction,
        maximum_acceptable_price=float(pick.limit_price) if pick.limit_price else None,
        signal_created_at=created,
        signal_expires_at=None,
        correlation_key=correlation_key,
        forecast_id=forecast_id,
    )


__all__ = [
    "AgentEdgeThresholds",
    "COORDINATED_AGENTS",
    "REASON_BELOW_MIN_STAKE",
    "REASON_CONSERVATIVE_EDGE",
    "REASON_EV_AFTER_COSTS",
    "REASON_EXPECTED_FILL_UNAVAILABLE",
    "REASON_INSUFFICIENT_SIGNALS",
    "REASON_LOW_DATA_COVERAGE",
    "REASON_ULTRA_TAIL_VALIDATION",
    "build_recommendation",
    "thresholds_for",
]
