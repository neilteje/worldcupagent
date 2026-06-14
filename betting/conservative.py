"""Cost-adjusted conservative edge calculations for coordinated agents."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConservativeEdgeConfig:
    fee_buffer: float = 0.01
    slippage_buffer: float = 0.01
    model_risk_buffer: float = 0.02


@dataclass(frozen=True)
class ConservativeEdge:
    gross_edge: float
    conservative_edge: float
    expected_value_after_costs: float


def calculate_conservative_edge(
    *,
    probability_mean: float,
    probability_lower_bound: float,
    expected_fill_price: float,
    config: ConservativeEdgeConfig | None = None,
) -> ConservativeEdge:
    """Return gross edge, lower-bound edge after costs, and EV after costs."""
    cfg = config or ConservativeEdgeConfig()
    gross = float(probability_mean) - float(expected_fill_price)
    costs = cfg.fee_buffer + cfg.slippage_buffer + cfg.model_risk_buffer
    conservative = float(probability_lower_bound) - float(expected_fill_price) - costs
    ev_after_costs = gross - costs
    return ConservativeEdge(
        gross_edge=round(gross, 6),
        conservative_edge=round(conservative, 6),
        expected_value_after_costs=round(ev_after_costs, 6),
    )

