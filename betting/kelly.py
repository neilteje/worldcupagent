"""
Kelly Criterion bet sizing for prediction market orders.

Kelly fraction formula for a binary YES bet:
  f* = (p - q) / (1 - q)

where:
  p = our model's probability that the outcome resolves YES (wins)
  q = current market price per share (cost to win $1)

A Kelly fraction > 0 means we have positive expected value vs the market.
We apply two caps:
  1. MAX_KELLY_FRACTION — never risk more than 20% of wallet on any single bet
  2. MAX_BET_USD        — hard USD ceiling

For multiple simultaneous bets (e.g. both a pre-match and HT position):
  - The pre-match reduces wallet for HT sizing
  - We use fractional Kelly (half-Kelly) by default for more conservative drawdown

Reference: Kelly (1956) "A New Interpretation of Information Rate"
"""
from __future__ import annotations
import math
import config


def kelly_fraction(model_prob: float, market_price: float) -> float:
    """
    Raw Kelly fraction for a single binary YES bet.
    Returns a value in [-1, 1]; negative means the bet has negative EV.
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    return (model_prob - market_price) / (1.0 - market_price)


def kelly_usd(
    model_prob: float,
    market_price: float,
    wallet_balance_usd: float,
    fraction: float = 0.5,    # use half-Kelly for risk management
) -> float:
    """
    Compute USD bet size using fractional Kelly.

    Args:
        model_prob:        our estimated probability the outcome wins
        market_price:      current Polymarket mid-price (cost per $1 share)
        wallet_balance_usd: current wallet balance
        fraction:          Kelly fraction multiplier (0.5 = half-Kelly)

    Returns:
        USD bet size rounded to 2 decimal places; 0 if no edge.
    """
    raw_f = kelly_fraction(model_prob, market_price)
    if raw_f <= 0:
        return 0.0

    # Apply fractional Kelly
    adjusted_f = raw_f * fraction

    # Cap at MAX_KELLY_FRACTION of wallet
    capped_f = min(adjusted_f, config.MAX_KELLY_FRACTION)

    usd = capped_f * wallet_balance_usd

    # Hard USD cap
    usd = min(usd, config.MAX_BET_USD)

    # Minimum viable order size ($1)
    if usd < 1.0:
        return 0.0

    return round(usd, 2)


def should_bet(
    model_prob: float,
    market_price: float,
    wallet_balance_usd: float,
) -> tuple[bool, float, str]:
    """
    Decide whether to place a bet and how much.

    Returns:
        (should_bet: bool, usd_size: float, reason: str)
    """
    edge = model_prob - market_price
    abs_edge = abs(edge)

    if abs_edge < config.MIN_EDGE:
        return (
            False,
            0.0,
            f"Edge {abs_edge:.3f} below MIN_EDGE threshold {config.MIN_EDGE}",
        )

    if edge < 0:
        return (
            False,
            0.0,
            f"Negative edge ({edge:.3f}): market overprices this outcome",
        )

    if wallet_balance_usd < 2.0:
        return (False, 0.0, f"Wallet balance ${wallet_balance_usd:.2f} too low")

    usd = kelly_usd(model_prob, market_price, wallet_balance_usd)
    if usd <= 0:
        return (False, 0.0, "Kelly sizing returned 0 — below minimum bet size")

    reason = (
        f"Edge={edge:.3f} (model={model_prob:.3f} vs market={market_price:.3f}), "
        f"Kelly size=${usd:.2f} ({100*usd/wallet_balance_usd:.1f}% of wallet)"
    )
    return (True, usd, reason)


def expected_value(model_prob: float, market_price: float, usd_size: float) -> float:
    """
    Expected profit in USD from placing this bet.
    Win: collect usd_size / market_price dollars (minus usd_size cost).
    Lose: lose usd_size.
    """
    if market_price <= 0:
        return 0.0
    win_payout = usd_size / market_price - usd_size
    return model_prob * win_payout - (1 - model_prob) * usd_size


def log_growth_rate(model_prob: float, market_price: float, fraction: float = 0.5) -> float:
    """
    Expected log-growth rate of wealth for this bet (Kelly objective).
    Useful for comparing two potential bets.
    """
    f = kelly_fraction(model_prob, market_price) * fraction
    if f <= 0 or f >= 1:
        return -math.inf
    p = model_prob
    q = 1 - p
    b = (1 - market_price) / market_price  # net odds on win
    return p * math.log(1 + b * f) + q * math.log(1 - f)
