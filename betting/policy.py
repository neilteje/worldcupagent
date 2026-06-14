"""
Shared order-selection policy — profile × forecast × market → sized picks.

This is the SINGLE place where an AgentProfile's policy knobs are applied to
the EV-ranked outcomes from betting/decision.py. Both the paper harness
(harness/paper_broker.py) and the live arena runner (live/cycle.py) call
select_picks(), so rehearsal and production cannot drift.

Layering (no double-gating):
  betting/decision.evaluate_game  — de-vig, EV per outcome, edge vs fair,
                                    Kelly sizing (profile.min_edge_vs_fair is
                                    the only edge bar)
  THIS MODULE                     — profile policy filters (windows, confidence,
                                    EV floor, scout veto, max-entry-price skew
                                    filter, max bets) + final stake sizing
  reasoning/gates.py              — optional risk overlay used by agent.py
                                    (wallet floor, consensus multipliers)
"""
from __future__ import annotations
from dataclasses import dataclass

from betting import decision as ev_decision
from harness.profiles import AgentProfile

# Polymarket CLOB enforces a $1.00 minimum per order; anything smaller is
# rejected at submission, so we don't emit it.
MIN_ORDER_USD = 1.00
BLITZ_DRAW_DISABLED_REASON = "blitz_draw_disabled"
KELLY_BELOW_MINIMUM_ORDER_SIZE = "kelly_below_minimum_order_size"


@dataclass
class SizedPick:
    slot: str               # "home" | "draw" | "away"
    code: str               # team code or "draw"
    stake_usd: float        # final sized stake (≥ MIN_ORDER_USD)
    entry_price: float      # raw YES mid we expect to pay
    limit_price: float      # mid + offset, what we actually submit
    our_prob: float
    fair_prob: float | None
    edge_vs_fair: float
    ev_per_dollar: float
    kelly_usd: float

    def to_dict(self) -> dict:
        return {
            "slot": self.slot, "code": self.code,
            "stake_usd": self.stake_usd, "entry_price": self.entry_price,
            "limit_price": self.limit_price, "our_prob": self.our_prob,
            "fair_prob": self.fair_prob, "edge_vs_fair": self.edge_vs_fair,
            "ev_per_dollar": self.ev_per_dollar, "kelly_usd": self.kelly_usd,
        }


def _high_flag_on(scout_flags, code: str) -> bool:
    for f in scout_flags or []:
        if (str(f.get("severity", "")).lower() == "high"
                and str(f.get("team", "")).lower() == str(code).lower()):
            return True
    return False


def is_draw_outcome(slot: str | None = None, code: str | None = None) -> bool:
    """Return True for every draw representation used by supported order paths."""
    values = {str(v or "").strip().lower() for v in (slot, code)}
    return bool(values & {"draw", "tie", "x"})


def suppress_blitz_draw_picks(
    profile: AgentProfile,
    picks: list[SizedPick],
    skip_reasons: list[str] | None = None,
) -> list[SizedPick]:
    """
    Remove draw candidates for BLITZ immediately before order creation.

    This deliberately runs after BLITZ's existing selection logic.  It preserves
    every non-draw pick already selected and never promotes a second-best outcome
    that BLITZ did not independently select.
    """
    if profile.name != "blitz":
        return picks
    kept: list[SizedPick] = []
    removed = 0
    for pick in picks:
        if is_draw_outcome(pick.slot, pick.code):
            removed += 1
            continue
        kept.append(pick)
    if removed and skip_reasons is not None:
        skip_reasons.extend([BLITZ_DRAW_DISABLED_REASON] * removed)
    return kept


def select_picks(
    profile: AgentProfile,
    probabilities: dict,
    moneyline: dict | None,
    home_code: str,
    away_code: str,
    bankroll: float,
    *,
    window: str = "PRE_MATCH",
    confidence_num: float = 0.5,
    scout_flags: list | None = None,
    limit_offset: float = 0.02,
    skip_reasons: list[str] | None = None,
) -> list[SizedPick]:
    """
    Apply one agent's full policy to a shared forecast.

    Returns sized picks ready to become orders/trades. `skip_reasons`, when
    provided, is appended with human-readable reasons (for ledger + metrics).
    """
    reasons = skip_reasons if skip_reasons is not None else []

    if window == "PRE_MATCH" and not profile.trade_prematch:
        reasons.append("profile does not trade PRE_MATCH")
        return []
    if window == "HT" and not profile.trade_halftime:
        reasons.append("profile does not trade HT")
        return []
    if confidence_num < profile.min_confidence:
        reasons.append(f"confidence {confidence_num:.2f} < floor {profile.min_confidence:.2f}")
        return []

    is_synthetic = bool(moneyline and moneyline.get("market_source") == "synthetic_demo")
    if is_synthetic and not profile.trade_synthetic:
        reasons.append("synthetic market — profile does not trade demo prices")
        return []

    game = ev_decision.evaluate_game(
        probabilities, moneyline, home_code, away_code,
        bankroll, kelly_fraction=profile.kelly_fraction,
        min_edge_vs_fair=profile.min_edge_vs_fair,
    )

    picks: list[SizedPick] = []
    for ev in game.ranked:
        if ev.raw_mid is None or ev.ev_per_dollar <= 0:
            continue
        if ev.edge_vs_fair < profile.min_edge_vs_fair:
            continue
        if ev.ev_per_dollar < profile.min_ev_per_dollar:
            reasons.append(f"{ev.slot}: EV/$ {ev.ev_per_dollar:.3f} < floor {profile.min_ev_per_dollar:.3f}")
            continue
        if profile.max_entry_price is not None and ev.raw_mid > profile.max_entry_price:
            reasons.append(f"{ev.slot}: price {ev.raw_mid:.2f} > max entry {profile.max_entry_price:.2f} (skew filter)")
            continue
        if profile.skip_on_high_scout_flag and _high_flag_on(scout_flags, ev.code):
            reasons.append(f"{ev.slot}: high-severity scout flag on {ev.code}")
            continue

        size = min(ev.kelly_usd, profile.max_bet_usd,
                   bankroll * profile.stake_cap_fraction, bankroll)
        if is_synthetic:
            size *= profile.synthetic_size_multiplier
        size = round(size, 2)
        if size < MIN_ORDER_USD:
            # A +EV bet we've decided to make shouldn't die to the venue's $1
            # minimum — bump it up to $1 if the profile allows and the wallet
            # (and cap) can cover it. Otherwise skip with a logged reason.
            # Floor-up is limited only by the hard caps (per-trade + wallet),
            # not the soft Kelly stake_cap_fraction.
            headroom = min(profile.max_bet_usd, bankroll)
            if profile.floor_to_min_order and headroom >= MIN_ORDER_USD:
                reasons.append(f"{ev.slot}: sized ${size:.2f} floored up to "
                               f"${MIN_ORDER_USD:.2f} (CLOB minimum)")
                size = MIN_ORDER_USD
            else:
                reason = KELLY_BELOW_MINIMUM_ORDER_SIZE if not profile.floor_to_min_order else (
                    "minimum_order_floor_unavailable"
                )
                reasons.append(f"{ev.slot}: {reason} sized ${size:.2f} < "
                               f"${MIN_ORDER_USD:.2f} CLOB minimum")
                continue

        picks.append(SizedPick(
            slot=ev.slot, code=ev.code, stake_usd=size,
            entry_price=float(ev.raw_mid),
            limit_price=round(min(0.99, float(ev.raw_mid) + limit_offset), 2),
            our_prob=ev.our_prob, fair_prob=ev.fair_prob,
            edge_vs_fair=ev.edge_vs_fair, ev_per_dollar=ev.ev_per_dollar,
            kelly_usd=ev.kelly_usd,
        ))
        if len(picks) >= profile.max_bets_per_window:
            break

    if not picks and not reasons:
        reasons.append("no outcome cleared the EV/edge bars")
    return picks


__all__ = [
    "BLITZ_DRAW_DISABLED_REASON",
    "KELLY_BELOW_MINIMUM_ORDER_SIZE",
    "MIN_ORDER_USD",
    "SizedPick",
    "is_draw_outcome",
    "select_picks",
    "suppress_blitz_draw_picks",
]
