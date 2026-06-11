"""
EV-ranked decision engine — evaluates EVERY outcome, not just the favorite.

The old flow checked the council's single most-likely outcome against the market
and skipped whenever that one outcome lacked edge. But the argmax outcome is
almost always the market favorite too — the most efficiently priced slot — so the
agent looked for edge in the one place least likely to have it and skipped nearly
every game. The value in a 3-way market usually sits on the draw or the underdog,
which never got evaluated.

This module fixes that. Given our full probability distribution and the raw market
mids, it:

  1. De-vigs the market (normalizes the YES mids to sum to 1) so we can see the
     market's *fair* implied probability, stripped of the overround we'd otherwise
     mistake for negative edge.
  2. Computes, for ALL outcomes (home / draw / away):
       - edge vs the fair (de-vigged) price        → genuine disagreement
       - EV per $1 staked vs the raw price we'd pay → real expected profit
       - half-Kelly USD size on the raw price
  3. Ranks outcomes by EV and returns a single structured decision for the game —
     a concrete BET (side + size) or a HOLD, but always with the full per-outcome
     payout table so there is a grounded decision on every game, never a bare skip.

Betting requires EV > 0, which means our_prob > raw_mid (you must beat the price
you actually pay). We additionally require the edge vs the *fair* price to clear a
small bar so we are capturing genuine mispricing rather than noise inside the vig.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import config


SLOTS = ("home", "draw", "away")


@dataclass
class OutcomeEval:
    slot: str                 # "home" | "draw" | "away"
    code: str                 # team code or "draw"
    our_prob: float           # council probability for this outcome
    raw_mid: float | None     # market YES mid we'd actually pay
    fair_prob: float | None   # de-vigged market implied probability
    edge_vs_fair: float       # our_prob - fair_prob (genuine disagreement)
    ev_per_dollar: float      # expected profit per $1 staked at raw_mid
    kelly_usd: float          # half-Kelly size on raw_mid (pre-gate)
    tradable: bool            # EV > 0 and edge clears the fair-price bar


@dataclass
class GameDecision:
    should_trade: bool
    best: OutcomeEval | None              # the recommended outcome (may be HOLD)
    ranked: list[OutcomeEval] = field(default_factory=list)   # all outcomes, EV-desc
    overround: float | None = None        # market vig (sum of raw mids - 1)
    summary: str = ""


def _norm_mids(moneyline: dict | None) -> dict[str, float | None]:
    """Raw YES mid per slot from a Polymarket moneyline dict."""
    out: dict[str, float | None] = {"home": None, "draw": None, "away": None}
    if not moneyline:
        return out
    for slot in SLOTS:
        out[slot] = (moneyline.get("outcomes", {}).get(slot) or {}).get("current_mid_yes")
    return out


def devig(mids: dict[str, float | None]) -> tuple[dict[str, float | None], float | None]:
    """
    Normalize present YES mids to sum to 1 (proportional de-vig).
    Returns (fair_probs, overround). overround = sum(raw mids) - 1.
    """
    present = {k: float(v) for k, v in mids.items() if isinstance(v, (int, float))}
    s = sum(present.values())
    if s <= 0:
        return {k: None for k in mids}, None
    fair = {k: (round(present[k] / s, 4) if k in present else None) for k in mids}
    return fair, round(s - 1.0, 4)


def _ev_per_dollar(our_prob: float, raw_mid: float) -> float:
    """Expected profit per $1 staked on a YES share priced at raw_mid."""
    if raw_mid <= 0 or raw_mid >= 1:
        return 0.0
    # Win: receive 1/raw_mid per $1 (net +(1-raw_mid)/raw_mid); lose: -1.
    return our_prob * (1.0 / raw_mid) - 1.0


def _slot_code(slot: str, home_code: str, away_code: str) -> str:
    return {"home": home_code, "draw": "draw", "away": away_code}[slot]


def _prob_for_slot(probabilities: dict, slot: str,
                   home_code: str, away_code: str) -> float:
    """Pull our probability for a slot from the council distribution."""
    key = _slot_code(slot, home_code, away_code)
    val = (probabilities or {}).get(key)
    if not isinstance(val, (int, float)):
        # tolerate distributions keyed by slot name
        val = (probabilities or {}).get(slot)
    return float(val) if isinstance(val, (int, float)) else 0.0


def evaluate_game(
    probabilities: dict,
    moneyline: dict | None,
    home_code: str,
    away_code: str,
    wallet_balance: float,
    kelly_fraction: float = 0.5,
    min_edge_vs_fair: float | None = None,
) -> GameDecision:
    """
    Rank all three outcomes by EV and return the game's decision.

    This is the SINGLE edge gate in the stack: tradability requires EV > 0 at
    the raw price AND edge ≥ `min_edge_vs_fair` against the de-vigged fair
    price. `reasoning/gates.py` adds a risk overlay (wallet, scout veto,
    consensus/confidence multipliers) but never re-applies an edge bar.

    Args:
        probabilities:     council distribution {home_code: .., "draw": .., away_code: ..}
        moneyline:         Polymarket moneyline dict (or None → no tradable market)
        wallet_balance:    current bankroll for Kelly sizing
        kelly_fraction:    fractional Kelly (0.5 = half-Kelly)
        min_edge_vs_fair:  edge bar vs fair price; None → config.MIN_EDGE_VS_FAIR.
                           Agent profiles pass their own bar here.
    """
    from betting.kelly import kelly_usd  # local import avoids cycle

    edge_bar = config.MIN_EDGE_VS_FAIR if min_edge_vs_fair is None else float(min_edge_vs_fair)
    mids = _norm_mids(moneyline)
    fair, overround = devig(mids)

    evals: list[OutcomeEval] = []
    for slot in SLOTS:
        code = _slot_code(slot, home_code, away_code)
        our_p = _prob_for_slot(probabilities, slot, home_code, away_code)
        raw = mids[slot]
        fair_p = fair[slot]

        if raw is None:
            evals.append(OutcomeEval(slot, code, our_p, None, fair_p,
                                     edge_vs_fair=0.0, ev_per_dollar=0.0,
                                     kelly_usd=0.0, tradable=False))
            continue

        edge_fair = our_p - fair_p if fair_p is not None else 0.0
        ev = _ev_per_dollar(our_p, raw)
        size = kelly_usd(our_p, raw, wallet_balance, fraction=kelly_fraction)
        tradable = (ev > 0) and (edge_fair >= edge_bar)
        evals.append(OutcomeEval(slot, code, our_p, raw, fair_p,
                                 edge_vs_fair=round(edge_fair, 4),
                                 ev_per_dollar=round(ev, 4),
                                 kelly_usd=size, tradable=tradable))

    ranked = sorted(evals, key=lambda e: e.ev_per_dollar, reverse=True)
    tradables = [e for e in ranked if e.tradable]
    best = tradables[0] if tradables else (ranked[0] if ranked else None)

    if best is None:
        summary = "no market — prediction only"
    elif tradables:
        summary = (f"BET {best.code}: EV {best.ev_per_dollar*100:+.1f}%/$, "
                   f"edge {best.edge_vs_fair*100:+.1f}pp vs fair "
                   f"(our {best.our_prob:.0%} vs pay {best.raw_mid:.0%})")
    else:
        summary = (f"HOLD — best is {best.code} at edge "
                   f"{best.edge_vs_fair*100:+.1f}pp vs fair, "
                   f"below {edge_bar*100:.1f}pp bar")

    return GameDecision(
        should_trade=bool(tradables),
        best=best,
        ranked=ranked,
        overround=overround,
        summary=summary,
    )
