"""
Deterministic decision gates — the non-LLM RISK OVERLAY between the council's
view and the order book.

`evaluate_gates` is a pure function: same inputs → same decision. It can VETO a
trade or scale its size (a multiplier applied on top of Kelly), and returns the
audit trail so the agent can log exactly why it sized or skipped a bet.

Edge gating lives in ONE place: `betting/decision.py` (edge vs the de-vigged
fair price, threshold supplied by the agent profile). The legacy raw-mid edge
bar here (Gate 2) is therefore OPT-IN: callers using the EV engine pass
`min_edge=None` (the default) and the gate is skipped; pass a float to restore
the standalone bar. The scout veto is profile-configurable via `scout_veto`.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import config
from data.kalshi import cross_market_signal


@dataclass
class GateResult:
    should_trade: bool
    bet_multiplier: float                 # scales the Kelly USD size (0 = no trade)
    edge: float                           # model_prob − pm_mid (signed)
    reasons: list[str] = field(default_factory=list)   # every gate's verdict
    veto_reason: str | None = None
    market_agreement: str = "n/a"


def evaluate_gates(
    outcome: str,
    model_prob: float,
    pm_mid: float | None,
    kalshi_mid: float | None,
    scout_flags: list[dict] | None,
    confidence: str,
    wallet_balance: float,
    *,
    min_edge: float | None = None,
    scout_veto: bool = True,
) -> GateResult:
    reasons: list[str] = []
    multiplier = 1.0
    scout_flags = scout_flags or []
    edge = (model_prob - pm_mid) if pm_mid is not None else 0.0

    # Gate 0 — wallet floor (capital preservation).
    if wallet_balance < config.MIN_WALLET_USD:
        return GateResult(False, 0.0, edge,
                          reasons=[f"wallet ${wallet_balance:.2f} < floor "
                                   f"${config.MIN_WALLET_USD:.2f}"],
                          veto_reason="wallet_floor")

    # Gate 1 — a tradable market must exist with a usable price.
    if pm_mid is None:
        return GateResult(False, 0.0, edge,
                          reasons=["no Polymarket mid for this outcome"],
                          veto_reason="no_market")

    # Gate 2 — OPT-IN raw-mid edge bar. Skipped (min_edge=None) when the EV
    # engine already enforced the profile's edge-vs-fair bar upstream.
    if min_edge is not None:
        if edge < min_edge:
            return GateResult(False, 0.0, edge,
                              reasons=[f"edge {edge*100:+.1f}pp below "
                                       f"{min_edge*100:.1f}pp threshold"],
                              veto_reason="insufficient_edge")
        reasons.append(f"edge {edge*100:+.1f}pp clears {min_edge*100:.1f}pp threshold")
    else:
        reasons.append("edge bar handled by EV engine (vs fair price)")

    # Gate 3 — cross-market consensus (Polymarket vs Kalshi).
    signal = cross_market_signal(pm_mid, kalshi_mid)
    agreement = signal["agreement"]
    if agreement == "consensus":
        multiplier *= config.CONSENSUS_MULTIPLIER
        reasons.append(f"markets agree (spread {signal['spread']:.3f}) "
                       f"→ ×{config.CONSENSUS_MULTIPLIER}")
    elif agreement == "contested":
        multiplier *= config.CONTESTED_MULTIPLIER
        reasons.append(f"markets contested (spread {signal['spread']:.3f}) "
                       f"→ ×{config.CONTESTED_MULTIPLIER}")
    elif agreement == "n/a":
        reasons.append("no Kalshi cross-check available")

    # Gate 4 — a high-severity scout flag on our predicted side is a hard veto
    # (profile-configurable: blitz runs with scout_veto=False).
    if scout_veto:
        for f in scout_flags:
            if str(f.get("severity", "")).lower() != "high":
                continue
            if str(f.get("team", "")).lower() == outcome.lower():
                return GateResult(False, 0.0, edge,
                                  reasons=reasons + [f"VETO: high-severity flag on "
                                                     f"{outcome}: {f.get('signal')}"],
                                  veto_reason="scout_high_flag",
                                  market_agreement=agreement)
    else:
        reasons.append("scout veto disabled by profile")

    # Gate 5 — council confidence scaling.
    conf = (confidence or "").lower()
    if conf == "low":
        multiplier *= config.CONFIDENCE_LOW_MULTIPLIER
        reasons.append(f"low confidence → ×{config.CONFIDENCE_LOW_MULTIPLIER}")
    elif conf == "high":
        multiplier *= config.CONFIDENCE_HIGH_MULTIPLIER
        reasons.append(f"high confidence → ×{config.CONFIDENCE_HIGH_MULTIPLIER}")

    return GateResult(
        should_trade=True,
        bet_multiplier=round(multiplier, 3),
        edge=edge,
        reasons=reasons,
        market_agreement=agreement,
    )
