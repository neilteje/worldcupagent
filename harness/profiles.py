"""
Agent aggressiveness profiles — the SINGLE source of truth for trading policy.

A profile is the trading policy layered on top of a shared forecast. The
forecast (council probabilities + confidence) is identical across agents;
profiles only change risk appetite. Both the arena agent (`agent.py
--profile` / `AGENT_PROFILE` env) and the harness load profiles from here, so
there is no drift between rehearsal and production.

## The four agents (mandates per docs/STRATEGY.md §4)

  monk   → ORACLE  — pure forecast quality (Stair Score track). Predicts every
           window; trades only enormous (≥10pp) edges. A handful of bets all
           tournament is the *design*, not paralysis.
  anchor → KEEL    — disciplined all-outcome EV accumulator. The control arm
           and risk-adjusted P&L play. ~20–40% of games.
  hunter → SAW     — skew harvester. Only buys outcomes priced ≤ 0.40 (draws
           and underdogs); near-max size when it fires. Variance with positive
           drift is the product.
  blitz  → SURGE   — event-driven aggression: both windows, scout veto off,
           thin-but-positive edges vs FAIR price. Endgame escalation vehicle.

## Which knob applies where (no double-gating)

  min_edge_vs_fair, min_ev_per_dollar → betting/decision.py tradability
                                        (edge measured vs de-vigged fair price)
  min_confidence, max_bets_per_window,
  trade_prematch/halftime              → policy filters before sizing
  skip_on_high_scout_flag              → reasoning/gates.py scout veto
  kelly_fraction, max_bet_usd,
  stake_cap_fraction                   → sizing (gates may scale via consensus/
                                        confidence multipliers, never re-gate edge)
  trade_synthetic (+ multiplier)       → harness-only honesty policy: synthetic
                                        demo markets are derived from our own
                                        forecast + noise, so "edge" against them
                                        is noise. Default: do not bet them.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path


# Council confidence is a label; map to a number for floor comparisons.
_CONFIDENCE_NUM = {"low": 0.40, "medium": 0.60, "high": 0.80, "": 0.50}


def confidence_to_num(conf: str | float | None) -> float:
    if isinstance(conf, (int, float)):
        return float(conf)
    return _CONFIDENCE_NUM.get(str(conf or "").strip().lower(), 0.50)


@dataclass
class AgentProfile:
    name: str                              # machine id, e.g. "anchor"
    label: str                             # human label
    bankroll: float = 100.0                # starting paper bankroll (harness)

    # ── When to bet (drives betting/decision.py tradability) ─────────────
    min_edge_vs_fair: float = 0.045        # edge vs DE-VIGGED fair price
    min_ev_per_dollar: float = 0.0         # EV/$ floor (0 = any +EV)
    min_confidence: float = 0.40           # numeric confidence floor
    max_entry_price: float | None = None   # only buy outcomes priced ≤ this (skew filter)
    trade_prematch: bool = True
    trade_halftime: bool = True
    skip_on_high_scout_flag: bool = True   # gates scout veto on our side

    # ── How much to bet ──────────────────────────────────────────────────
    kelly_fraction: float = 0.40           # fraction of full Kelly
    max_bet_usd: float = 5.0               # hard per-trade cap (arena rule ≤ $5)
    stake_cap_fraction: float = 0.10       # per-bet cap as fraction of bankroll
    max_bets_per_window: int = 1           # outcomes backed per window

    # ── Market honesty (harness) ─────────────────────────────────────────
    trade_synthetic: bool = False          # bet synthetic_demo markets at all?
    synthetic_size_multiplier: float = 0.25  # size-down when trading synthetic

    def to_dict(self) -> dict:
        return asdict(self)


# ── The four tuned agents ───────────────────────────────────────────────────

MONK = AgentProfile(
    name="monk", label="oracle — forecast specialist",
    min_edge_vs_fair=0.10,      # trades only enormous, genuine disagreement
    min_ev_per_dollar=0.08,     # and a real EV cushion
    min_confidence=0.55,        # medium+ council confidence
    kelly_fraction=0.20,
    max_bet_usd=2.0,
    stake_cap_fraction=0.04,
    max_bets_per_window=1,
    skip_on_high_scout_flag=True,
)

ANCHOR = AgentProfile(
    name="anchor", label="keel — disciplined EV accumulator",
    min_edge_vs_fair=0.045,
    min_ev_per_dollar=0.02,
    min_confidence=0.40,        # low-confidence allowed: the bars do the work
    kelly_fraction=0.35,
    max_bet_usd=4.0,
    stake_cap_fraction=0.08,
    max_bets_per_window=1,
    skip_on_high_scout_flag=True,
)

HUNTER = AgentProfile(
    name="hunter", label="saw — skew harvester (draws + dogs only)",
    min_edge_vs_fair=0.03,
    min_ev_per_dollar=0.01,
    min_confidence=0.40,
    max_entry_price=0.40,       # never buys favorites — payout asymmetry is the product
    kelly_fraction=0.75,        # when it fires, fire near the cap
    max_bet_usd=5.0,
    stake_cap_fraction=0.10,
    max_bets_per_window=2,
    skip_on_high_scout_flag=True,
)

BLITZ = AgentProfile(
    name="blitz", label="surge — event-driven aggression",
    min_edge_vs_fair=0.02,      # thin edges — but still +EV vs FAIR price
    min_ev_per_dollar=0.0,
    min_confidence=0.35,        # fires even on low-confidence reads
    kelly_fraction=0.65,
    max_bet_usd=5.0,
    stake_cap_fraction=0.15,
    max_bets_per_window=2,
    skip_on_high_scout_flag=False,
    trade_synthetic=True,       # may trade demo markets, but ×0.25 sized
    synthetic_size_multiplier=0.25,
)

DEFAULT_PROFILES: dict[str, AgentProfile] = {
    p.name: p for p in (MONK, ANCHOR, HUNTER, BLITZ)
}

DEFAULT_PROFILE_NAME = "anchor"

# Back-compat aliases for the two retired profile names.
_ALIASES = {"conservative": "anchor", "aggressive": "blitz"}


def get_profile(name: str | None = None) -> AgentProfile:
    """
    Resolve ONE profile for an agent process.

    Priority: explicit `name` arg → AGENT_PROFILE env var → 'anchor'.
    This is what `agent.py --profile` and the arena deployments use; four arena
    identities = four processes, each with its own AGENT_PROFILE + API key.
    """
    raw = (name or os.getenv("AGENT_PROFILE") or DEFAULT_PROFILE_NAME).strip().lower()
    raw = _ALIASES.get(raw, raw)
    if raw not in DEFAULT_PROFILES:
        valid = ", ".join(DEFAULT_PROFILES)
        raise ValueError(f"Unknown AGENT_PROFILE '{raw}'. Valid: {valid}")
    return DEFAULT_PROFILES[raw]


def load_profiles(override_path: str | Path | None = None) -> dict[str, AgentProfile]:
    """
    Return all profiles (harness runs every agent side by side), optionally
    merged with an override JSON: {profile_name: {field: value, ...}}.
    """
    profiles = {k: AgentProfile(**v.to_dict()) for k, v in DEFAULT_PROFILES.items()}
    if not override_path:
        return profiles
    path = Path(override_path)
    if not path.exists():
        return profiles
    data = json.loads(path.read_text(encoding="utf-8"))
    valid = {f for f in AgentProfile.__dataclass_fields__}
    for name, patch in (data or {}).items():
        patch = {k: v for k, v in (patch or {}).items() if k in valid}
        if name in profiles:
            base = profiles[name].to_dict()
            base.update(patch)
            profiles[name] = AgentProfile(**base)
        else:
            patch.setdefault("name", name)
            patch.setdefault("label", name)
            profiles[name] = AgentProfile(**patch)
    return profiles
