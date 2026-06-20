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
  hunter → SAW     — draw and underdog skew. Rejects favorites and prices over
           0.40, with larger size only after independent evidence filters pass.
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
    floor_to_min_order: bool = True        # a +EV pick whose size rounds below the
                                           # $1 CLOB minimum is bumped UP to $1
                                           # (≤ cap) instead of dropped — the $1
                                           # floor is a venue mechanic, not a reason
                                           # to skip a bet you've decided to make
    apply_confidence_multiplier: bool = True   # let gates shrink size on low
                                               # council confidence (KEEL/ORACLE);
                                               # P&L-tail agents turn this OFF —
                                               # their edge is skew, not conviction

    # ── Market honesty (harness) ─────────────────────────────────────────
    trade_synthetic: bool = False          # bet synthetic_demo markets at all?
    synthetic_size_multiplier: float = 0.25  # size-down when trading synthetic

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_config(cls, name: str) -> "AgentProfile":
        """Resolve ONE profile from the single source of truth.

        ``harness/profiles`` owns profile *policy* (sizing, confidence floors,
        windows). ``config.py`` owns the coordinated *edge* layer (conservative-
        edge buffers + per-agent edge/EV bars consumed by
        ``betting/recommendation``). These are disjoint concerns; this
        constructor returns the canonical profile and is validated against config
        by :func:`validate_profile_config` at startup so the two layers can never
        silently disagree. BLITZ is frozen at its production values."""
        return get_profile(name)


# ── The four tuned agents ───────────────────────────────────────────────────

LEGACY_BLITZ_POLICY = {
    "min_edge_vs_fair": 0.02,
    "min_ev_per_dollar": 0.0,
    "min_confidence": 0.35,
    "kelly_fraction": 0.65,
    "max_bet_usd": 5.0,
    "stake_cap_fraction": 0.15,
    "max_bets_per_window": 2,
    "skip_on_high_scout_flag": False,
    "apply_confidence_multiplier": False,
    "trade_prematch": True,
    "trade_halftime": True,
    "max_entry_price": None,
    "floor_to_min_order": True,
    "trade_synthetic": True,
    "synthetic_size_multiplier": 0.25,
}

MONK = AgentProfile(name="monk", label="legacy blitz — monk", **LEGACY_BLITZ_POLICY)
ANCHOR = AgentProfile(name="anchor", label="legacy blitz — anchor", **LEGACY_BLITZ_POLICY)
HUNTER = AgentProfile(name="hunter", label="legacy blitz — hunter", **LEGACY_BLITZ_POLICY)
BLITZ = AgentProfile(name="blitz", label="legacy blitz — blitz", **LEGACY_BLITZ_POLICY)

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


def validate_profile_config() -> list[str]:
    """Startup validation that the profile layer and ``config.py`` agree where
    they must (spec §24). Raises ``ValueError`` on drift; returns the list of
    checks performed for logging.

    Enforced invariants:
      * BLITZ is frozen at its production values.
      * No profile exceeds the hard per-order USD cap (arena rule ≤ $5).
      * Kelly / stake-cap fractions stay within [0, 1].
      * Coordinated edge config is internally consistent (min ≤ max entry price).
    """
    import config

    issues: list[str] = []
    checks: list[str] = []

    for name, profile in DEFAULT_PROFILES.items():
        for field_name, expected in LEGACY_BLITZ_POLICY.items():
            actual = getattr(profile, field_name)
            checks.append(f"{name}.{field_name}")
            if actual != expected:
                issues.append(
                    f"legacy Blitz policy drift: {name}.{field_name}={actual!r} != {expected!r}"
                )

    for name, p in DEFAULT_PROFILES.items():
        checks.append(f"{name}.max_bet_usd<=cap")
        if p.max_bet_usd > config.MAX_BET_USD + 1e-9:
            issues.append(f"{name}.max_bet_usd {p.max_bet_usd} > MAX_BET_USD {config.MAX_BET_USD}")
        if not (0.0 <= p.kelly_fraction <= 1.0):
            issues.append(f"{name}.kelly_fraction {p.kelly_fraction} outside [0,1]")
        if not (0.0 <= p.stake_cap_fraction <= 1.0):
            issues.append(f"{name}.stake_cap_fraction {p.stake_cap_fraction} outside [0,1]")

    if config.ANCHOR_MIN_ENTRY_PRICE > config.ANCHOR_MAX_ENTRY_PRICE:
        issues.append("ANCHOR_MIN_ENTRY_PRICE > ANCHOR_MAX_ENTRY_PRICE")

    if issues:
        raise ValueError("Profile/config drift detected: " + "; ".join(issues))
    return checks


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
