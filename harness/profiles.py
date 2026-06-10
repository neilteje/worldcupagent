"""
Agent aggressiveness profiles.

A profile is the *trading policy* layered on top of a shared prediction. The
prediction (probabilities + confidence) is identical across agents; profiles only
change risk appetite — how often to bet, how much, and under what guardrails. This
makes the head-to-head a clean A/B on aggressiveness alone.

Two profiles ship today (the user wants two agents tomorrow; the design scales to
the four planned for the live arena):

  conservative ("anchor")  — bets only on clear edges, small fractional Kelly,
                             tight caps, high confidence floor. Capital first.
  aggressive  ("blitz")    — acts on thin edges, larger fractional Kelly, looser
                             caps, low confidence floor, trades both windows.

Tune freely: every knob below is a plain field, and `load_profiles` can read an
override JSON so you can retune without touching code.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import json
from pathlib import Path


# Council confidence is a label; map to a number for floor comparisons.
_CONFIDENCE_NUM = {"low": 0.40, "medium": 0.60, "high": 0.80, "": 0.50}


def confidence_to_num(conf: str | float | None) -> float:
    if isinstance(conf, (int, float)):
        return float(conf)
    return _CONFIDENCE_NUM.get(str(conf or "").strip().lower(), 0.50)


@dataclass
class AgentProfile:
    name: str                              # machine id, e.g. "conservative"
    label: str                             # human label, e.g. "anchor"
    bankroll: float = 100.0                # starting paper bankroll (USD)

    # ── When to bet ──────────────────────────────────────────────────────
    min_edge_vs_fair: float = 0.05         # edge vs de-vigged fair price to act
    min_ev_per_dollar: float = 0.0         # require at least this EV/$ (0 = any +EV)
    min_confidence: float = 0.55           # numeric confidence floor (see map above)
    trade_prematch: bool = True
    trade_halftime: bool = True
    skip_on_high_scout_flag: bool = True   # veto if a high-severity flag hits our side

    # ── How much to bet ──────────────────────────────────────────────────
    kelly_fraction: float = 0.40           # fraction of full Kelly
    max_bet_usd: float = 5.0               # hard per-trade cap (arena rule = $5)
    stake_cap_fraction: float = 0.10       # also cap each bet at this fraction of bankroll
    max_bets_per_window: int = 1           # how many outcomes to back per window

    def to_dict(self) -> dict:
        return asdict(self)


# ── The two tuned agents for tomorrow ───────────────────────────────────────

CONSERVATIVE = AgentProfile(
    name="conservative",
    label="anchor",
    bankroll=100.0,
    min_edge_vs_fair=0.06,      # only genuine, sizable disagreement
    min_ev_per_dollar=0.05,     # demand a real EV cushion
    min_confidence=0.60,        # medium+ council confidence
    trade_prematch=True,
    trade_halftime=True,
    skip_on_high_scout_flag=True,
    kelly_fraction=0.25,        # quarter-Kelly → low variance
    max_bet_usd=3.0,            # voluntarily stakes under the $5 arena cap
    stake_cap_fraction=0.05,    # never risk >5% of bankroll on one bet
    max_bets_per_window=1,
)

AGGRESSIVE = AgentProfile(
    name="aggressive",
    label="blitz",
    bankroll=100.0,
    min_edge_vs_fair=0.025,     # acts on thin edges the anchor skips
    min_ev_per_dollar=0.0,      # any positive EV is fair game
    min_confidence=0.40,        # will fire on low-confidence reads
    trade_prematch=True,
    trade_halftime=True,
    skip_on_high_scout_flag=False,  # tolerates flagged sides for more action
    kelly_fraction=0.60,        # heavier sizing
    max_bet_usd=5.0,            # uses the full arena cap
    stake_cap_fraction=0.15,    # up to 15% of bankroll per bet
    max_bets_per_window=2,      # may back two outcomes (e.g. underdog + draw)
)

DEFAULT_PROFILES: dict[str, AgentProfile] = {
    CONSERVATIVE.name: CONSERVATIVE,
    AGGRESSIVE.name: AGGRESSIVE,
}


def load_profiles(override_path: str | Path | None = None) -> dict[str, AgentProfile]:
    """
    Return the agent profiles, optionally merged with an override JSON.

    The override file is a map of {profile_name: {field: value, ...}} and only
    needs to contain the fields you want to change. Unknown profile names create
    new agents (must include at least `name` and `label`).
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
