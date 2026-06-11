"""
The 4-agent roster — which arena identities run, with which profiles.

1 API key = 1 arena agent. Keys come from the environment (.env):

    AGENT_KEY_MONK=...      # oracle — forecast specialist  (Score track)
    AGENT_KEY_ANCHOR=...    # keel   — disciplined EV       (control / P&L)
    AGENT_KEY_HUNTER=...    # saw    — skew harvester       (P&L upside)
    AGENT_KEY_BLITZ=...     # surge  — event-driven         (P&L tail)

Any subset works — agents without a key are skipped (with a loud warning).
If NO per-agent key is set, we fall back to the single STAIR_API_KEY running
the AGENT_PROFILE (default anchor), so the loop is still testable with one key.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

import config
from harness.profiles import AgentProfile, get_profile

ROSTER_ORDER = ("monk", "anchor", "hunter", "blitz")


@dataclass
class LiveAgent:
    name: str
    profile: AgentProfile
    api_key: str

    def __repr__(self) -> str:  # never leak the key
        return f"LiveAgent({self.name}, key=…{self.api_key[-4:] if self.api_key else '????'})"


def load_roster(only: list[str] | None = None) -> list[LiveAgent]:
    """Build the roster from env keys. `only` filters by profile name."""
    agents: list[LiveAgent] = []
    for name in ROSTER_ORDER:
        if only and name not in only:
            continue
        key = os.getenv(f"AGENT_KEY_{name.upper()}", "").strip()
        if key:
            agents.append(LiveAgent(name=name, profile=get_profile(name), api_key=key))
        else:
            print(f"[roster] WARNING: AGENT_KEY_{name.upper()} not set — {name} will not run")

    if not agents:
        if config.ARENA_KEY:
            prof = get_profile()  # AGENT_PROFILE env or anchor
            print(f"[roster] no per-agent keys found — falling back to single agent "
                  f"'{prof.name}' on STAIR_API_KEY")
            agents.append(LiveAgent(name=prof.name, profile=prof, api_key=config.ARENA_KEY))
        else:
            raise RuntimeError(
                "No agent keys configured. Set AGENT_KEY_MONK/ANCHOR/HUNTER/BLITZ "
                "(or at least STAIR_API_KEY) in .env")
    return agents
