"""Rolling form (spec §15).

Exponentially time-weighted goals/xG for and against, over a short and a long
half-life. Only matches strictly BEFORE the forecast time contribute, so form is
leakage-free by construction. xG and realized goals are kept separate; when a
data source has no xG the caller passes a proxy and flags coverage.
"""
from __future__ import annotations

import math
from datetime import datetime


def _half_life_weight(days_ago: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (max(0.0, days_ago) / half_life_days)


class RollingFormBuilder:
    """Accumulates per-team match rows; queries weighted form as of a date."""

    def __init__(self, short_half_life_days: float = 75.0, long_half_life_days: float = 240.0):
        self.short_half_life = short_half_life_days
        self.long_half_life = long_half_life_days
        # team_id -> list of (date, gf, ga, xgf, xga)
        self._matches: dict[str, list[tuple]] = {}

    def add_match(self, team_id: str, date: datetime, goals_for: float, goals_against: float,
                  xg_for: float | None = None, xg_against: float | None = None) -> None:
        self._matches.setdefault(team_id, []).append(
            (date, float(goals_for), float(goals_against),
             float(xg_for) if xg_for is not None else None,
             float(xg_against) if xg_against is not None else None)
        )

    def _weighted(self, rows: list[tuple], as_of: datetime, half_life: float) -> dict:
        gf = ga = xgf = xga = wsum = 0.0
        n = 0
        for (date, g_for, g_ag, x_for, x_ag) in rows:
            if date >= as_of:
                continue  # strict leakage guard
            days_ago = (as_of - date).total_seconds() / 86400.0
            w = _half_life_weight(days_ago, half_life)
            gf += w * g_for
            ga += w * g_ag
            xgf += w * (x_for if x_for is not None else g_for)
            xga += w * (x_ag if x_ag is not None else g_ag)
            wsum += w
            n += 1
        if wsum <= 0:
            return {"goals_for": None, "goals_against": None,
                    "xg_for": None, "xg_against": None, "matches": 0}
        return {
            "goals_for": gf / wsum, "goals_against": ga / wsum,
            "xg_for": xgf / wsum, "xg_against": xga / wsum, "matches": n,
        }

    def form(self, team_id: str, as_of: datetime) -> dict:
        rows = self._matches.get(team_id, [])
        short = self._weighted(rows, as_of, self.short_half_life)
        long = self._weighted(rows, as_of, self.long_half_life)
        return {"short": short, "long": long}
