"""
The fixtures the harness trades, plus their window trigger times.

Tomorrow's four friendlies are the defaults. Each match yields two windows:
  PRE_MATCH — fired at (kickoff - prematch_lead_min), default 5 min before.
  HT        — fired at (kickoff + ht_offset_min), default 50 min after kickoff,
              which is roughly when a real half-time whistle + stoppage lands.

Times are interpreted in the local zone (America/Chicago / CDT) the user gave them
in. Everything is overridable via a fixtures JSON so you can reuse this for any
match day without editing code.

Polymarket slugs for friendlies are usually unknown; we leave `pm_slug` empty and
let the broker fall back to a labeled synthetic reference. If you find a real slug
(e.g. `fif-por-nga-2026-06-10`), drop it into the override JSON and the broker
will trade against live mids instead.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("America/Chicago")
DEFAULT_DATE = "2026-06-10"          # "tomorrow" relative to the request
PREMATCH_LEAD_MIN = 5
HT_OFFSET_MIN = 50


@dataclass
class Fixture:
    fixture_code: str
    home: str
    away: str
    home_code: str
    away_code: str
    kickoff_local: str               # "HH:MM" 24h local time
    date: str = DEFAULT_DATE         # "YYYY-MM-DD"
    pm_slug: str = ""                # Polymarket event slug, if known
    prematch_lead_min: int = PREMATCH_LEAD_MIN
    ht_offset_min: int = HT_OFFSET_MIN

    def kickoff_dt(self) -> datetime:
        return datetime.fromisoformat(f"{self.date}T{self.kickoff_local}:00").replace(tzinfo=LOCAL_TZ)

    def window_trigger(self, window: str) -> datetime:
        ko = self.kickoff_dt()
        if window.upper() in {"HT", "HALFTIME"}:
            return ko + timedelta(minutes=self.ht_offset_min)
        return ko - timedelta(minutes=self.prematch_lead_min)

    def guess_slug(self) -> str:
        """Best-effort Polymarket friendly slug guess (fif-<home>-<away>-<date>)."""
        return self.pm_slug or f"fif-{self.home_code.lower()}-{self.away_code.lower()}-{self.date}"

    def to_dict(self) -> dict:
        return {
            "fixture_code": self.fixture_code, "home": self.home, "away": self.away,
            "home_code": self.home_code, "away_code": self.away_code,
            "kickoff_local": self.kickoff_local, "date": self.date, "pm_slug": self.pm_slug,
        }


# ── Tomorrow's four friendlies (local CDT times the user provided) ───────────

DEFAULT_FIXTURES: list[Fixture] = [
    Fixture("FRD-PAK-AFG", "Pakistan",  "Afghanistan", "PAK", "AFG", "11:00"),
    Fixture("FRD-POR-NGA", "Portugal",  "Nigeria",     "POR", "NGA", "14:45"),
    Fixture("FRD-ENG-CRC", "England",   "Costa Rica",  "ENG", "CRC", "15:00"),
    Fixture("FRD-BOL-DZA", "Bolivia",   "Algeria",     "BOL", "DZA", "19:00"),
]


def load_fixtures(override_path: str | Path | None = None) -> list[Fixture]:
    """
    Return the fixtures, optionally replaced/extended by an override JSON.

    Override format: a JSON list of objects with the Fixture fields. If present it
    fully replaces the defaults (so you can supply a different match day).
    """
    if not override_path:
        return list(DEFAULT_FIXTURES)
    path = Path(override_path)
    if not path.exists():
        return list(DEFAULT_FIXTURES)
    rows = json.loads(path.read_text(encoding="utf-8"))
    valid = {f for f in Fixture.__dataclass_fields__}
    out: list[Fixture] = []
    for row in rows:
        out.append(Fixture(**{k: v for k, v in row.items() if k in valid}))
    return out or list(DEFAULT_FIXTURES)


def all_windows(fixtures: list[Fixture]) -> list[tuple[Fixture, str]]:
    """Flatten fixtures into (fixture, window) pairs sorted by trigger time."""
    pairs: list[tuple[Fixture, str]] = []
    for f in fixtures:
        if True:
            pairs.append((f, "PRE_MATCH"))
            pairs.append((f, "HT"))
    pairs.sort(key=lambda p: p[0].window_trigger(p[1]))
    return pairs
