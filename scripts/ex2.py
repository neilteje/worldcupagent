"""Quick exploration script for World Cup fixtures.

This script pulls the fixtures scheduled for *today* (UTC) using the
``data.sportmonks`` helper functions and displays a concise table in the
terminal using ``rich``.  It demonstrates how to retrieve the basic fixture
information together with the ML predictions and bookmaker odds (if
available).

Run the script directly::

	python -m scripts.exploration

The output is a ``rich.Table`` with the following columns:

* **Fixture** – ``Home Team vs Away Team``
* **Kickoff (UTC)** – ISO timestamp truncated to ``HH:MM``
* **ML Home**, **ML Draw**, **ML Away** – probabilities from the Sportmonks
  ML model (values between 0 and 1).
* **Book Home**, **Book Draw**, **Book Away** – implied probabilities derived
  from the best bookmaker odds.

If a particular piece of data is missing the cell is left blank.
"""

from __future__ import annotations

import datetime as _dt
import sys
from datetime import timezone
from pathlib import Path
from typing import List, Mapping, Optional

# Ensure the parent directory (worldcupagent/) is on sys.path so that
# top-level packages like ``data`` are importable when this script is run
# directly from the ``scripts/`` subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from data import sportmonks

console = Console()


def _is_today(iso_ts: Optional[str]) -> bool:
	"""Return ``True`` if the ISO timestamp falls on the current UTC date.

	The ``starting_at`` field from Sportmonks is an ISO‑8601 string (e.g.
	``2026-06-10T14:00:00Z``).  We parse it and compare the ``date`` part to
	``datetime.utcnow().date()``.
	"""
	if not iso_ts:
		return False
	try:
		dt = _dt.datetime.fromisoformat(iso_ts.rstrip("Z"))
		# Ensure the datetime is timezone‑aware in UTC
		if dt.tzinfo is None:
			dt = dt.replace(tzinfo=timezone.utc)
		return dt.date() == _dt.datetime.now(timezone.utc).date()
	except Exception:
		return False


def _team_name(fixture: Mapping, side: str) -> str:
	"""Extract a readable team name from a fixture dict.

	``side`` should be ``"home"`` or ``"away"``.  The Sportmonks payload can
	contain the team under ``home_team`` / ``away_team`` with a ``name`` or
	``country`` field.  We fall back to the raw dict representation if the
	expected keys are missing.
	"""
	key = f"{side}_team"
	team = fixture.get(key) or {}
	return (
		team.get("name")
		or team.get("country")
		or team.get("short_code")
		or str(team)
	)


def _format_prob(p: Optional[float]) -> str:
	"""Format a probability (0‑1) as a percentage with one decimal place."""
	return f"{p * 100:.1f}%" if p is not None else ""


def main() -> None:
	# Pull all fixtures for the configured season and keep only those that
	# start today.
	fixtures: List[Mapping] = sportmonks.get_fixtures_by_season()
	todays = [f for f in fixtures if _is_today(f.get("starting_at"))]

	if not todays:
		console.print("[bold yellow]No fixtures scheduled for today.[/]")
		return

	table = Table(title="World Cup 2026 – Fixtures for Today (UTC)")
	table.add_column("Fixture", style="cyan", no_wrap=True)
	table.add_column("Kickoff (UTC)", style="magenta")
	table.add_column("ML Home", justify="right")
	table.add_column("ML Draw", justify="right")
	table.add_column("ML Away", justify="right")
	table.add_column("Book Home", justify="right")
	table.add_column("Book Draw", justify="right")
	table.add_column("Book Away", justify="right")

	for f in todays:
		# Basic fixture info
		home = _team_name(f, "home")
		away = _team_name(f, "away")
		kickoff_iso = f.get("starting_at") or ""
		try:
			kickoff_dt = _dt.datetime.fromisoformat(kickoff_iso.rstrip("Z"))
			kickoff_str = kickoff_dt.strftime("%H:%M")
		except Exception:
			kickoff_str = kickoff_iso

		# Detailed data – we request the full fixture record which includes
		# predictions and odds.  ``get_fixture_detail_safe`` falls back to a
		# minimal stub on failure, so the script never crashes.
		detail = sportmonks.get_fixture_detail_safe(f.get("id"))
		ml = sportmonks.extract_ml_probabilities(detail) or {}
		book = sportmonks.extract_bookmaker_odds(detail) or {}

		table.add_row(
			f"{home} vs {away}",
			kickoff_str,
			_format_prob(ml.get("home_win")),
			_format_prob(ml.get("draw")),
			_format_prob(ml.get("away_win")),
			_format_prob(book.get("home_win")),
			_format_prob(book.get("draw")),
			_format_prob(book.get("away_win")),
		)

	console.print(table)


if __name__ == "__main__":
	main()
