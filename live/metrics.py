"""
Append-only metrics log for retrospective evaluation.

Every interesting event becomes one JSON line in storage/live/events.jsonl:
forecasts (with grounding + market mids), per-agent decisions (picks AND skip
reasons), order outcomes, settlements, and errors. `python -m live report`
reads this back to compute P&L, Brier-vs-market, fire rates, etc. — the
M-1…M-4 measurements from docs/STRATEGY.md §6.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

EVENTS_PATH = Path(__file__).resolve().parent.parent / "storage" / "live" / "events.jsonl"


def log_event(event_type: str, **fields) -> None:
    """Append one event line. Never raises — metrics must not kill the loop."""
    try:
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "type": event_type, **fields}
        with EVENTS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception as exc:
        print(f"[metrics] WARNING: failed to log {event_type}: {exc!r}")


def read_events(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path else EVENTS_PATH
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
