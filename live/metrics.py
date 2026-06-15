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
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import hashlib

from models.forecast_layers import FORECAST_PIPELINE_VERSION
from harness.profiles import DEFAULT_PROFILES

EVENTS_PATH = Path(__file__).resolve().parent.parent / "storage" / "live" / "events.jsonl"
REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_VERSION = "four_agent_p0_p1_v1"
MODEL_VERSION = "deterministic_v2_market_blind_v1"


def _commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={REPO_ROOT}", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:
        return "unknown"


def _profile_hash() -> str:
    payload = json.dumps(
        {k: v.to_dict() for k, v in DEFAULT_PROFILES.items()},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def runtime_metadata() -> dict:
    return {
        "commit_sha": _commit_sha(),
        "strategy_version": STRATEGY_VERSION,
        "forecast_pipeline_version": FORECAST_PIPELINE_VERSION,
        "model_version": MODEL_VERSION,
        "profile_configuration_hash": _profile_hash(),
        "enabled_feature_flags": {
            "typed_evidence": True,
            "single_market_calibration": True,
            "joint_allocation": True,
            "full_halftime_1x2": True,
        },
        "active_data_sources": [
            "sportmonks", "supabase", "bzzoiro", "web", "reddit", "grok",
            "polymarket", "odds2prob", "live_state",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def log_event(event_type: str, **fields) -> None:
    """Append one event line. Never raises — metrics must not kill the loop."""
    try:
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "type": event_type, **runtime_metadata(), **fields}
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
