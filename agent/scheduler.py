from __future__ import annotations
import time
from agent.config import Settings
from agent.run_cycle import run_cycle
from data.sportmonks import discover_fixtures_safe


def run_once(settings: Settings, fixture_code: str | None = None, window: str | None = None) -> list[dict]:
    fixtures = discover_fixtures_safe()
    if fixture_code:
        fixtures = [f for f in fixtures if str(f.get("fixture_code") or f.get("code") or f.get("id")) == str(fixture_code)] or [{"id": fixture_code, "fixture_code": fixture_code, "demo": True}]
    windows = [window] if window else ["PRE_MATCH"]
    return [run_cycle(f, w or "PRE_MATCH", settings) for f in fixtures[:5] for w in windows]


def run_daemon(settings: Settings, interval_seconds: int = 300, fixture_code: str | None = None, window: str | None = None) -> None:
    while True:
        run_once(settings, fixture_code, window)
        time.sleep(interval_seconds)
