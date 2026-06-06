from __future__ import annotations
import time
from agent.config import Settings
from agent.run_cycle import run_cycle
from data.sportmonks import discover_fixtures_safe
from data.synthetic_fixtures import synthetic_fixtures


def run_once(settings: Settings, fixture_code: str | None = None, window: str | None = None, *, use_synthetic_fixtures: bool = False, verbose: bool = False, use_llm_analyst: bool = False, use_llm_claims: bool = False) -> list[dict]:
    fixtures = synthetic_fixtures() if use_synthetic_fixtures else discover_fixtures_safe()
    if fixture_code:
        fixtures = [f for f in fixtures if str(f.get("fixture_code") or f.get("code") or f.get("id")) == str(fixture_code)] or [{"id": fixture_code, "fixture_code": fixture_code, "demo": True}]
    decisions: list[dict] = []
    for f in fixtures[:10 if use_synthetic_fixtures else 5]:
        windows = [window] if window else [f.get("preferred_window") or "PRE_MATCH"]
        for w in windows:
            decisions.append(run_cycle(f, w or "PRE_MATCH", settings, verbose=verbose, use_llm_analyst=use_llm_analyst, use_llm_claims=use_llm_claims))
    return decisions


def run_daemon(settings: Settings, interval_seconds: int = 300, fixture_code: str | None = None, window: str | None = None, *, use_synthetic_fixtures: bool = False, verbose: bool = False, max_iterations: int | None = None, use_llm_analyst: bool = False, use_llm_claims: bool = False) -> list[dict]:
    all_decisions: list[dict] = []
    iteration = 0
    while True:
        iteration += 1
        all_decisions.extend(run_once(settings, fixture_code, window, use_synthetic_fixtures=use_synthetic_fixtures, verbose=verbose, use_llm_analyst=use_llm_analyst, use_llm_claims=use_llm_claims))
        if max_iterations is not None and iteration >= max_iterations:
            return all_decisions
        time.sleep(interval_seconds)
