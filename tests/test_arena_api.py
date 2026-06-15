"""
Arena API GET endpoint tester.
Tests every known read endpoint and prints structured results.

Usage:
    python test_arena_api.py
"""
from __future__ import annotations
import json
import os
import sys
import textwrap
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

ARENA_KEY = os.getenv("ARENA_KEY") or os.getenv("STAIR_API_KEY", "")
ARENA_BASE = os.getenv("ARENA_BASE", "https://stair-ai.com")
ARENA_API = f"{ARENA_BASE}/api"
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ezvbmtvrvzageqixvdak.supabase.co/rest/v1")
SUPABASE_KEY = (
    os.getenv("SUPABASE_PUBLISHABLE_KEY")
    or os.getenv("SUPABASE_KEY", "sb_publishable__m8bOkD05ToFwATpaWST5w_2-3fGS7V")
)

ARENA_HEADERS = {"x-api-key": ARENA_KEY, "Content-Type": "application/json"}

# WC2026 season id from GUIDE.md
WC2026_SEASON_ID = 26618


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _print_section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _print_result(label: str, status: int, ok: bool, body: Any, note: str = "") -> None:
    tag = "[OK  ]" if ok else "[FAIL]"
    print(f"\n{tag} {label}")
    print(f"       HTTP {status}  {note}")
    if isinstance(body, (dict, list)):
        snippet = json.dumps(body, indent=2, default=str)
        lines = snippet.splitlines()
        cutoff = 30
        if len(lines) > cutoff:
            snippet = "\n".join(lines[:cutoff]) + f"\n       ... [{len(lines) - cutoff} more lines]"
        for line in snippet.splitlines():
            print(f"       {line}")
    elif body:
        for line in textwrap.wrap(str(body)[:800], 76):
            print(f"       {line}")


def get(url: str, headers: dict, params: dict | None = None, label: str = "", note: str = "") -> dict:
    label = label or url
    try:
        r = httpx.get(url, headers=headers, params=params or {}, timeout=20)
        try:
            body = r.json()
        except Exception:
            body = r.text[:500]
        _print_result(label, r.status_code, r.is_success, body, note)
        return {"ok": r.is_success, "status": r.status_code, "body": body}
    except Exception as exc:
        _print_result(label, 0, False, str(exc), "connection error")
        return {"ok": False, "status": 0, "body": str(exc)}


# --------------------------------------------------------------------------
# test groups
# --------------------------------------------------------------------------

def test_agent_identity() -> None:
    _print_section("AGENT IDENTITY -- GET /v1/arena/agents/me")
    get(
        f"{ARENA_API}/v1/arena/agents/me",
        ARENA_HEADERS,
        label="agents/me -- agent_id, display_name, wallet balance, lifecycle_phase",
        note="Core identity check: confirms key is valid and wallet is funded",
    )


def test_exposure() -> None:
    _print_section("EXPOSURE -- GET /v1/arena/exposure")
    get(
        f"{ARENA_API}/v1/arena/exposure",
        ARENA_HEADERS,
        label="exposure -- open positions and locked USD per fixture",
        note="Shows what positions are currently open and how much USDC is locked",
    )


def test_order_status() -> None:
    _print_section("ORDER STATUS -- GET /v1/arena/orders/{id}")
    # Probe with a valid UUID format to see what response shape we get
    import uuid
    probe_id = str(uuid.uuid4())
    get(
        f"{ARENA_API}/v1/arena/orders/{probe_id}",
        ARENA_HEADERS,
        label=f"orders/{probe_id} -- probe with random UUID",
        note="Checks if endpoint is live; GUIDE said 'not yet ready' -- seeing current state",
    )


def test_fixture_mapping() -> None:
    _print_section("MAPPING -- GET /v1/web/mapping")
    get(
        f"{ARENA_API}/v1/web/mapping",
        ARENA_HEADERS,
        params={"fixture_id": 19157983},
        label="web/mapping?fixture_id=19157983 -- fixture_id to polymarket_event_slug",
        note="Maps a Sportmonks fixture_id to its Polymarket event slug",
    )
    get(
        f"{ARENA_API}/v1/web/mapping",
        ARENA_HEADERS,
        label="web/mapping (no params) -- does it return all mappings?",
        note="Probing whether a parameterless call returns the full mapping list",
    )


def test_sportmonks_proxy() -> None:
    _print_section("SPORTMONKS PROXY -- /v1/data/proxy/sportmonks/v3/football/")
    base = f"{ARENA_API}/v1/data/proxy/sportmonks/v3/football"

    get(
        f"{base}/fixtures",
        ARENA_HEADERS,
        params={"filters": f"seasonId:{WC2026_SEASON_ID}", "per_page": 5},
        label=f"fixtures?seasonId={WC2026_SEASON_ID}&per_page=5 -- WC2026 fixture list",
        note="First 5 fixtures in the WC2026 season",
    )
    get(
        f"{base}/seasons/{WC2026_SEASON_ID}",
        ARENA_HEADERS,
        label=f"seasons/{WC2026_SEASON_ID} -- WC2026 season metadata",
        note="Season name, dates, league info",
    )
    get(
        f"{base}/livescores/inplay",
        ARENA_HEADERS,
        params={"per_page": 3},
        label="livescores/inplay -- matches currently live",
        note="Matches in progress right now (empty outside match windows)",
    )


def test_polymarket_proxy() -> None:
    _print_section("POLYMARKET PROXY -- Gamma + CLOB")
    gamma_base = f"{ARENA_API}/v1/data/proxy/polymarket-gamma"
    clob_base = f"{ARENA_API}/v1/data/proxy/polymarket-clob"

    slug = "will-mexico-win-vs-south-africa-2026-06-11"
    get(
        f"{gamma_base}/events",
        ARENA_HEADERS,
        params={"slug": slug},
        label=f"gamma/events?slug={slug}",
        note="Polymarket event + 3 nested winner markets (conditionIds + tokenIds)",
    )
    get(
        f"{clob_base}/midpoint",
        ARENA_HEADERS,
        params={"token_id": "probe-invalid"},
        label="clob/midpoint?token_id=probe-invalid -- error shape probe",
        note="Checking endpoint availability; dummy token will return an error",
    )


def test_supabase() -> None:
    _print_section("SUPABASE -- direct REST (no arena proxy needed)")
    sb_headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }
    get(
        f"{SUPABASE_URL}/catalog_tables",
        {**sb_headers, "Accept-Profile": "world_cup_arena"},
        params={"select": "table_name,row_count,description", "limit": "20"},
        label="catalog_tables -- all tables in the world_cup_arena schema",
        note="Data dictionary: table names, row counts, descriptions",
    )
    get(
        f"{SUPABASE_URL}/ads_a_team_tournament_performance",
        {**sb_headers, "Accept-Profile": "world_cup_arena"},
        params={"limit": "3"},
        label="ads_a_team_tournament_performance -- StatsBomb historical priors sample",
        note="Country-level historical tournament performance from StatsBomb",
    )


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------

def main() -> None:
    if not ARENA_KEY:
        print("ERROR: ARENA_KEY not set. Add it to .env or set ARENA_KEY env var.")
        sys.exit(1)

    print("\nArena API tester")
    print(f"Base : {ARENA_BASE}")
    print(f"Key  : {ARENA_KEY[:8]}...{ARENA_KEY[-4:]}")

    test_agent_identity()
    test_exposure()
    test_order_status()
    test_fixture_mapping()
    test_sportmonks_proxy()
    test_polymarket_proxy()
    test_supabase()

    _print_section("DONE")
    print("Check [OK  ] / [FAIL] above for each endpoint status.\n")


if __name__ == "__main__":
    main()
