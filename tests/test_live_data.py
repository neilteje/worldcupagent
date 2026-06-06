"""
Integration tests for live Arena API + Supabase data connections.

These tests require a valid ARENA_KEY in the environment and make real HTTP
calls.  They are marked as integration tests and skipped automatically when
the key is absent.

Run with:
    pytest tests/test_live_data.py -v
"""
from __future__ import annotations
import os
import pytest
from dotenv import load_dotenv

load_dotenv(override=True)

ARENA_KEY = os.getenv("ARENA_KEY") or os.getenv("STAIR_API_KEY", "")
needs_key = pytest.mark.skipif(not ARENA_KEY, reason="ARENA_KEY not set")


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def settings():
    from agent.config import load_settings
    return load_settings()


@pytest.fixture(scope="module")
def first_mapping(settings):
    """Return the first arena mapping entry (Mexico vs South Africa)."""
    from data.polymarket import get_all_mappings
    mappings = get_all_mappings()
    assert mappings, "Mapping endpoint returned no results"
    return mappings[0]


# --------------------------------------------------------------------------
# Arena API: agent identity & wallet
# --------------------------------------------------------------------------

@needs_key
def test_agent_identity(settings):
    import httpx
    r = httpx.get(
        f"{settings.arena_api}/v1/arena/agents/me",
        headers=settings.headers,
        timeout=15,
    )
    assert r.status_code == 200, f"agents/me returned {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert body.get("agent_id"), "agent_id missing from agents/me response"
    assert body.get("lifecycle_phase") == "active", f"Agent not active: {body.get('lifecycle_phase')}"
    wallet = body.get("wallet") or {}
    balance = float(wallet.get("available_balance_usdc") or 0)
    assert balance >= 0, "Negative wallet balance"
    print(f"\n  agent: {body['display_name']}  |  wallet: ${balance:.2f} USDC")


@needs_key
def test_exposure(settings):
    import httpx
    r = httpx.get(
        f"{settings.arena_api}/v1/arena/exposure",
        headers=settings.headers,
        timeout=15,
    )
    assert r.status_code == 200, f"exposure returned {r.status_code}"
    body = r.json()
    assert "positions" in body, "positions key missing from exposure response"
    print(f"\n  open positions: {len(body['positions'])}")


# --------------------------------------------------------------------------
# Arena API: mapping (fixture discovery)
# --------------------------------------------------------------------------

@needs_key
def test_all_mappings_count(settings):
    from data.polymarket import get_all_mappings
    mappings = get_all_mappings()
    assert len(mappings) >= 48, f"Expected >=48 WC2026 group-stage fixtures, got {len(mappings)}"
    print(f"\n  total mapped fixtures: {len(mappings)}")


@needs_key
def test_mapping_has_token_ids(first_mapping):
    for field in ("polymarket_home_token_yes", "polymarket_draw_token_yes", "polymarket_away_token_yes"):
        assert first_mapping.get(field), f"{field} missing from mapping entry"
    assert first_mapping.get("sportmonks_fixture_id"), "sportmonks_fixture_id missing"
    assert first_mapping.get("polymarket_event_slug"), "polymarket_event_slug missing"
    print(f"\n  fixture: {first_mapping.get('sportmonks_match_name')}  |  slug: {first_mapping.get('polymarket_event_slug')}")


@needs_key
def test_fixture_discovery(settings):
    from data.sportmonks import discover_fixtures_safe
    fixtures = discover_fixtures_safe(limit=5)
    assert fixtures, "discover_fixtures_safe returned empty list"
    f = fixtures[0]
    assert f.get("id") != "DEMO-FIXTURE", "discover_fixtures_safe returned demo fixture (key missing?)"
    assert f.get("home_country"), "home_country missing from discovered fixture"
    assert f.get("polymarket_home_token_yes"), "polymarket token missing from discovered fixture"
    print(f"\n  first fixture: {f['name']} | kickoff: {f.get('starting_at')}")


# --------------------------------------------------------------------------
# Polymarket: CLOB pricing from mapping tokens
# --------------------------------------------------------------------------

@needs_key
def test_market_probs_via_tokens(first_mapping):
    from data.polymarket import get_three_way_from_tokens
    result = get_three_way_from_tokens(
        home_token=first_mapping["polymarket_home_token_yes"],
        draw_token=first_mapping["polymarket_draw_token_yes"],
        away_token=first_mapping["polymarket_away_token_yes"],
        slug=first_mapping["polymarket_event_slug"],
    )
    assert result.get("complete"), f"Market fetch failed: {result.get('reason')}"
    probs = result["normalized_probs"]
    assert probs, "normalized_probs is None"
    assert abs(sum(probs.values()) - 1.0) < 0.02, f"Probs don't sum to 1: {probs}"
    assert all(0 < v < 1 for v in probs.values()), f"Out-of-range probability: {probs}"
    print(f"\n  {first_mapping['sportmonks_match_name']}: home={probs['home']:.3f}  draw={probs['draw']:.3f}  away={probs['away']:.3f}")


@needs_key
def test_get_three_way_market_probs_with_tokens(settings, first_mapping):
    from data.polymarket import get_three_way_market_probs
    tokens = {
        "home": first_mapping["polymarket_home_token_yes"],
        "draw": first_mapping["polymarket_draw_token_yes"],
        "away": first_mapping["polymarket_away_token_yes"],
        "slug": first_mapping["polymarket_event_slug"],
    }
    result = get_three_way_market_probs(
        fixture_id=first_mapping["sportmonks_fixture_id"],
        tokens=tokens,
    )
    assert result["complete"], f"Fast-path market fetch failed: {result.get('reason')}"


@needs_key
def test_get_three_way_market_probs_fixture_id_fallback(settings, first_mapping):
    """Slow path: pass fixture_id only (triggers mapping lookup + CLOB)."""
    from data.polymarket import get_three_way_market_probs
    result = get_three_way_market_probs(
        fixture_id=first_mapping["sportmonks_fixture_id"]
    )
    # Mapping lookup + CLOB should work (even without tokens pre-loaded)
    assert result.get("complete"), f"Slow-path market fetch failed: {result.get('reason')}"


# --------------------------------------------------------------------------
# Sportmonks: fixture detail via proxy
# --------------------------------------------------------------------------

@needs_key
def test_sportmonks_fixture_detail(first_mapping):
    from data.sportmonks import get_fixture_detail_safe
    fixture_id = int(first_mapping["sportmonks_fixture_id"])
    detail = get_fixture_detail_safe(fixture_id)
    assert isinstance(detail, dict), "get_fixture_detail_safe returned non-dict"
    assert not detail.get("demo"), "Fixture fell back to demo stub -- Sportmonks call failed"
    participants = detail.get("participants") or []
    assert len(participants) == 2, f"Expected 2 participants, got {len(participants)}"
    print(f"\n  participants: {[p.get('name') for p in participants]}")
    odds_count = len(detail.get("odds") or [])
    print(f"  odds entries: {odds_count}")


@needs_key
def test_extract_bookmaker_probs(first_mapping):
    from data.sportmonks import get_fixture_detail_safe, extract_bookmaker_probs
    fixture_id = int(first_mapping["sportmonks_fixture_id"])
    detail = get_fixture_detail_safe(fixture_id)
    probs = extract_bookmaker_probs(detail)
    assert probs is not None, "extract_bookmaker_probs returned None (no odds in fixture?)"
    assert set(probs.keys()) == {"home", "draw", "away"}, f"Unexpected keys: {set(probs.keys())}"
    assert abs(sum(probs.values()) - 1.0) < 0.02, f"Bookmaker probs don't sum to 1: {probs}"
    print(f"\n  bookmaker: home={probs['home']:.3f}  draw={probs['draw']:.3f}  away={probs['away']:.3f}")


# --------------------------------------------------------------------------
# Supabase: priors from historical data
# --------------------------------------------------------------------------

@needs_key
def test_supabase_catalog(settings):
    from data.supabase_data import get_catalog
    catalog = get_catalog(settings)
    assert catalog, "catalog_full returned empty list"
    table_names = [r["table_name"] for r in catalog]
    for expected in ("ads_a_h2h_country", "ads_a_stage_record", "d_checkpoint_snapshot", "d_match_scores"):
        assert expected in table_names, f"Expected table '{expected}' not in catalog"
    print(f"\n  catalog tables: {len(catalog)}")


@needs_key
def test_supabase_h2h_priors(settings, first_mapping):
    from data.supabase_data import get_priors
    home = first_mapping.get("home_country", "Mexico")
    away = first_mapping.get("away_country", "South Africa")
    priors = get_priors(settings, "test", home_name=home, away_name=away)
    # H2H may not exist for every pair -- None is acceptable, but if returned it must be valid
    if priors is not None:
        assert set(priors.keys()) >= {"home", "draw", "away"}, f"Missing keys in priors: {priors}"
        assert abs(sum([priors["home"], priors["draw"], priors["away"]]) - 1.0) < 0.02, \
            f"Priors don't sum to 1: {priors}"
        print(f"\n  {home} vs {away} priors: home={priors['home']:.3f}  draw={priors['draw']:.3f}  away={priors['away']:.3f}")
        print(f"  source={priors.get('source')}  h2h_matches={priors.get('h2h_matches')}")
    else:
        print(f"\n  No h2h data for {home} vs {away} -- None returned (expected for some pairs)")


@needs_key
def test_supabase_argentina_h2h(settings):
    """Argentina is a well-represented country -- should have h2h data against many opponents."""
    from data.supabase_data import get_priors
    priors = get_priors(settings, "test", home_name="Argentina", away_name="Brazil")
    assert priors is not None, "Argentina vs Brazil h2h should exist in StatsBomb data"
    assert priors["home"] > 0 and priors["away"] > 0
    print(f"\n  Argentina vs Brazil: home={priors['home']:.3f}  draw={priors['draw']:.3f}  away={priors['away']:.3f}")
    print(f"  h2h_matches={priors.get('h2h_matches')}")


@needs_key
def test_supabase_live_checkpoint_no_data(settings):
    """Pre-tournament: d_match_scores should have no HT entry for a WC2026 fixture."""
    from data.supabase_data import get_live_checkpoint
    result = get_live_checkpoint(settings, "19609127")  # Mexico vs South Africa
    # Before kickoff: should be None (no checkpoint data yet)
    assert result is None or isinstance(result, dict), "get_live_checkpoint returned unexpected type"
    if result is None:
        print("\n  No HT checkpoint yet (expected pre-tournament)")
    else:
        print(f"\n  HT checkpoint found: {result}")


# --------------------------------------------------------------------------
# End-to-end: full run_cycle with real fixture
# --------------------------------------------------------------------------

@needs_key
def test_run_cycle_real_fixture(settings, first_mapping):
    """Full run_cycle with real Mexico vs South Africa fixture (dry_run=True)."""
    from agent.run_cycle import run_cycle
    from agent.config import Settings

    s = Settings(dry_run=True)
    fixture = {
        "id": int(first_mapping["sportmonks_fixture_id"]),
        "fixture_code": first_mapping["sportmonks_fixture_id"],
        "name": first_mapping["sportmonks_match_name"],
        "home_team_code": first_mapping["home_short_code"],
        "away_team_code": first_mapping["away_short_code"],
        "home_country": first_mapping["home_country"],
        "away_country": first_mapping["away_country"],
        "polymarket_home_token_yes": first_mapping["polymarket_home_token_yes"],
        "polymarket_draw_token_yes": first_mapping["polymarket_draw_token_yes"],
        "polymarket_away_token_yes": first_mapping["polymarket_away_token_yes"],
        "polymarket_event_slug": first_mapping["polymarket_event_slug"],
    }
    decision = run_cycle(fixture, "PRE_MATCH", s, verbose=False)
    assert decision.get("final_probs"), "No final_probs in decision"
    assert decision.get("action") in ("BET", "SKIP"), f"Unexpected action: {decision.get('action')}"
    assert decision.get("dry_run") is True
    mkt = decision.get("market_probs")
    print(f"\n  action={decision['action']}  edge_tier={decision.get('edge_tier')}")
    print(f"  final: home={decision['final_probs']['home']:.3f}  draw={decision['final_probs']['draw']:.3f}  away={decision['final_probs']['away']:.3f}")
    print(f"  market: {mkt}")
    print(f"  blocking_flags: {decision.get('blocking_risk_flags')}")
