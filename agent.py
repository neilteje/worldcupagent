"""
World Cup Arena Agent — main entry point.

Implements the full 9-step notebook flow for each fixture window:

  PRE-MATCH:
    1. Fetch schedule + pick fixture
    2. Fetch Polymarket slug mapping
    3. Fetch Sportmonks fixture detail → digest with Claude
    4. Fetch Polymarket moneyline (Gamma + CLOB) → digest with Claude
    5. Fetch Supabase priors (5 tables) → digest with Claude
    6. Predict (Claude + Gemini ensemble, market-blind)
    7. Strategy (Claude, compare prediction vs market)
    8. Place order (long only, ≤$5 per trade)
    9. Build + submit full ledger trace (14–15 records)

  HALF-TIME:
    1-4. Same as above (live prices updated)
    5. Fetch Supabase HT checkpoint
    6. HT predict (Bayesian update with live xG/score)
    7. Strategy
    8. Order
    9. Ledger

Usage:
  python agent.py --fixture-id 19609127 --window prematch
  python agent.py --fixture-id 19609127 --window halftime
  python agent.py --scan [--window prematch|halftime]
  python agent.py --test-connection
"""
from __future__ import annotations
import argparse
import json
import time
import uuid
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import httpx

import config
from data import sportmonks, supabase_client, polymarket as pm
from reasoning import llm
from reasoning.prompts import (
    sportmonks_digest_input, supabase_digest_input,
    predict_input, strategy_input, ht_predict_input,
)
from ledger.client import LedgerSession
from betting.kelly import should_bet, expected_value

console = Console()
_H = {"x-api-key": config.ARENA_KEY, "Content-Type": "application/json"}

# ── Utilities ──────────────────────────────────────────────────────────────

def get_wallet_balance() -> float:
    resp = httpx.get(f"{config.ARENA_API}/v1/arena/agents/me", headers=_H, timeout=15)
    resp.raise_for_status()
    return float(resp.json().get("wallet_balance_usd") or 0)


def place_order(
    fixture_id: int,
    team_code: str,
    usd_size: float,
    limit_price: float,
) -> dict:
    """POST /api/v1/arena/orders — usd_size must be a string."""
    payload = {
        "fixture_code":          str(fixture_id),
        "team_code":             team_code,
        "usd_size":              str(round(usd_size, 2)),
        "limit_price":           round(limit_price, 4),
        "time_in_force_seconds": config.DEFAULT_TIF_SECONDS,
        "idempotency_key":       str(uuid.uuid4()),
    }
    resp = httpx.post(
        f"{config.ARENA_API}/v1/arena/orders",
        headers=_H,
        json=payload,
        timeout=60,
    )
    if resp.status_code == 404:
        return {"status": "not_live", "payload": payload}
    resp.raise_for_status()
    return resp.json()


def get_season_fixtures() -> list[dict]:
    """Return flat list of fixture dicts from the WC 2026 schedule."""
    schedule = sportmonks.get_season_schedule()
    fixtures: list[dict] = []
    for entry in schedule:
        # Schedule entries nest: stage → rounds → fixtures
        rounds = entry.get("rounds") or []
        for rnd in rounds:
            for fx in (rnd.get("fixtures") or []):
                fixtures.append(fx)
        # Some entries are direct fixtures
        if entry.get("id") and entry.get("participants"):
            fixtures.append(entry)
    return fixtures


def lookup_country_id(participant: dict) -> int:
    """
    Return the country_id from the Sportmonks participant dict.
    Used as the best available proxy for StatsBomb country_id;
    StatsBomb tables are fetched with a fallback to all rows when no match.
    """
    return participant.get("country_id") or participant.get("id") or 0


# ── Pre-match run ──────────────────────────────────────────────────────────

def run_prematch(fixture_id: int) -> dict | None:
    """
    Full pre-match flow for one fixture.
    Returns the prediction dict for HT reference, or None on failure.
    """
    console.print(Panel(f"[bold cyan]PRE-MATCH[/bold cyan]  fixture_id={fixture_id}",
                        expand=False))
    session = LedgerSession(fixture_id, f"fixture_{fixture_id}", "PRE_MATCH")

    # ── (1) Trigger record ─────────────────────────────────────────────────
    rec_trigger = session.trigger("cron_or_manual")

    # ── (2) Wallet ─────────────────────────────────────────────────────────
    console.print("[dim]Checking wallet…[/dim]")
    try:
        wallet = get_wallet_balance()
    except Exception:
        wallet = 5.0  # dev-day default
    console.print(f"  Wallet: [green]${wallet:.2f}[/green]")

    # ── (3) Season schedule (discover fixtures) ────────────────────────────
    console.print("[dim][1/7] Fetching WC2026 schedule…[/dim]")
    schedule = sportmonks.get_season_schedule()
    rec_sm_schedule = session.tool_call(
        name="sportmonks",
        endpoint=f"/v3/football/schedules/seasons/{config.SEASON_ID}",
        description="List WC2026 season schedule to discover fixtures",
        input_payload={"season_id": config.SEASON_ID},
        output_payload={"entry_count": len(schedule), "picked_fixture_id": fixture_id},
        upstream_ids=[rec_trigger["record_id"]],
    )

    # ── (4) Polymarket slug mapping ────────────────────────────────────────
    console.print("[dim][2/7] Mapping fixture to Polymarket slug…[/dim]")
    pm_slug = pm.get_event_slug(fixture_id)
    rec_pm_slug = session.tool_call(
        name="arena.mapping",
        endpoint="/v1/web/mapping",
        description="Look up curated Polymarket event_slug for this Sportmonks fixture",
        input_payload={"fixture_id": fixture_id},
        output_payload={"polymarket_event_slug": pm_slug},
        upstream_ids=[rec_sm_schedule["record_id"]],
    )
    console.print(f"  PM slug: [cyan]{pm_slug or 'none — predict-only mode'}[/cyan]")

    # ── (5) Sportmonks fixture detail + digest ─────────────────────────────
    console.print("[dim][3/7] Fetching Sportmonks fixture detail…[/dim]")
    fixture = sportmonks.get_fixture(fixture_id)
    participants = fixture.get("participants") or []
    home = next((p for p in participants if p.get("meta", {}).get("location") == "home"), {})
    away = next((p for p in participants if p.get("meta", {}).get("location") == "away"), {})
    home_code = home.get("short_code") or "HOME"
    away_code = away.get("short_code") or "AWAY"
    fixture_name = fixture.get("name", f"{home_code} vs {away_code}")
    # Update session name now that we know it
    session.fixture_name = fixture_name

    console.print(f"  [bold]{fixture_name}[/bold]  ({fixture.get('starting_at','')})")

    rec_sm_fixture = session.tool_call(
        name="sportmonks",
        endpoint=f"/v3/football/fixtures/{fixture_id}",
        description="Fetch fixture detail with predictions, odds, xGFixture",
        input_payload={"fixture_id": fixture_id,
                       "include":    "participants;predictions;odds;xGFixture"},
        output_payload={
            "fixture_name":      fixture_name,
            "kickoff":           fixture.get("starting_at"),
            "predictions_count": len(fixture.get("predictions") or []),
            "odds_count":        len(fixture.get("odds") or []),
        },
        upstream_ids=[rec_sm_schedule["record_id"]],
    )

    console.print("[dim]  Digesting Sportmonks data with Claude…[/dim]")
    sm_digest_result = llm.digest_sportmonks(
        sportmonks_digest_input(fixture, home_code, away_code)
    )
    rec_th_sportmonks = session.thinking(
        prompt_system="[SPORTMONKS_DIGEST_SYS]",
        inputs=[{"record_id": rec_sm_fixture["record_id"],
                 "payload": sm_digest_result.parsed}],
        output_payload=sm_digest_result.parsed,
        provider=sm_digest_result.provider,
        model_name=sm_digest_result.model,
        internal_reasoning=sm_digest_result.thinking,
        tokens_in=sm_digest_result.tokens_in,
        tokens_out=sm_digest_result.tokens_out,
        upstream_ids=[rec_sm_fixture["record_id"]],
    )
    console.print(f"  SM digest: {sm_digest_result.parsed.get('summary','')[:80]}…")

    # ── (6) Polymarket moneyline + digest ──────────────────────────────────
    console.print("[dim][4/7] Fetching Polymarket moneyline…[/dim]")
    moneyline = None
    if pm_slug:
        moneyline = pm.get_moneyline(fixture_id)

    rec_pm_event = session.tool_call(
        name="polymarket-gamma",
        endpoint="/api/v1/data/proxy/polymarket-gamma/events",
        description="Fetch Polymarket event + 3 child winner markets by slug",
        input_payload={"slug": pm_slug},
        output_payload=moneyline or {"status": "no_market"},
        upstream_ids=[rec_pm_slug["record_id"]],
    )
    rec_pm_mids = session.tool_call(
        name="polymarket-clob",
        endpoint="/api/v1/data/proxy/polymarket-clob/midpoint",
        description="Fetch CLOB midpoint per outcome YES token (home/draw/away)",
        input_payload={"token_ids": [
            (moneyline["outcomes"].get(k) or {}).get("token_yes")
            for k in ("home", "draw", "away")
        ] if moneyline else None},
        output_payload={k: (moneyline["outcomes"].get(k) or {}).get("current_mid_yes")
                        for k in ("home", "draw", "away")} if moneyline else None,
        upstream_ids=[rec_pm_event["record_id"]],
    )

    if moneyline:
        mids = {k: (moneyline["outcomes"].get(k) or {}).get("current_mid_yes")
                for k in ("home", "draw", "away")}
        console.print(f"  Market mids: {mids}")
        pm_digest_input = json.dumps(moneyline)
        pm_digest_result = llm.digest_polymarket(pm_digest_input)
    else:
        pm_digest_result = type("R", (), {  # minimal mock
            "parsed": {"data_availability": "no_market", "implied_win_prob": None,
                       "execution_handles": None, "market_handle": None},
            "thinking": "", "model": "", "provider": "",
            "tokens_in": 0, "tokens_out": 0,
        })()
    rec_th_polymarket = session.thinking(
        prompt_system="[POLYMARKET_DIGEST_SYS]",
        inputs=[{"record_id": rec_pm_mids["record_id"],
                 "payload": pm_digest_result.parsed}],
        output_payload=pm_digest_result.parsed,
        provider=pm_digest_result.provider,
        model_name=pm_digest_result.model,
        internal_reasoning=pm_digest_result.thinking,
        tokens_in=pm_digest_result.tokens_in,
        tokens_out=pm_digest_result.tokens_out,
        upstream_ids=[rec_pm_event["record_id"], rec_pm_mids["record_id"]],
    )

    # ── (7) Supabase priors + digest ───────────────────────────────────────
    console.print("[dim][5/7] Fetching Supabase historical priors…[/dim]")

    # First: catalog discovery
    try:
        catalog = supabase_client.get_catalog()
    except Exception:
        catalog = []
    rec_sb_catalog = session.tool_call(
        name="supabase",
        endpoint="/rest/v1/catalog_full",
        description="Discover available Supabase tables via the public catalog",
        input_payload={"select": "table_name,category,row_count,table_description"},
        output_payload={"available_tables": [t.get("table_name") for t in catalog],
                        "count": len(catalog)},
        upstream_ids=[rec_trigger["record_id"]],
    )

    # Look up country_ids from Sportmonks participant (best available proxy)
    home_country_id = lookup_country_id(home)
    away_country_id = lookup_country_id(away)

    priors: dict = {}
    if home_country_id and away_country_id:
        try:
            priors = supabase_client.get_all_priors(home_country_id, away_country_id)
        except Exception as e:
            priors = {"error": str(e)}
    else:
        priors = {"note": f"country_id lookup failed for {home_code}/{away_code}"}

    rec_sb_priors = session.tool_call(
        name="supabase",
        endpoint="/rest/v1/ads_a_country_style+others",
        description="Fetch priors tables for both teams (style, struct, h2h, ko, stage)",
        input_payload={"home_country_id": home_country_id, "away_country_id": away_country_id},
        output_payload=priors,
        upstream_ids=[rec_sb_catalog["record_id"], rec_sm_fixture["record_id"]],
    )

    sb_digest_content = supabase_digest_input(
        fixture_name, home_code, away_code,
        home_country_id, away_country_id,
        home.get("name", home_code), away.get("name", away_code),
        priors,
    )
    sb_digest_result = llm.digest_supabase(sb_digest_content)
    rec_th_supabase = session.thinking(
        prompt_system="[SUPABASE_DIGEST_SYS]",
        inputs=[{"record_id": rec_sb_priors["record_id"],
                 "payload": sb_digest_result.parsed}],
        output_payload=sb_digest_result.parsed,
        provider=sb_digest_result.provider,
        model_name=sb_digest_result.model,
        internal_reasoning=sb_digest_result.thinking,
        tokens_in=sb_digest_result.tokens_in,
        tokens_out=sb_digest_result.tokens_out,
        upstream_ids=[rec_sb_priors["record_id"]],
    )
    console.print(f"  Supabase digest: {sb_digest_result.parsed.get('summary','')[:80]}…")

    # ── (8) Predict (market-blind) ─────────────────────────────────────────
    console.print("[dim][6/7] Predicting (Claude + Gemini ensemble, market-blind)…[/dim]")
    pred_content = predict_input(
        fixture_name, home_code, away_code,
        sm_digest_result.parsed, sb_digest_result.parsed,
    )
    pred_result = llm.predict(pred_content)
    prediction = pred_result.parsed

    rec_th_predict = session.thinking(
        prompt_system="[PREDICT_SYS]",
        inputs=[
            {"record_id": rec_th_sportmonks["record_id"], "payload": sm_digest_result.parsed},
            {"record_id": rec_th_supabase["record_id"],   "payload": sb_digest_result.parsed},
        ],
        output_payload=prediction,
        provider=pred_result.provider,
        model_name=pred_result.model,
        internal_reasoning=pred_result.thinking,
        tokens_in=pred_result.tokens_in,
        tokens_out=pred_result.tokens_out,
        upstream_ids=[rec_th_sportmonks["record_id"], rec_th_supabase["record_id"]],
    )

    # Acting record — prediction (arena scores this for PSL)
    pred_outcome  = prediction.get("outcome", home_code)
    pred_prob     = float(prediction.get("probability", 0.33))
    rec_act_pred  = session.acting_prediction(
        outcome=pred_outcome,
        probability=pred_prob,
        upstream_ids=[rec_th_predict["record_id"]],
    )

    console.print(
        f"  Prediction: [bold]{pred_outcome}[/bold] @ "
        f"[green]{pred_prob:.1%}[/green] "
        f"({prediction.get('confidence_level','?')} confidence)"
    )
    if pred_result.gemini_parsed:
        console.print(f"  Ensemble note: {prediction.get('ensemble_note','')}")

    # ── (9) Strategy ──────────────────────────────────────────────────────
    console.print("[dim][7/7] Deciding trade strategy…[/dim]")
    strat_content = strategy_input(prediction, pm_digest_result.parsed)
    strat_result  = llm.strategy(strat_content)
    strategy_data = strat_result.parsed

    rec_th_strategy = session.thinking(
        prompt_system="[STRATEGY_SYS]",
        inputs=[
            {"record_id": rec_th_predict["record_id"],    "payload": prediction},
            {"record_id": rec_th_polymarket["record_id"], "payload": pm_digest_result.parsed},
        ],
        output_payload=strategy_data,
        provider=strat_result.provider,
        model_name=strat_result.model,
        internal_reasoning=strat_result.thinking,
        tokens_in=strat_result.tokens_in,
        tokens_out=strat_result.tokens_out,
        upstream_ids=[rec_th_predict["record_id"], rec_th_polymarket["record_id"]],
    )

    console.print(f"  Strategy: should_trade={strategy_data.get('should_trade')}  "
                  f"edge={strategy_data.get('edge_pp',0):+.1f}pp  "
                  f"size=${strategy_data.get('size_usdc',0):.2f}")

    # ── (10) Order ────────────────────────────────────────────────────────
    order_response = None
    order_payload  = None

    if strategy_data.get("should_trade") and pm_slug:
        team_code   = strategy_data.get("team_code") or strategy_data.get("outcome")
        size_usdc   = float(strategy_data.get("size_usdc") or 0)
        limit_price = float(strategy_data.get("limit_price") or 0)

        if team_code and size_usdc > 0 and limit_price > 0:
            order_payload = {
                "fixture_code":          str(fixture_id),
                "team_code":             team_code,
                "usd_size":              str(round(size_usdc, 2)),
                "limit_price":           round(limit_price, 4),
                "time_in_force_seconds": config.DEFAULT_TIF_SECONDS,
                "idempotency_key":       str(uuid.uuid4()),
            }
            try:
                order_response = place_order(fixture_id, team_code, size_usdc, limit_price)
                order_id       = order_response.get("order_id") or order_response.get("status")
                ok             = isinstance(order_response, dict) and "order_id" in order_response
                console.print(f"  Order: [{'green' if ok else 'yellow'}]{order_id}[/]")
            except Exception as e:
                order_response = {"error": str(e)}
                console.print(f"  Order failed: [red]{e}[/red]")

            submitted_ok = isinstance(order_response, dict) and "order_id" in order_response
            session.acting_order(
                direction=strategy_data.get("direction", "long"),
                outcome=strategy_data.get("outcome", ""),
                size_usdc=size_usdc,
                limit_price=limit_price,
                order_payload=order_payload,
                execution_status="pending" if submitted_ok else "failed",
                execution_id=order_response.get("order_id") if submitted_ok else None,
                upstream_ids=[rec_th_strategy["record_id"]],
            )

    # ── (11) Submit ledger ────────────────────────────────────────────────
    console.print(
        f"\n[dim]Submitting ledger trace "
        f"({session.record_count()} records)…[/dim]"
    )
    ledger_resp = session.submit()
    n_stored = len(ledger_resp.get("records") or [])
    n_errors = len(ledger_resp.get("errors") or [])
    if ledger_resp.get("status") == "404_not_yet_live":
        console.print("  [yellow]Ledger endpoint not yet live on staging.[/yellow]")
    else:
        console.print(
            f"  Ledger: [green]{n_stored} stored[/green]"
            f"{', ' + str(n_errors) + ' error(s)' if n_errors else ''}"
        )
        for e in (ledger_resp.get("errors") or []):
            console.print(f"    [red]#{e.get('index')}: {e.get('code')}: {e.get('message')}[/red]")

    console.print(Panel(
        f"[bold green]Pre-match complete.[/bold green]  "
        f"Predicted [bold]{pred_outcome}[/bold] @ {pred_prob:.1%}  "
        f"{'| Trade: ' + str(strategy_data.get('size_usdc','skip')) + 'usdc' if strategy_data.get('should_trade') else '| No trade'}",
        expand=False,
    ))
    return prediction


# ── Half-time run ──────────────────────────────────────────────────────────

def run_halftime(fixture_id: int, prematch_prediction: dict | None = None) -> None:
    console.print(Panel(f"[bold yellow]HALF-TIME[/bold yellow]  fixture_id={fixture_id}",
                        expand=False))
    session = LedgerSession(fixture_id, f"fixture_{fixture_id}", "HT")
    rec_trigger = session.trigger("ht_window_cron")

    try:
        wallet = get_wallet_balance()
    except Exception:
        wallet = 5.0
    console.print(f"  Wallet at HT: [green]${wallet:.2f}[/green]")

    # Fixture + team info
    fixture = sportmonks.get_fixture(fixture_id)
    participants = fixture.get("participants") or []
    home = next((p for p in participants if p.get("meta", {}).get("location") == "home"), {})
    away = next((p for p in participants if p.get("meta", {}).get("location") == "away"), {})
    home_code = home.get("short_code") or "HOME"
    away_code = away.get("short_code") or "AWAY"
    fixture_name = fixture.get("name", f"{home_code} vs {away_code}")
    session.fixture_name = fixture_name

    # HT stats from Sportmonks
    ht_stats_sm = sportmonks.extract_ht_stats(fixture)
    rec_sm_ht = session.tool_call(
        name="sportmonks",
        endpoint=f"/v3/football/fixtures/{fixture_id}",
        description="Fetch HT fixture state (scores, statistics, xG)",
        input_payload={"fixture_id": fixture_id},
        output_payload=ht_stats_sm,
        upstream_ids=[rec_trigger["record_id"]],
    )

    # Supabase HT checkpoint
    ht_snapshot = supabase_client.get_ht_snapshot(fixture_id)
    ht_score    = supabase_client.get_ht_score(fixture_id)
    rec_sb_ht = session.tool_call(
        name="supabase",
        endpoint="/rest/v1/d_checkpoint_snapshot",
        description="Fetch Supabase HT checkpoint snapshot",
        input_payload={"fixture_id": fixture_id, "checkpoint": "HT"},
        output_payload={"snapshot": ht_snapshot, "score": ht_score},
        upstream_ids=[rec_sm_ht["record_id"]],
    )

    # Updated Polymarket prices
    pm_slug    = pm.get_event_slug(fixture_id)
    moneyline  = pm.get_moneyline(fixture_id) if pm_slug else None
    rec_pm_ht = session.tool_call(
        name="polymarket-clob",
        endpoint="/api/v1/data/proxy/polymarket-clob/midpoint",
        description="Fetch updated CLOB mid prices at HT",
        input_payload={"fixture_id": fixture_id},
        output_payload=moneyline or {"status": "no_market"},
        upstream_ids=[rec_trigger["record_id"]],
    )

    if moneyline:
        pm_digest_result = llm.digest_polymarket(json.dumps(moneyline))
    else:
        pm_digest_result = type("R", (), {
            "parsed": {"data_availability": "no_market", "implied_win_prob": None,
                       "execution_handles": None, "market_handle": None},
            "thinking": "", "model": "", "provider": "",
            "tokens_in": 0, "tokens_out": 0,
        })()

    rec_th_pm = session.thinking(
        prompt_system="[POLYMARKET_DIGEST_SYS]",
        inputs=[{"record_id": rec_pm_ht["record_id"], "payload": pm_digest_result.parsed}],
        output_payload=pm_digest_result.parsed,
        provider=pm_digest_result.provider,
        model_name=pm_digest_result.model,
        internal_reasoning=pm_digest_result.thinking,
        tokens_in=pm_digest_result.tokens_in,
        tokens_out=pm_digest_result.tokens_out,
        upstream_ids=[rec_pm_ht["record_id"]],
    )

    # HT prediction
    console.print("[dim]Predicting at HT (Bayesian update)…[/dim]")
    ht_pred_content = ht_predict_input(
        fixture_name, home_code, away_code,
        prematch_prediction, ht_snapshot, ht_score, ht_stats_sm,
    )
    ht_pred_result = llm.ht_predict(ht_pred_content)
    prediction = ht_pred_result.parsed
    pred_outcome = prediction.get("outcome", home_code)
    pred_prob    = float(prediction.get("probability", 0.33))

    rec_th_pred = session.thinking(
        prompt_system="[HT_PREDICT_SYS]",
        inputs=[
            {"record_id": rec_sb_ht["record_id"], "payload": ht_snapshot},
            {"record_id": rec_th_pm["record_id"], "payload": pm_digest_result.parsed},
        ],
        output_payload=prediction,
        provider=ht_pred_result.provider,
        model_name=ht_pred_result.model,
        internal_reasoning=ht_pred_result.thinking,
        tokens_in=ht_pred_result.tokens_in,
        tokens_out=ht_pred_result.tokens_out,
        upstream_ids=[rec_sb_ht["record_id"], rec_th_pm["record_id"]],
    )

    session.acting_prediction(
        outcome=pred_outcome,
        probability=pred_prob,
        upstream_ids=[rec_th_pred["record_id"]],
    )
    console.print(
        f"  HT prediction: [bold]{pred_outcome}[/bold] @ [green]{pred_prob:.1%}[/green]  "
        f"changed={prediction.get('changed_from_prematch', '?')}"
    )

    # Strategy
    strat_content = strategy_input(prediction, pm_digest_result.parsed)
    strat_result  = llm.strategy(strat_content)
    strategy_data = strat_result.parsed

    rec_th_strat = session.thinking(
        prompt_system="[STRATEGY_SYS]",
        inputs=[
            {"record_id": rec_th_pred["record_id"], "payload": prediction},
            {"record_id": rec_th_pm["record_id"],   "payload": pm_digest_result.parsed},
        ],
        output_payload=strategy_data,
        provider=strat_result.provider,
        model_name=strat_result.model,
        internal_reasoning=strat_result.thinking,
        tokens_in=strat_result.tokens_in,
        tokens_out=strat_result.tokens_out,
        upstream_ids=[rec_th_pred["record_id"], rec_th_pm["record_id"]],
    )

    if strategy_data.get("should_trade") and pm_slug:
        team_code   = strategy_data.get("team_code") or strategy_data.get("outcome")
        size_usdc   = float(strategy_data.get("size_usdc") or 0)
        limit_price = float(strategy_data.get("limit_price") or 0)
        if team_code and size_usdc > 0:
            try:
                order_response = place_order(fixture_id, team_code, size_usdc, limit_price)
                ok = "order_id" in order_response
            except Exception as e:
                order_response = {"error": str(e)}
                ok = False
            order_payload = {"fixture_code": str(fixture_id), "team_code": team_code,
                             "usd_size": str(size_usdc), "limit_price": limit_price}
            session.acting_order(
                direction=strategy_data.get("direction", "long"),
                outcome=strategy_data.get("outcome", ""),
                size_usdc=size_usdc,
                limit_price=limit_price,
                order_payload=order_payload,
                execution_status="pending" if ok else "failed",
                execution_id=order_response.get("order_id") if ok else None,
                upstream_ids=[rec_th_strat["record_id"]],
            )

    console.print(f"[dim]Submitting HT ledger ({session.record_count()} records)…[/dim]")
    ledger_resp = session.submit()
    n_stored = len(ledger_resp.get("records") or [])
    console.print(Panel(
        f"[bold green]HT complete.[/bold green]  {fixture_name}  "
        f"Predicted {pred_outcome} @ {pred_prob:.1%}",
        expand=False,
    ))


# ── Scanner ────────────────────────────────────────────────────────────────

def scan_and_run(window: str = "prematch") -> None:
    console.print("[bold]Scanning for upcoming WC2026 fixtures…[/bold]")
    schedule = sportmonks.get_season_schedule()
    _print_schedule(schedule)
    fixtures = []
    for entry in schedule:
        for rnd in (entry.get("rounds") or []):
            for fx in (rnd.get("fixtures") or []):
                fixtures.append(fx)
        if entry.get("id") and entry.get("participants"):
            fixtures.append(entry)

    console.print(f"Found {len(fixtures)} fixtures. Processing…\n")
    for fx in fixtures:
        fid = fx.get("id")
        if not fid:
            continue
        try:
            if window == "halftime":
                run_halftime(fid)
            else:
                run_prematch(fid)
        except Exception as e:
            console.print(f"[red]Error on fixture {fid}: {e}[/red]")
        time.sleep(3)


def _print_schedule(schedule: list[dict]) -> None:
    t = Table(title="WC2026 Schedule")
    t.add_column("Stage", style="cyan")
    t.add_column("Fixtures")
    for entry in schedule[:10]:
        name = entry.get("name") or entry.get("stage", {}).get("name") or "?"
        rounds = entry.get("rounds") or []
        count  = sum(len(r.get("fixtures") or []) for r in rounds)
        t.add_row(name, str(count))
    console.print(t)


def test_connection() -> None:
    """Smoke test all three data sources."""
    console.print("[bold]Testing API connections…[/bold]")
    # Arena
    try:
        resp = httpx.get(f"{config.ARENA_API}/v1/arena/agents/me", headers=_H, timeout=10)
        console.print(f"  Arena agents/me: [green]HTTP {resp.status_code}[/green]  "
                      f"wallet=${resp.json().get('wallet_balance_usd','?')}")
    except Exception as e:
        console.print(f"  Arena: [red]{e}[/red]")

    # Sportmonks
    try:
        data = sportmonks.get_season_schedule()
        console.print(f"  Sportmonks schedule: [green]{len(data)} entries[/green]")
    except Exception as e:
        console.print(f"  Sportmonks: [red]{e}[/red]")

    # Supabase
    try:
        cat = supabase_client.get_catalog()
        tables = [t.get("table_name") for t in cat]
        console.print(f"  Supabase catalog: [green]{len(tables)} tables[/green] — "
                      f"{', '.join(tables[:5])}…")
    except Exception as e:
        console.print(f"  Supabase: [red]{e}[/red]")

    # Polymarket mapping
    try:
        slug = pm.get_event_slug(19609127)
        console.print(f"  Polymarket mapping (fixture 19609127): [green]{slug}[/green]")
    except Exception as e:
        console.print(f"  Polymarket mapping: [red]{e}[/red]")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World Cup Arena Agent")
    parser.add_argument("--fixture-id", type=int,
                        help="Sportmonks fixture id, e.g. 19609127")
    parser.add_argument("--window", choices=["prematch", "halftime"],
                        default="prematch")
    parser.add_argument("--scan", action="store_true",
                        help="Auto-scan all WC2026 fixtures and run for each")
    parser.add_argument("--test-connection", action="store_true",
                        help="Smoke-test all three data sources")
    args = parser.parse_args()

    if args.test_connection:
        test_connection()
    elif args.scan:
        scan_and_run(window=args.window)
    elif args.fixture_id:
        if args.window == "halftime":
            run_halftime(args.fixture_id)
        else:
            run_prematch(args.fixture_id)
    else:
        parser.print_help()
