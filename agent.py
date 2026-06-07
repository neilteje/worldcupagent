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
from data import web_search, reddit_sentiment, kalshi
from reasoning import llm, council, gates
from reasoning.prompts import (
    sportmonks_digest_input, supabase_digest_input,
    predict_input, strategy_input, ht_predict_input,
)
from ledger.client import LedgerSession
from betting.kelly import should_bet, expected_value, kelly_usd
from betting import decision as ev_decision

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
    Resolve a participant to its StatsBomb country_id (e.g. Mexico → 147).

    The Supabase priors tables key on StatsBomb country_id, which differs from
    Sportmonks ids. We resolve by team name against the names embedded in the
    h2h table; if that fails we fall back to the Sportmonks id (the priors fetch
    then degrades to its all-rows fallback).
    """
    name = participant.get("name") or ""
    cid = supabase_client.resolve_country_id(name)
    if cid:
        return cid
    return participant.get("country_id") or participant.get("id") or 0


def _outcome_key(outcome: str, home_code: str, away_code: str) -> str | None:
    """Map a council outcome (team_code | 'draw') to the home/draw/away slot."""
    if outcome == "draw":
        return "draw"
    if outcome == home_code:
        return "home"
    if outcome == away_code:
        return "away"
    return None


def _pm_mid_for(moneyline: dict | None, outcome: str,
                home_code: str, away_code: str) -> float | None:
    """Polymarket YES mid for our predicted outcome."""
    if not moneyline:
        return None
    key = _outcome_key(outcome, home_code, away_code)
    if key is None:
        return None
    return (moneyline.get("outcomes", {}).get(key) or {}).get("current_mid_yes")


def _kalshi_mid_for(kalshi_ml: dict | None, outcome: str,
                    home_code: str, away_code: str) -> float | None:
    """Kalshi YES mid for our predicted outcome."""
    if not kalshi_ml:
        return None
    key = _outcome_key(outcome, home_code, away_code)
    return kalshi_ml.get(key) if key else None


def _print_decision_table(decision: ev_decision.GameDecision) -> None:
    """Render the per-outcome EV / payout table so every game shows its math."""
    t = Table(title="Per-outcome EV (de-vigged)")
    t.add_column("Outcome", style="cyan")
    t.add_column("Our prob")
    t.add_column("Pay (raw)")
    t.add_column("Fair")
    t.add_column("Edge vs fair")
    t.add_column("EV / $1")
    t.add_column("Kelly $")
    for e in decision.ranked:
        mark = " ★" if (decision.best and e.slot == decision.best.slot
                        and decision.should_trade) else ""
        t.add_row(
            f"{e.code}{mark}",
            f"{e.our_prob:.0%}",
            f"{e.raw_mid:.0%}" if e.raw_mid is not None else "—",
            f"{e.fair_prob:.0%}" if e.fair_prob is not None else "—",
            f"{e.edge_vs_fair*100:+.1f}pp",
            f"{e.ev_per_dollar*100:+.1f}%",
            f"${e.kelly_usd:.2f}" if e.kelly_usd else "—",
        )
    console.print(t)
    ov = f"{decision.overround*100:+.1f}%" if decision.overround is not None else "n/a"
    console.print(f"  [dim]market overround (vig): {ov}[/dim]  {decision.summary}")


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

    # ── (5b) External research: web search + Reddit crowd sentiment ─────────
    home_name = home.get("name", home_code)
    away_name = away.get("name", away_code)
    match_date = str(fixture.get("starting_at", ""))[:10]
    have_lineups = any(
        (p.get("meta", {}) or {}).get("position") for p in participants
    )

    console.print("[dim]  Researching injuries / lineups / previews…[/dim]")
    web_research = web_search.gather_research(
        home_name, away_name, match_date, have_confirmed_lineups=have_lineups)
    rec_web = session.tool_call(
        name="web_search",
        endpoint=f"{web_research.get('backend','none')}:search",
        description="Targeted injury/lineup/preview search for both teams",
        input_payload={"home": home_name, "away": away_name, "date": match_date,
                       "skip_lineups": have_lineups},
        output_payload=web_research,
        via="external.web",
        success=web_research.get("total_results", 0) > 0,
        upstream_ids=[rec_sm_fixture["record_id"]],
    )
    console.print(f"  Web: {web_research.get('total_results',0)} results "
                  f"from {len(web_research.get('sources',[]))} sources "
                  f"via {web_research.get('backend')}")

    console.print("[dim]  Pulling r/soccer crowd sentiment…[/dim]")
    reddit_bundle = reddit_sentiment.get_sentiment_bundle(home_name, away_name)
    rec_reddit = session.tool_call(
        name="reddit",
        endpoint="r/soccer/search.json",
        description="Fetch top r/soccer thread comments for crowd sentiment",
        input_payload={"query": f"{home_name} {away_name}"},
        output_payload=reddit_bundle,
        via="external.reddit",
        success=reddit_bundle.get("threads_found", 0) > 0,
        upstream_ids=[rec_sm_fixture["record_id"]],
    )
    console.print(f"  Reddit: {reddit_bundle.get('threads_found',0)} threads, "
                  f"{len(reddit_bundle.get('top_comments',[]))} comments")

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

    # ── (6b) Kalshi cross-market odds ──────────────────────────────────────
    console.print("[dim]  Fetching Kalshi cross-market odds…[/dim]")
    kalshi_ml = kalshi.get_moneyline(home_name, away_name)
    rec_kalshi = session.tool_call(
        name="kalshi",
        endpoint="/trade-api/v2/markets",
        description="Fetch Kalshi moneyline to triangulate against Polymarket",
        input_payload={"home": home_name, "away": away_name},
        output_payload=kalshi_ml,
        via="external.kalshi",
        success=kalshi_ml.get("markets_found", 0) > 0,
        upstream_ids=[rec_pm_mids["record_id"]],
    )
    console.print(f"  Kalshi: {kalshi_ml.get('markets_found',0)} markets matched")

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

    # ── (8) Reasoning council: Pulse → Scout → Analyst → Devil → Judge ─────
    console.print("[dim][6/7] Convening reasoning council…[/dim]")
    cr = council.run_council(
        fixture_name, home_code, away_code,
        home_name, away_name, str(fixture.get("starting_at", "")),
        sm_digest_result.parsed, sb_digest_result.parsed,
        pm_digest_result.parsed, kalshi_ml,
        web_research, reddit_bundle,
    )

    # Grok social pulse → ToolCalling record (live X/Twitter intelligence)
    rec_pulse = session.tool_call(
        name="grok",
        endpoint=f"{config.XAI_BASE_URL}/chat/completions",
        description="Grok live X/Twitter + news social pulse for the fixture",
        input_payload={"home": home_name, "away": away_name},
        output_payload=cr.social_pulse,
        via="external.xai",
        success=bool(cr.social_pulse),
        upstream_ids=[rec_sm_fixture["record_id"]],
    )
    pulse_lean = (cr.social_pulse or {}).get("overall_lean", "n/a")
    console.print(f"  Grok pulse: lean={pulse_lean} "
                  f"({(cr.social_pulse or {}).get('confidence','?')} conf)")

    # Scout → Planning record
    rec_scout = session.planning(
        description="Scout triage of injuries / lineups / crowd + social pulse",
        output_payload=cr.scout.parsed if cr.scout else {},
        provider=cr.scout.provider if cr.scout else "",
        model_name=cr.scout.model if cr.scout else "",
        internal_reasoning=cr.scout.thinking if cr.scout else "",
        tokens_in=cr.scout.tokens_in if cr.scout else 0,
        tokens_out=cr.scout.tokens_out if cr.scout else 0,
        inputs=[
            {"record_id": rec_web["record_id"], "payload": web_research},
            {"record_id": rec_reddit["record_id"], "payload": reddit_bundle},
            {"record_id": rec_pulse["record_id"], "payload": cr.social_pulse},
        ],
        upstream_ids=[rec_web["record_id"], rec_reddit["record_id"],
                      rec_pulse["record_id"], rec_th_sportmonks["record_id"]],
    )
    console.print(f"  Scout: {len(cr.scout_flags)} flag(s)")

    # Analyst → Thinking record (market-blind)
    rec_analyst = session.thinking(
        prompt_system="[ANALYST_SYS]",
        inputs=[
            {"record_id": rec_th_sportmonks["record_id"], "payload": sm_digest_result.parsed},
            {"record_id": rec_th_supabase["record_id"], "payload": sb_digest_result.parsed},
            {"record_id": rec_scout["record_id"], "payload": cr.scout.parsed if cr.scout else {}},
        ],
        output_payload=cr.analyst.parsed if cr.analyst else {},
        provider=cr.analyst.provider if cr.analyst else "",
        model_name=cr.analyst.model if cr.analyst else "",
        internal_reasoning=cr.analyst.thinking if cr.analyst else "",
        tokens_in=cr.analyst.tokens_in if cr.analyst else 0,
        tokens_out=cr.analyst.tokens_out if cr.analyst else 0,
        upstream_ids=[rec_th_sportmonks["record_id"], rec_th_supabase["record_id"],
                      rec_scout["record_id"]],
    )

    # Devil's advocate → Thinking record (raw chain-of-thought)
    rec_devil = session.thinking(
        prompt_system="[DEVIL_SYS]",
        inputs=[{"record_id": rec_analyst["record_id"],
                 "payload": cr.analyst.parsed if cr.analyst else {}}],
        output_payload=cr.devil.parsed if cr.devil else {},
        provider=cr.devil.provider if cr.devil else "",
        model_name=cr.devil.model if cr.devil else "",
        internal_reasoning=cr.devil.thinking if cr.devil else "",
        tokens_in=cr.devil.tokens_in if cr.devil else 0,
        tokens_out=cr.devil.tokens_out if cr.devil else 0,
        upstream_ids=[rec_analyst["record_id"]],
    )

    # Judge → Thinking record (synthesis, sees markets) — DAG converges here
    rec_judge = session.thinking(
        prompt_system="[JUDGE_SYS]",
        inputs=[
            {"record_id": rec_analyst["record_id"], "payload": cr.analyst.parsed if cr.analyst else {}},
            {"record_id": rec_devil["record_id"], "payload": cr.devil.parsed if cr.devil else {}},
            {"record_id": rec_th_polymarket["record_id"], "payload": pm_digest_result.parsed},
            {"record_id": rec_kalshi["record_id"], "payload": kalshi_ml},
        ],
        output_payload=cr.judge.parsed if cr.judge else {},
        provider=cr.judge.provider if cr.judge else "",
        model_name=cr.judge.model if cr.judge else "",
        internal_reasoning=cr.judge.thinking if cr.judge else "",
        tokens_in=cr.judge.tokens_in if cr.judge else 0,
        tokens_out=cr.judge.tokens_out if cr.judge else 0,
        upstream_ids=[rec_analyst["record_id"], rec_devil["record_id"],
                      rec_th_polymarket["record_id"], rec_kalshi["record_id"]],
    )

    pred_outcome = cr.outcome
    pred_prob = float(cr.probability)
    prediction = {
        "outcome": pred_outcome,
        "probability": pred_prob,
        "probabilities": cr.probabilities,
        "confidence_level": cr.confidence,
        "rationale": cr.council_summary,
        "market_alignment": cr.market_alignment,
    }

    # Acting record — prediction (arena scores this for PSL)
    session.acting_prediction(
        outcome=pred_outcome,
        probability=pred_prob,
        upstream_ids=[rec_judge["record_id"]],
    )
    console.print(
        f"  Council verdict: [bold]{pred_outcome}[/bold] @ "
        f"[green]{pred_prob:.1%}[/green] ({cr.confidence} confidence, "
        f"market {cr.market_alignment})"
    )

    # ── (9) EV-ranked decision across ALL outcomes + deterministic gates ───
    console.print("[dim][7/7] Ranking all outcomes by EV + running gates…[/dim]")

    # The EV engine evaluates home/draw/away (not just the favorite), de-vigs the
    # market, and picks the highest-EV tradable side. The gates then act as a risk
    # overlay (scout veto / cross-market consensus / confidence) on THAT side.
    decision = ev_decision.evaluate_game(
        cr.probabilities, moneyline, home_code, away_code, wallet,
    )
    _print_decision_table(decision)

    best = decision.best
    trade_outcome = best.code if best else pred_outcome
    pm_mid = best.raw_mid if best else None
    trade_prob = best.our_prob if best else pred_prob
    kalshi_mid = _kalshi_mid_for(kalshi_ml, trade_outcome, home_code, away_code)

    gate = gates.evaluate_gates(
        outcome=trade_outcome,
        model_prob=trade_prob,
        pm_mid=pm_mid,
        kalshi_mid=kalshi_mid,
        scout_flags=cr.scout_flags,
        confidence=cr.confidence,
        wallet_balance=wallet,
    )
    # Trade only when BOTH the EV engine finds a +EV side and the gates allow it.
    should_trade = bool(decision.should_trade and gate.should_trade)

    size_usdc = 0.0
    limit_price = 0.0
    if should_trade and pm_mid and best:
        # Base size from the EV engine's half-Kelly, scaled by the gate multiplier.
        size_usdc = round(min(best.kelly_usd * gate.bet_multiplier,
                              config.MAX_BET_USD, wallet), 2)
        if size_usdc < 1.0:
            should_trade = False
            gate.reasons.append(f"sized ${size_usdc:.2f} below $1 minimum")
            size_usdc = 0.0
        else:
            limit_price = min(round(pm_mid + 0.02, 4), 0.99)

    rec_gate = session.planning(
        description="EV-ranked all-outcome decision + deterministic gates + Kelly sizing",
        output_payload={
            "should_trade": should_trade,
            "trade_outcome": trade_outcome,
            "decision_summary": decision.summary,
            "market_overround": decision.overround,
            "ranked_outcomes": [
                {"outcome": e.code, "our_prob": e.our_prob, "raw_mid": e.raw_mid,
                 "fair_prob": e.fair_prob, "edge_vs_fair": e.edge_vs_fair,
                 "ev_per_dollar": e.ev_per_dollar, "kelly_usd": e.kelly_usd,
                 "tradable": e.tradable}
                for e in decision.ranked
            ],
            "edge": round(gate.edge, 4),
            "bet_multiplier": gate.bet_multiplier,
            "market_agreement": gate.market_agreement,
            "veto_reason": gate.veto_reason,
            "reasons": gate.reasons,
            "pm_mid": pm_mid,
            "kalshi_mid": kalshi_mid,
            "size_usdc": size_usdc,
            "limit_price": limit_price,
        },
        upstream_ids=[rec_judge["record_id"], rec_kalshi["record_id"]],
    )
    ev_str = f"  EV={best.ev_per_dollar*100:+.1f}%/$" if best else ""
    console.print(f"  Decision: should_trade={should_trade}  "
                  f"side={trade_outcome}{ev_str}")
    console.print(f"  Gates: ×{gate.bet_multiplier}  size=${size_usdc:.2f}"
                  + (f"  [yellow](veto: {gate.veto_reason})[/yellow]"
                     if gate.veto_reason else ""))

    # ── (10) Order ────────────────────────────────────────────────────────
    order_response = None
    if should_trade and pm_slug and size_usdc >= 1.0 and limit_price > 0:
        team_code = trade_outcome
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
            order_id = order_response.get("order_id") or order_response.get("status")
            ok = isinstance(order_response, dict) and "order_id" in order_response
            console.print(f"  Order: [{'green' if ok else 'yellow'}]{order_id}[/]")
        except Exception as e:
            order_response = {"error": str(e)}
            console.print(f"  Order failed: [red]{e}[/red]")

        submitted_ok = isinstance(order_response, dict) and "order_id" in order_response
        session.acting_order(
            direction="long",
            outcome=trade_outcome,
            size_usdc=size_usdc,
            limit_price=limit_price,
            order_payload=order_payload,
            execution_status="pending" if submitted_ok else "failed",
            execution_id=order_response.get("order_id") if submitted_ok else None,
            upstream_ids=[rec_gate["record_id"]],
        )

    # ── (10b) Closing reflection ───────────────────────────────────────────
    session.reflecting(
        description="Post-decision reflection on the council run",
        output_payload={
            "fixture": fixture_name,
            "final_pick": pred_outcome,
            "final_probability": pred_prob,
            "trade_side": trade_outcome if should_trade else None,
            "decision_summary": decision.summary,
            "market_alignment": cr.market_alignment,
            "traded": bool(should_trade and size_usdc >= 1.0),
            "size_usdc": size_usdc,
            "edge": round(gate.edge, 4),
            "devils_advocate_raised": (cr.devil.parsed or {}).get("strongest_risks")
                                       if cr.devil else None,
            "what_to_improve": (
                "Tighten priors mapping and confirm lineups closer to kickoff; "
                "revisit if market and council diverge after team news."
            ),
            "social_pulse_lean": (cr.social_pulse or {}).get("overall_lean"),
            "data_gaps": {
                "web_results": web_research.get("total_results", 0),
                "web_sources": len(web_research.get("sources", [])),
                "reddit_comments": len(reddit_bundle.get("top_comments", [])),
                "kalshi_markets": kalshi_ml.get("markets_found", 0),
                "grok_pulse": bool(cr.social_pulse),
            },
        },
        upstream_ids=[rec_judge["record_id"], rec_gate["record_id"]],
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

    traded = should_trade and size_usdc >= 1.0
    trade_str = (f"| Trade: ${size_usdc:.2f} on {trade_outcome}"
                 if traded else "| No trade (prediction still scored)")
    console.print(Panel(
        f"[bold green]Pre-match complete.[/bold green]  "
        f"Predicted [bold]{pred_outcome}[/bold] @ {pred_prob:.1%}  "
        f"{trade_str}",
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


def list_fixtures() -> None:
    """Print every WC2026 fixture with its id, so you can pick one to run."""
    console.print("[bold]Fetching WC2026 fixtures…[/bold]")
    fixtures = get_season_fixtures()
    fixtures = [f for f in fixtures if f.get("id")]
    fixtures.sort(key=lambda f: str(f.get("starting_at", "")))
    t = Table(title=f"WC2026 Fixtures ({len(fixtures)})")
    t.add_column("Fixture ID", style="cyan", no_wrap=True)
    t.add_column("Kickoff (UTC)", style="dim")
    t.add_column("Match")
    for fx in fixtures:
        t.add_row(str(fx.get("id")), str(fx.get("starting_at", "?")),
                  fx.get("name", "?"))
    console.print(t)
    console.print("\nRun one with: "
                  "[green]python agent.py --fixture-id <ID> --window prematch[/green]")


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

    # Country-id resolver (Supabase priors keying)
    try:
        mex = supabase_client.resolve_country_id("Mexico")
        console.print(f"  Country-id resolver (Mexico): [green]{mex}[/green] "
                      f"(expect 147)")
    except Exception as e:
        console.print(f"  Country-id resolver: [red]{e}[/red]")

    # Web search (Serper / DDG)
    try:
        wr = web_search.gather_research("Mexico", "South Africa", "2026-06-11")
        console.print(f"  Web search: [green]{wr['total_results']} results "
                      f"from {len(wr['sources'])} sources[/green] "
                      f"via {wr['backend']}")
    except Exception as e:
        console.print(f"  Web search: [red]{e}[/red]")

    # Kalshi
    try:
        km = kalshi.get_moneyline("Mexico", "South Africa")
        console.print(f"  Kalshi: [green]{km['markets_found']} markets[/green]")
    except Exception as e:
        console.print(f"  Kalshi: [red]{e}[/red]")

    # LLM providers
    providers = []
    if config.ANTHROPIC_KEY: providers.append("anthropic")
    if config.DEEPSEEK_KEY:  providers.append("deepseek")
    if config.XAI_KEY:       providers.append("grok")
    if config.GEMINI_KEY:    providers.append("gemini")
    if config.OPENAI_KEY:    providers.append("openai")
    console.print(f"  LLM providers configured: [green]{', '.join(providers)}[/green]")
    if not config.XAI_KEY:
        console.print("    [yellow]Grok (XAI_API_KEY) not set — social pulse "
                      "will be skipped.[/yellow]")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World Cup Arena Agent")
    parser.add_argument("--fixture-id", type=int,
                        help="Sportmonks fixture id, e.g. 19609127")
    parser.add_argument("--window", choices=["prematch", "halftime"],
                        default="prematch")
    parser.add_argument("--scan", action="store_true",
                        help="Auto-scan all WC2026 fixtures and run for each")
    parser.add_argument("--list", action="store_true",
                        help="List every WC2026 fixture with its id, then exit")
    parser.add_argument("--test-connection", action="store_true",
                        help="Smoke-test all data sources")
    args = parser.parse_args()

    if args.list:
        list_fixtures()
    elif args.test_connection:
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
