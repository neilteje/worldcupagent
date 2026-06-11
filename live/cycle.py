"""
One fixture × window cycle for the 4-agent portfolio.

The expensive part — data fetches + LLM digests + the council — runs ONCE per
window (the agents deliberately share one brain, per docs/STRATEGY.md). Each
agent then runs its own cheap, deterministic tail: wallet → policy →
order(s) → its own full reasoning-ledger session under its own API key.

PRE_MATCH uses the full council; HT uses the Bayesian HT update (and the
runner only calls it when the arena says the HT window is actually open —
release notes 20260610: HT is not enabled yet).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any

import config
from data import sportmonks, supabase_client, polymarket as pm
from data import web_search, reddit_sentiment, kalshi
from data import fixture_bundle
from reasoning import llm, council, gates
from reasoning.prompts import ht_predict_input
from ledger.client import LedgerSession
from betting import policy as bet_policy
from harness.profiles import confidence_to_num
from live.arena_client import ArenaClient
from live.roster import LiveAgent
from live import metrics

# Per-cycle spend ceiling per agent: min($5 arena rule, wallet − 5¢ buffer).
WALLET_BUFFER_USD = 0.05


class _EmptyResult:
    """Stand-in for a failed/skipped LLM step (keeps ledger building uniform)."""
    parsed: dict = {}
    thinking = ""
    model = ""
    provider = ""
    tokens_in = 0
    tokens_out = 0


@dataclass
class Forecast:
    """Everything one window's shared brain produced."""
    fixture_id: int
    window: str                      # "PRE_MATCH" | "HT"
    fixture_name: str = ""
    kickoff: str = ""
    home_code: str = "HOME"
    away_code: str = "AWAY"
    home_name: str = ""
    away_name: str = ""
    pm_slug: str | None = None
    moneyline: dict | None = None
    market_source: str = "none"
    mids: dict = field(default_factory=dict)
    sm_digest: dict | None = None
    sb_digest: dict | None = None
    pm_digest_result: Any = None     # LLM result obj (parsed/provider/model/…)
    kalshi_ml: dict = field(default_factory=dict)
    web_research: dict = field(default_factory=dict)
    reddit_bundle: dict = field(default_factory=dict)
    # PRE_MATCH: council result. HT: ht_pred result + dict.
    cr: Any = None
    ht_pred_result: Any = None
    ht_context: dict = field(default_factory=dict)
    # Unified outputs every consumer reads:
    probabilities: dict = field(default_factory=dict)
    outcome: str = ""
    probability: float = 0.0
    confidence: str = "low"
    scout_flags: list = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    summary: str = ""


def _fetch_market(fixture_id: int) -> tuple[str | None, dict | None, dict]:
    """(pm_slug, moneyline, mids) — all failures degrade to predict-only."""
    slug, ml = None, None
    try:
        slug = pm.get_event_slug(fixture_id)
    except Exception as exc:
        print(f"  [live] Polymarket slug lookup failed: {exc!r}")
    if slug:
        try:
            ml = pm.get_moneyline(fixture_id)
        except Exception as exc:
            print(f"  [live] Polymarket moneyline fetch failed: {exc!r}")
    mids = {}
    if ml:
        mids = {k: (ml["outcomes"].get(k) or {}).get("current_mid_yes")
                for k in ("home", "draw", "away")}
        if not any(isinstance(v, (int, float)) for v in mids.values()):
            ml = None  # market exists but no prices → predict-only
    return slug, ml, mids


# ── Shared brain: PRE_MATCH ─────────────────────────────────────────────────

def gather_prematch(fixture_id: int) -> Forecast:
    fx = Forecast(fixture_id=fixture_id, window="PRE_MATCH")

    fixture = sportmonks.get_fixture(fixture_id)
    participants = fixture.get("participants") or []
    home = next((p for p in participants if p.get("meta", {}).get("location") == "home"), {})
    away = next((p for p in participants if p.get("meta", {}).get("location") == "away"), {})
    fx.home_code = home.get("short_code") or "HOME"
    fx.away_code = away.get("short_code") or "AWAY"
    fx.home_name = home.get("name", fx.home_code)
    fx.away_name = away.get("name", fx.away_code)
    fx.fixture_name = fixture.get("name", f"{fx.home_name} vs {fx.away_name}")
    fx.kickoff = str(fixture.get("starting_at", ""))

    print(f"  [live] {fx.fixture_name}  kickoff={fx.kickoff}")

    # Market
    fx.pm_slug, fx.moneyline, fx.mids = _fetch_market(fixture_id)
    fx.market_source = "polymarket" if fx.moneyline else "none"
    if fx.moneyline:
        fx.moneyline["market_source"] = "polymarket"
    print(f"  [live] market={fx.market_source}  mids={fx.mids or 'n/a'}")

    # Research (each degrades independently)
    match_date = fx.kickoff[:10]
    have_lineups = any((p.get("meta", {}) or {}).get("position") for p in participants)
    try:
        fx.web_research = web_search.gather_research(
            fx.home_name, fx.away_name, match_date, have_confirmed_lineups=have_lineups)
    except Exception as exc:
        print(f"  [live] web research failed: {exc!r}")
        fx.web_research = {"total_results": 0, "sources": []}
    try:
        fx.reddit_bundle = reddit_sentiment.get_sentiment_bundle(fx.home_name, fx.away_name)
    except Exception as exc:
        print(f"  [live] reddit failed: {exc!r}")
        fx.reddit_bundle = {"threads_found": 0, "top_comments": []}
    try:
        fx.kalshi_ml = kalshi.get_moneyline(fx.home_name, fx.away_name)
    except Exception as exc:
        print(f"  [live] kalshi failed: {exc!r}")
        fx.kalshi_ml = {"markets_found": 0}

    # Structured digests (Sportmonks via fixture id, Supabase via names)
    ctx = fixture_bundle.build_context(
        fx.home_name, fx.away_name, fx.home_code, fx.away_code,
        sportmonks_fixture_id=fixture_id, fixture_name=fx.fixture_name)
    fx.sm_digest = ctx.get("sportmonks_digest")
    fx.sb_digest = ctx.get("supabase_digest")

    if fx.moneyline:
        try:
            fx.pm_digest_result = llm.digest_polymarket(json.dumps(fx.moneyline))
        except Exception as exc:
            print(f"  [live] polymarket digest failed: {exc!r}")
    if fx.pm_digest_result is None:
        r = _EmptyResult()
        r.parsed = {"data_availability": "no_market", "implied_win_prob": None,
                    "execution_handles": None, "market_handle": None}
        fx.pm_digest_result = r

    # The council (Scout → Analyst → Devil → Judge + grounding)
    cr = council.run_council(
        fx.fixture_name, fx.home_code, fx.away_code, fx.home_name, fx.away_name,
        fx.kickoff, fx.sm_digest, fx.sb_digest, fx.pm_digest_result.parsed,
        fx.kalshi_ml, fx.web_research, fx.reddit_bundle,
    )
    fx.cr = cr
    fx.probabilities = cr.probabilities
    fx.outcome = cr.outcome
    fx.probability = float(cr.probability)
    fx.confidence = cr.confidence
    fx.scout_flags = cr.scout_flags
    fx.grounding = cr.grounding
    fx.summary = cr.council_summary
    print(f"  [live] council: {fx.outcome} @ {fx.probability:.1%} "
          f"({fx.confidence})  probs={ {k: round(v, 3) for k, v in fx.probabilities.items()} }")
    return fx


# ── Shared brain: HT ────────────────────────────────────────────────────────

def gather_halftime(fixture_id: int, prematch_note: dict | None = None) -> Forecast:
    fx = Forecast(fixture_id=fixture_id, window="HT")

    fixture = sportmonks.get_fixture(fixture_id)
    participants = fixture.get("participants") or []
    home = next((p for p in participants if p.get("meta", {}).get("location") == "home"), {})
    away = next((p for p in participants if p.get("meta", {}).get("location") == "away"), {})
    fx.home_code = home.get("short_code") or "HOME"
    fx.away_code = away.get("short_code") or "AWAY"
    fx.home_name = home.get("name", fx.home_code)
    fx.away_name = away.get("name", fx.away_code)
    fx.fixture_name = fixture.get("name", f"{fx.home_name} vs {fx.away_name}")
    fx.kickoff = str(fixture.get("starting_at", ""))

    ht_stats_sm = {}
    try:
        ht_stats_sm = sportmonks.extract_ht_stats(fixture)
    except Exception as exc:
        print(f"  [live] HT stats extract failed: {exc!r}")
    ht_snapshot, ht_score = [], []
    try:
        ht_snapshot = supabase_client.get_ht_snapshot(fixture_id)
        ht_score = supabase_client.get_ht_score(fixture_id)
    except Exception as exc:
        print(f"  [live] HT supabase snapshot failed: {exc!r}")
    fx.ht_context = {"ht_stats_sm": ht_stats_sm, "ht_snapshot": ht_snapshot,
                     "ht_score": ht_score}

    fx.pm_slug, fx.moneyline, fx.mids = _fetch_market(fixture_id)
    fx.market_source = "polymarket" if fx.moneyline else "none"
    if fx.moneyline:
        fx.moneyline["market_source"] = "polymarket"

    if fx.moneyline:
        try:
            fx.pm_digest_result = llm.digest_polymarket(json.dumps(fx.moneyline))
        except Exception:
            pass
    if fx.pm_digest_result is None:
        r = _EmptyResult()
        r.parsed = {"data_availability": "no_market"}
        fx.pm_digest_result = r

    result = llm.ht_predict(ht_predict_input(
        fx.fixture_name, fx.home_code, fx.away_code,
        prematch_note, ht_snapshot, ht_score, ht_stats_sm))
    fx.ht_pred_result = result
    parsed = result.parsed or {}
    fx.outcome = parsed.get("outcome") or fx.home_code
    fx.probability = float(parsed.get("probability") or 0.34)
    fx.confidence = parsed.get("confidence_level", "low")
    fx.summary = parsed.get("rationale", "")

    # HT predictor emits a single (outcome, p); rebuild a 3-way distribution by
    # giving the residual mass to the other outcomes proportional to the
    # pre-match forecast (uniform if we have none).
    pre_probs = (prematch_note or {}).get("probabilities") or {}
    keys = [fx.home_code, "draw", fx.away_code]
    rest = [k for k in keys if k != fx.outcome]
    rest_pre = [max(float(pre_probs.get(k, 1.0)), 1e-6) for k in rest]
    rest_tot = sum(rest_pre)
    residual = max(0.0, 1.0 - fx.probability)
    fx.probabilities = {fx.outcome: fx.probability}
    for k, w in zip(rest, rest_pre):
        fx.probabilities[k] = residual * w / rest_tot
    print(f"  [live] HT update: {fx.outcome} @ {fx.probability:.1%} ({fx.confidence})")
    return fx


# ── Per-agent tail: policy → orders → ledger ───────────────────────────────

def act_for_agent(agent: LiveAgent, fx: Forecast, *, dry_run: bool = False) -> dict:
    """
    Run one agent's decision + execution + ledger for a shared forecast.
    Returns a summary dict for state/metrics. Never raises (logs + degrades).
    """
    profile = agent.profile
    client = ArenaClient(agent.api_key, agent.name)

    # Wallet (a dead wallet ⇒ predict-only, never crash the cycle)
    try:
        wallet = client.wallet()
    except Exception as exc:
        print(f"  [{agent.name}] wallet fetch failed: {exc!r} — predict-only")
        wallet = {"available": 0.0, "locked": 0.0, "address": None}
    available = float(wallet["available"])

    # ── Policy: shared forecast → sized picks ────────────────────────────
    skip_reasons: list[str] = []
    picks = bet_policy.select_picks(
        profile, fx.probabilities, fx.moneyline,
        fx.home_code, fx.away_code, max(available, 0.0),
        window=fx.window,
        confidence_num=confidence_to_num(fx.confidence),
        scout_flags=fx.scout_flags,
        skip_reasons=skip_reasons,
    )

    # Risk overlay (consensus / scout multipliers) on the top pick only.
    gate_info = {}
    if picks:
        top = picks[0]
        kalshi_mid = None
        try:
            from agent import _kalshi_mid_for
            kalshi_mid = _kalshi_mid_for(fx.kalshi_ml, top.code, fx.home_code, fx.away_code)
        except Exception:
            pass
        # P&L-tail agents (SAW/SURGE) opt out of confidence-based shrink: their
        # edge is skew, not conviction (STRATEGY §5). Neutralize the low/high
        # confidence multiplier for them by passing a neutral confidence label.
        gate_conf = fx.confidence if profile.apply_confidence_multiplier else "medium"
        g = gates.evaluate_gates(
            outcome=top.code, model_prob=top.our_prob, pm_mid=top.entry_price,
            kalshi_mid=kalshi_mid, scout_flags=fx.scout_flags,
            confidence=gate_conf, wallet_balance=max(available, 0.0),
            min_edge=None, scout_veto=profile.skip_on_high_scout_flag,
        )
        gate_info = {"bet_multiplier": g.bet_multiplier,
                     "market_agreement": g.market_agreement,
                     "veto_reason": g.veto_reason, "reasons": g.reasons}
        if not g.should_trade:
            skip_reasons.append(f"gates veto: {g.veto_reason or g.reasons}")
            picks = []
        elif g.bet_multiplier != 1.0:
            cap = min(profile.max_bet_usd, max(0.0, available))
            survivors = []
            for p in picks:
                scaled = round(p.stake_usd * g.bet_multiplier, 2)
                if scaled < bet_policy.MIN_ORDER_USD:
                    # Don't let the multiplier silently kill a +EV pick under
                    # the $1 floor — bump to $1 if allowed and affordable.
                    if profile.floor_to_min_order and cap >= bet_policy.MIN_ORDER_USD:
                        skip_reasons.append(
                            f"{p.slot}: gate ×{g.bet_multiplier} → ${scaled:.2f} "
                            f"floored up to ${bet_policy.MIN_ORDER_USD:.2f}")
                        scaled = bet_policy.MIN_ORDER_USD
                    else:
                        skip_reasons.append(
                            f"{p.slot}: gate ×{g.bet_multiplier} → ${scaled:.2f} "
                            f"< ${bet_policy.MIN_ORDER_USD:.2f} (dropped)")
                        continue
                p.stake_usd = min(scaled, cap)
                survivors.append(p)
            picks = survivors

    # Per-cycle wallet cap: min($5, available − 5¢) across ALL orders.
    cycle_cap = min(config.MAX_BET_USD, max(0.0, available - WALLET_BUFFER_USD))
    spent = 0.0
    capped: list = []
    for p in picks:
        room = round(cycle_cap - spent, 2)
        if room < bet_policy.MIN_ORDER_USD:
            skip_reasons.append(f"{p.slot}: cycle cap ${cycle_cap:.2f} exhausted")
            continue
        p.stake_usd = min(p.stake_usd, room)
        spent += p.stake_usd
        capped.append(p)
    picks = capped

    # ── Ledger session (this agent's own trace, own key) ─────────────────
    session = LedgerSession(fx.fixture_id, fx.fixture_name, fx.window,
                            api_key=agent.api_key, agent_tag=agent.name)
    rec_trigger = session.trigger("live-runner")
    session.planning(
        goal=f"[{profile.name}] {fx.window} cycle for {fx.fixture_name}: calibrated "
             f"prediction + at most {profile.max_bets_per_window} +EV buy-YES order(s)",
        steps=[
            "Ingest shared data bundle (Sportmonks, Polymarket, Kalshi, web, Reddit, Supabase)",
            "Run council forecast and ground it against the bookmaker anchor"
            if fx.window == "PRE_MATCH" else
            "Run Bayesian HT update from live score + xG",
            "Submit the scored prediction",
            f"Apply the {profile.name} trading policy ({profile.label})",
            "Place and poll any orders; reflect; submit this trace",
        ],
        contingencies=["Degrade to predict-only when market or wallet is unavailable"],
        upstream_ids=[rec_trigger["record_id"]],
    )

    # Tool-call records for the shared bundle (the calls genuinely happened
    # for this cycle; each agent reports them in its own trace).
    rec_sm = session.tool_call(
        name="sportmonks", endpoint=f"/v3/football/fixtures/{fx.fixture_id}",
        description="Fixture detail with predictions/odds/xG",
        input_payload={"fixture_id": fx.fixture_id},
        output_payload={"fixture": fx.fixture_name, "kickoff": fx.kickoff,
                        "sm_digest": fx.sm_digest},
        success=fx.sm_digest is not None,
        upstream_ids=[rec_trigger["record_id"]])
    rec_pm = session.tool_call(
        name="polymarket", endpoint="/proxy/polymarket-gamma+clob",
        description="Moneyline market + CLOB mids",
        input_payload={"slug": fx.pm_slug},
        output_payload={"market_source": fx.market_source, "mids": fx.mids},
        success=fx.moneyline is not None,
        upstream_ids=[rec_trigger["record_id"]])
    rec_kalshi = session.tool_call(
        name="kalshi", endpoint="/trade-api/v2/markets",
        description="Kalshi cross-market moneyline",
        input_payload={"home": fx.home_name, "away": fx.away_name},
        output_payload=fx.kalshi_ml, via="external.kalshi",
        success=(fx.kalshi_ml or {}).get("markets_found", 0) > 0,
        upstream_ids=[rec_trigger["record_id"]])
    upstream_data = [rec_sm["record_id"], rec_pm["record_id"], rec_kalshi["record_id"]]

    if fx.window == "PRE_MATCH" and fx.cr is not None:
        rec_web = session.tool_call(
            name="web_search", endpoint="search",
            description="Injury/lineup/preview research",
            input_payload={"home": fx.home_name, "away": fx.away_name},
            output_payload=fx.web_research, via="external.web",
            success=(fx.web_research or {}).get("total_results", 0) > 0,
            upstream_ids=[rec_trigger["record_id"]])
        rec_reddit = session.tool_call(
            name="reddit", endpoint="r/soccer/search.json",
            description="Crowd sentiment bundle",
            input_payload={"query": f"{fx.home_name} {fx.away_name}"},
            output_payload=fx.reddit_bundle, via="external.reddit",
            success=(fx.reddit_bundle or {}).get("threads_found", 0) > 0,
            upstream_ids=[rec_trigger["record_id"]])
        upstream_data += [rec_web["record_id"], rec_reddit["record_id"]]

        cr = fx.cr
        rec_scout = session.thinking(
            prompt_system="[SCOUT_SYS] Severity-ranked triage of news/sentiment/pulse",
            inputs=[{"record_id": rec_web["record_id"], "payload": fx.web_research},
                    {"record_id": rec_reddit["record_id"], "payload": fx.reddit_bundle},
                    {"payload": cr.social_pulse}],
            output_payload=cr.scout.parsed if cr.scout else {},
            provider=cr.scout.provider if cr.scout else "",
            model_name=cr.scout.model if cr.scout else "",
            internal_reasoning=cr.scout.thinking if cr.scout else "",
            upstream_ids=[rec_web["record_id"], rec_reddit["record_id"]])
        rec_analyst = session.thinking(
            prompt_system="[ANALYST_SYS] Market-blind base-rate forecast vs anchor",
            inputs=[{"record_id": rec_sm["record_id"], "payload": fx.sm_digest},
                    {"payload": fx.sb_digest},
                    {"record_id": rec_scout["record_id"],
                     "payload": cr.scout.parsed if cr.scout else {}}],
            output_payload=cr.analyst.parsed if cr.analyst else {},
            provider=cr.analyst.provider if cr.analyst else "",
            model_name=cr.analyst.model if cr.analyst else "",
            internal_reasoning=cr.analyst.thinking if cr.analyst else "",
            upstream_ids=[rec_sm["record_id"], rec_scout["record_id"]])
        rec_devil = session.thinking(
            prompt_system="[DEVIL_SYS] Attack the weakest assumption",
            inputs=[{"record_id": rec_analyst["record_id"],
                     "payload": cr.analyst.parsed if cr.analyst else {}}],
            output_payload=cr.devil.parsed if cr.devil else {},
            provider=cr.devil.provider if cr.devil else "",
            model_name=cr.devil.model if cr.devil else "",
            internal_reasoning=cr.devil.thinking if cr.devil else "",
            upstream_ids=[rec_analyst["record_id"]])
        rec_final = session.thinking(
            prompt_system="[JUDGE_SYS] Calibrated synthesis vs market + grounding pass",
            inputs=[{"record_id": rec_analyst["record_id"],
                     "payload": cr.analyst.parsed if cr.analyst else {}},
                    {"record_id": rec_devil["record_id"],
                     "payload": cr.devil.parsed if cr.devil else {}},
                    {"record_id": rec_pm["record_id"],
                     "payload": fx.pm_digest_result.parsed},
                    {"record_id": rec_kalshi["record_id"], "payload": fx.kalshi_ml}],
            output_payload={"probabilities": fx.probabilities, "outcome": fx.outcome,
                            "probability": fx.probability, "confidence": fx.confidence,
                            "grounding": fx.grounding, "summary": fx.summary},
            provider=cr.judge.provider if cr.judge else "",
            model_name=cr.judge.model if cr.judge else "",
            internal_reasoning=cr.judge.thinking if cr.judge else "",
            upstream_ids=[rec_analyst["record_id"], rec_devil["record_id"],
                          rec_pm["record_id"], rec_kalshi["record_id"]])
    else:
        # HT trace: one Thinking record for the Bayesian update.
        r = fx.ht_pred_result or _EmptyResult()
        rec_final = session.thinking(
            prompt_system="[HT_PREDICT_SYS] Bayesian half-time update (score + xG)",
            inputs=[{"record_id": rec_sm["record_id"], "payload": fx.ht_context},
                    {"record_id": rec_pm["record_id"],
                     "payload": fx.pm_digest_result.parsed}],
            output_payload={"probabilities": fx.probabilities, "outcome": fx.outcome,
                            "probability": fx.probability, "confidence": fx.confidence,
                            "summary": fx.summary},
            provider=getattr(r, "provider", ""),
            model_name=getattr(r, "model", ""),
            internal_reasoning=getattr(r, "thinking", ""),
            upstream_ids=upstream_data)

    # ── Prediction (scored even when we don't bet) ────────────────────────
    session.acting_prediction(
        outcome=fx.outcome, probability=fx.probability,
        upstream_ids=[rec_final["record_id"]])

    # ── Decision record ───────────────────────────────────────────────────
    rec_decision = session.thinking(
        prompt_system="[DETERMINISTIC] Profile policy + EV ranking + gates + sizing (no LLM)",
        inputs=[{"record_id": rec_final["record_id"],
                 "payload": {"probabilities": fx.probabilities,
                             "confidence": fx.confidence}},
                {"record_id": rec_pm["record_id"], "payload": fx.mids}],
        output_payload={
            "profile": profile.name,
            "profile_thresholds": {
                "min_edge_vs_fair": profile.min_edge_vs_fair,
                "min_ev_per_dollar": profile.min_ev_per_dollar,
                "min_confidence": profile.min_confidence,
                "max_entry_price": profile.max_entry_price,
                "kelly_fraction": profile.kelly_fraction,
                "max_bet_usd": profile.max_bet_usd,
                "scout_veto": profile.skip_on_high_scout_flag,
            },
            "market_source": fx.market_source,
            "wallet_available": available,
            "cycle_cap_usd": cycle_cap,
            "picks": [p.to_dict() for p in picks],
            "skip_reasons": skip_reasons,
            "gates": gate_info,
            "grounding": fx.grounding,
        },
        upstream_ids=[rec_final["record_id"], rec_pm["record_id"]])

    # ── Orders ────────────────────────────────────────────────────────────
    order_results: list[dict] = []
    for p in picks:
        if dry_run:
            order_results.append({"pick": p.to_dict(), "status": "dry_run"})
            session.acting_order(
                direction="long", outcome=p.code, size_usdc=p.stake_usd,
                limit_price=p.limit_price,
                order_payload={"fixture_id": str(fx.fixture_id), "team_code": p.code,
                               "usd_size": f"{p.stake_usd:.2f}",
                               "limit_price": p.limit_price, "dry_run": True},
                execution_status="simulated",
                upstream_ids=[rec_decision["record_id"]])
            continue
        resp = client.place_order(fx.fixture_id, p.code, p.stake_usd, p.limit_price)
        submitted_ok = isinstance(resp, dict) and "order_id" in resp
        poll = client.poll_order(resp["order_id"]) if submitted_ok else {}
        exec_status = ArenaClient.execution_status_for(
            poll.get("final_status"), poll.get("tx_hash"), submitted_ok)
        print(f"  [{agent.name}] order {p.code} ${p.stake_usd:.2f} @ ≤{p.limit_price:.2f} "
              f"→ {poll.get('final_status') or resp.get('status')}")
        order_results.append({"pick": p.to_dict(), "order_id": resp.get("order_id"),
                              "status": poll.get("final_status") or resp.get("status"),
                              "reject_reason": poll.get("reject_reason"),
                              "tx_hash": poll.get("tx_hash"),
                              "filled_usdc": poll.get("filled_usdc"),
                              "exec_status": exec_status})
        session.acting_order(
            direction="long", outcome=p.code, size_usdc=p.stake_usd,
            limit_price=p.limit_price,
            order_payload=resp.get("payload") or {},
            execution_status=exec_status,
            execution_id=resp.get("order_id") if submitted_ok else None,
            upstream_ids=[rec_decision["record_id"]])

    # ── Reflection ────────────────────────────────────────────────────────
    session.reflecting(
        inputs=[{"record_id": rec_final["record_id"],
                 "payload": {"outcome": fx.outcome, "probability": fx.probability}},
                {"record_id": rec_decision["record_id"],
                 "payload": {"n_orders": len(order_results),
                             "skip_reasons": skip_reasons}}],
        output_payload={
            "fixture": fx.fixture_name, "window": fx.window, "profile": profile.name,
            "prediction": {"outcome": fx.outcome, "probability": fx.probability,
                           "confidence": fx.confidence},
            "traded": bool(order_results),
            "orders": [{k: o.get(k) for k in ("order_id", "status", "exec_status")}
                       for o in order_results],
            "grounding_flags": (fx.grounding or {}).get("sanity_flags"),
            "what_to_improve": (
                "Compare this forecast against the de-vigged close and the result "
                "at settlement; revisit the policy bars if skip_reasons dominated."),
        },
        upstream_ids=[rec_final["record_id"], rec_decision["record_id"]])

    # ── Submit ledger ─────────────────────────────────────────────────────
    ledger_result: dict = {}
    if dry_run:
        v = session.validate()
        ledger_result = {"dry_run": True,
                         "validate": (v or {}).get("valid", "endpoint-unavailable"),
                         "records_built": session.record_count()}
    else:
        try:
            resp = session.submit()
            ledger_result = {
                "stored": len(resp.get("records") or []),
                "errors": resp.get("errors") or [],
                "status": resp.get("status"),
            }
            if ledger_result["errors"]:
                print(f"  [{agent.name}] ledger errors: "
                      f"{json.dumps(ledger_result['errors'][:3], default=str)[:300]}")
        except Exception as exc:
            print(f"  [{agent.name}] ledger submit FAILED: {exc!r}")
            ledger_result = {"error": repr(exc),
                             "records_built": session.record_count()}

    summary = {
        "prediction": {"outcome": fx.outcome, "probability": fx.probability,
                       "confidence": fx.confidence},
        "wallet_available": available,
        "n_picks": len(picks),
        "orders": order_results,
        "skip_reasons": skip_reasons,
        "ledger": ledger_result,
        "session_id": session.session_id,
    }
    metrics.log_event(
        "agent_window",
        fixture_id=fx.fixture_id, window=fx.window, fixture_name=fx.fixture_name,
        agent=agent.name, profile=profile.name,
        probabilities=fx.probabilities, confidence=fx.confidence,
        market_source=fx.market_source, mids=fx.mids,
        grounding=fx.grounding, **{k: summary[k] for k in
                                   ("prediction", "wallet_available", "orders",
                                    "skip_reasons", "ledger")})
    return summary


# ── The full window cycle ───────────────────────────────────────────────────

def run_window_cycle(fixture_id: int, window: str, agents: list[LiveAgent],
                     *, prematch_note: dict | None = None,
                     dry_run: bool = False) -> dict:
    """Shared brain once, then every agent's tail. Returns per-agent summaries."""
    if window == "PRE_MATCH":
        fx = gather_prematch(fixture_id)
    else:
        fx = gather_halftime(fixture_id, prematch_note)

    metrics.log_event(
        "forecast",
        fixture_id=fixture_id, window=window, fixture_name=fx.fixture_name,
        kickoff=fx.kickoff, home_code=fx.home_code, away_code=fx.away_code,
        probabilities=fx.probabilities, outcome=fx.outcome,
        probability=fx.probability, confidence=fx.confidence,
        market_source=fx.market_source, mids=fx.mids, grounding=fx.grounding,
        scout_flags=fx.scout_flags, summary=fx.summary)

    results: dict[str, dict] = {}
    for agent in agents:
        try:
            results[agent.name] = act_for_agent(agent, fx, dry_run=dry_run)
        except Exception as exc:
            print(f"  [{agent.name}] agent tail FAILED: {exc!r}")
            metrics.log_event("error", fixture_id=fixture_id, window=window,
                              agent=agent.name, error=repr(exc))
            results[agent.name] = {"error": repr(exc)}
    return {"fixture_name": fx.fixture_name, "window": window,
            "forecast": {"outcome": fx.outcome, "probability": fx.probability,
                         "probabilities": fx.probabilities,
                         "confidence": fx.confidence},
            "agents": results}
