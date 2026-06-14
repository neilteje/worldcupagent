#!/usr/bin/env python3
"""
World Cup Agent Order Report
Pulls all fixtures from storage/ledgers + state.json + events.jsonl and
prints a comprehensive breakdown: forecast probs, market mids, each agent's
decision, orders placed, and why.
"""
import json, os, sys
from pathlib import Path

ROOT = Path("/root/worldcupagent")
STATE_FILE  = ROOT / "storage" / "live" / "state.json"
EVENTS_FILE = ROOT / "storage" / "live" / "events.jsonl"
LEDGER_DIR  = ROOT / "storage" / "ledgers"

AGENTS = ["monk", "anchor", "hunter", "blitz"]

PROFILE_DESCS = {
    "monk":   "ORACLE  — min_edge_vs_fair≥10pp, conf≥0.55, max_bet=$2  | forecast specialist",
    "anchor": "KEEL    — min_edge_vs_fair≥4.5pp, min_ev≥2¢, max_bet=$4  | disciplined EV accumulator",
    "hunter": "SAW     — min_edge_vs_fair≥3pp, entry_price≤0.40, max_bet=$5 | draws+underdogs only",
    "blitz":  "SURGE   — min_edge_vs_fair≥2pp, scout_veto=off, max_bet=$5  | event-driven aggression",
}

# ── Load data ────────────────────────────────────────────────────────────────

state = json.loads(STATE_FILE.read_text())

forecasts = {}   # fixture_id -> forecast event
events = []
with open(EVENTS_FILE) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            e = json.loads(line)
            events.append(e)
            if e.get("type") == "forecast":
                forecasts[e["fixture_id"]] = e
        except:
            pass

# ── Council reasoning from ledger files ──────────────────────────────────────

def get_council_trace(fixture_id, agent):
    """Return list of (model, output_snippet) from Thinking records."""
    pattern = f"pre_match_{fixture_id}_{agent}_"
    files = [f for f in os.listdir(LEDGER_DIR) if f.startswith(pattern)]
    if not files:
        return []
    fpath = LEDGER_DIR / sorted(files)[-1]
    try:
        d = json.loads(fpath.read_text())
    except:
        return []
    trace = []
    for r in d.get("records", []):
        if r.get("behavior") == "Thinking":
            mi = r.get("model_invocation") or {}
            model = mi.get("model_name") or "deterministic"
            out = r.get("output_payload", "")
            if isinstance(out, str):
                try: out = json.loads(out)
                except: pass
            # grab key fields
            if isinstance(out, dict):
                snippet = {}
                for k in ("role", "probabilities", "confidence", "recommendation",
                          "summary", "action", "verdict"):
                    if k in out:
                        snippet[k] = out[k]
                trace.append((model, snippet))
            elif out:
                trace.append((model, str(out)[:200]))
    return trace

# ── Print report ─────────────────────────────────────────────────────────────

SEP = "═" * 80

print(SEP)
print("  WORLD CUP AGENT PORTFOLIO — ORDER & PREDICTION REPORT")
print(f"  Generated from {len(state['windows'])} windows across {len(forecasts)} forecasted fixtures")
print(SEP)

# Walk windows in chronological order
windows = list(state["windows"].items())

# Group by fixture (only PRE_MATCH for now; HT windows are all skipped)
seen_fixtures = []
fixture_windows = {}
for wkey, wdata in windows:
    parts = wkey.split(":")
    fid = int(parts[0])
    window = parts[1]
    if window == "PRE_MATCH":
        fixture_windows[fid] = wdata

for fid, wdata in fixture_windows.items():
    fname = wdata.get("fixture_name", f"Fixture {fid}")
    status = wdata.get("status", "?")
    fc = forecasts.get(fid, {})
    probs = fc.get("probabilities", {})
    mids  = fc.get("mids", {})

    print(f"\n{'─'*80}")
    print(f"  FIXTURE {fid}: {fname}  [{status.upper()}]")
    print(f"  Kickoff: {fc.get('kickoff', 'unknown')}  |  Market source: {fc.get('market_source','?')}")
    print(f"{'─'*80}")

    if probs:
        home_code = list(probs.keys())[0]
        draw_code = "draw"
        away_code = list(probs.keys())[2] if len(probs) > 2 else "?"
        print(f"\n  SHARED FORECAST (council_with_deterministic_v2):")
        print(f"    {home_code:8s} (home): our={probs.get(home_code,0):.3f}  market={mids.get('home','?')}")
        print(f"    {'draw':8s}       : our={probs.get(draw_code,0):.3f}  market={mids.get('draw','?')}")
        print(f"    {away_code:8s} (away): our={probs.get(away_code,0):.3f}  market={mids.get('away','?')}")
        print(f"    Top pick: {fc.get('outcome','?')} @ {fc.get('probability',0):.3f}")

    agents_data = wdata.get("agents", {})
    if not agents_data:
        print(f"\n  [!] No agent data — run errored or was skipped")
        continue

    # Council reasoning (use monk's ledger as representative — all share same brain)
    trace = get_council_trace(fid, "monk")
    if trace:
        print(f"\n  COUNCIL REASONING TRACE (shared brain):")
        for model, snippet in trace:
            print(f"    [{model}]")
            if isinstance(snippet, dict):
                for k, v in snippet.items():
                    val = str(v)[:120]
                    print(f"      {k}: {val}")
            else:
                print(f"      {str(snippet)[:120]}")

    print(f"\n  PER-AGENT DECISIONS:")
    total_orders = 0
    for agent in AGENTS:
        adata = agents_data.get(agent, {})
        if not adata:
            print(f"\n  [{agent.upper()}] — no data")
            continue

        if "error" in adata:
            print(f"\n  [{agent.upper()}] ERROR: {adata['error']}")
            continue

        pred    = adata.get("prediction", {})
        orders  = adata.get("orders", [])
        skips   = adata.get("skip_reasons", [])
        wallet  = adata.get("wallet_available", 0)
        conf    = pred.get("confidence", "?")
        outcome = pred.get("outcome", "?")
        prob    = pred.get("probability", 0)

        print(f"\n  ┌─ [{agent.upper()}] {PROFILE_DESCS[agent]}")
        print(f"  │  Wallet: ${wallet:.2f}  |  Confidence: {conf}  |  Best: {outcome} @ {prob:.3f}")
        print(f"  │  Predicted: {outcome} ({conf} confidence)")

        if orders:
            print(f"  │  ORDERS PLACED ({len(orders)}):")
            for o in orders:
                pick = o.get("pick", {})
                slot = pick.get("slot","?").upper()
                code = pick.get("code","?")
                stake = pick.get("stake_usd", 0)
                entry = pick.get("entry_price", 0)
                limit = pick.get("limit_price", 0)
                our_p = pick.get("our_prob", 0)
                fair_p = pick.get("fair_prob", 0)
                edge  = pick.get("edge_vs_fair", 0)
                ev    = pick.get("ev_per_dollar", 0)
                kelly = pick.get("kelly_usd", 0)
                oid   = o.get("order_id","?")[:8]
                ostatus = o.get("status","?")
                tx    = o.get("tx_hash","?")[:18] if o.get("tx_hash") else "none"
                print(f"  │    [{slot}] {code}: ${stake:.2f} @ entry≤{entry:.4f} (limit {limit:.4f})")
                print(f"  │         our_p={our_p:.3f}  fair_p={fair_p:.4f}  edge_vs_fair={edge:+.4f}  ev/$ = {ev:.4f}")
                print(f"  │         Kelly suggested ${kelly:.2f}  |  order_id={oid}…  status={ostatus}  tx={tx}…")
                total_orders += 1
        else:
            print(f"  │  NO ORDER — {'; '.join(skips) if skips else 'no reason given'}")
        print(f"  └{'─'*60}")

    print(f"\n  Total orders fired this fixture: {total_orders}")

# ── Portfolio summary ─────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  PORTFOLIO SUMMARY")
print(SEP)

all_orders = {ag: [] for ag in AGENTS}
total_staked = {ag: 0.0 for ag in AGENTS}
total_bets   = {ag: 0   for ag in AGENTS}
no_edge_games = 0
error_games  = 0

for wkey, wdata in windows:
    if ":PRE_MATCH" not in wkey:
        continue
    agents_data = wdata.get("agents", {})
    for agent in AGENTS:
        adata = agents_data.get(agent, {})
        if not adata or "error" in adata:
            if agent == "monk" and "error" in adata:
                error_games += 1
            continue
        for o in adata.get("orders", []):
            pick = o.get("pick", {})
            stake = pick.get("stake_usd", 0)
            code  = pick.get("code","?")
            edge  = pick.get("edge_vs_fair", 0)
            ev    = pick.get("ev_per_dollar", 0)
            fname = wdata.get("fixture_name","?")
            all_orders[agent].append({
                "fixture": fname,
                "code": code,
                "stake": stake,
                "edge": edge,
                "ev": ev,
            })
            total_staked[agent] += stake
            total_bets[agent]   += 1

print(f"\n  Games errored (first run, LedgerSession API bug): 1 (Mexico vs South Africa)")
print(f"  Games processed: {len(fixture_windows) - 1}")
print()

for agent in AGENTS:
    orders = all_orders[agent]
    staked = total_staked[agent]
    n = total_bets[agent]
    avg_edge = (sum(o["edge"] for o in orders)/len(orders)) if orders else 0
    avg_ev   = (sum(o["ev"]   for o in orders)/len(orders)) if orders else 0
    last_state = list(state["windows"].values())
    wallet_final = 0
    for wdata in reversed(list(state["windows"].values())):
        adata = wdata.get("agents", {}).get(agent, {})
        if adata and "wallet_available" in adata:
            wallet_final = adata["wallet_available"]
            break
    print(f"  {agent.upper():8s}: {n} orders | ${staked:.2f} staked | "
          f"avg edge_vs_fair={avg_edge:+.4f} | avg EV/$ = {avg_ev:.4f} | "
          f"wallet now ${wallet_final:.2f}")
    for o in orders:
        print(f"           └ {o['fixture']:40s}  {o['code']:5s}  ${o['stake']:.2f}  edge={o['edge']:+.4f}")

print(f"\n{SEP}")
print("  BEHAVIORAL NOTES")
print(SEP)
print("""
  1. IDENTICAL PROBABILITIES — All 4 agents run the SAME shared brain (council +
     deterministic_v2 ensemble). The predicted probabilities are identical across
     agents every game. Differences in orders come entirely from profile thresholds.

  2. MONK (ORACLE) trades almost never by design — it requires ≥10pp edge vs the
     de-vigged fair price AND council confidence ≥ medium (0.55). In most games
     the market mid is too close to our forecast for monk to fire.

  3. BLITZ (SURGE) has the most orders: min_edge_vs_fair=2pp, no scout veto, fires
     even on low-confidence reads. It often buys draws and underdogs (thin +EV vs fair).

  4. HUNTER (SAW) only buys outcomes priced ≤ 0.40 — it skipped Brazil vs Morocco
     because of a high-severity scout flag on MAR ("away: high-severity scout flag").
     It fired hard on Australia (0.175, ≤ 0.40 threshold) and Germany draws/Curacao.

  5. ANCHOR (KEEL) fires on ≥4.5pp edge vs fair with at least 2¢ EV per dollar.
     It placed orders on Australia vs Türkiye (huge 10.41pp edge) and Germany vs Curacao.

  6. STAIR LEDGER submissions confirm the probability prediction for each agent
     window (Acting "prediction" record) — the arena PSL score is based on these.
     Every ledger file is stored in storage/ledgers/ for forensics.

  7. CONTRADICTORY ORDERS vs PREDICTION — Agents often bet on the DRAW or UNDERDOG
     even when their top predicted outcome is the favorite. This is by design:
     the EV engine ranks ALL outcomes vs de-vigged fair price; if the market
     underprices a draw/underdog relative to the de-vigged fair, that outcome has
     higher EV even though it has lower predicted probability.
     e.g. Germany vs Curacao: best prediction = GER (84.7%), but orders = DRAW + CUW
          because GER's market price (94.4%) already bakes in even more certainty
          than our model — so GER has *negative* edge vs fair.

  8. ALL WINDOWS HALFTIME SKIPPED — The HT window requires a live match state
     (ht_window_not_open). This is normal; the runner checks and skips if
     the halftime data endpoint isn't serving live data.
""")
print(SEP)
