#!/usr/bin/env python3
"""
World Cup Agent Order Report
Pulls all fixtures from storage/ledgers + state.json + events.jsonl,
fetches live results from Sportmonks, and prints a comprehensive breakdown:
forecast probs, market mids, each agent's decision, orders placed,
actual result, and P&L analysis.
"""
import json, os, sys
from pathlib import Path

ROOT = Path("/root/worldcupagent")
STATE_FILE  = ROOT / "storage" / "live" / "state.json"
EVENTS_FILE = ROOT / "storage" / "live" / "events.jsonl"
LEDGER_DIR  = ROOT / "storage" / "ledgers"

sys.path.insert(0, str(ROOT))

# Bootstrap env from .env before importing project modules
from dotenv import dotenv_values
_env = dotenv_values(ROOT / ".env")
for k, v in _env.items():
    os.environ.setdefault(k, v)

from data.sportmonks import get_fixture

AGENTS = ["monk", "anchor", "hunter", "blitz"]

PROFILE_DESCS = {
    "monk":   "ORACLE  — min_edge_vs_fair≥10pp, conf≥0.55, max_bet=$2",
    "anchor": "KEEL    — min_edge_vs_fair≥4.5pp, min_ev≥2¢, max_bet=$4",
    "hunter": "SAW     — entry_price≤0.40 (draws+dogs), max_bet=$5",
    "blitz":  "SURGE   — min_edge_vs_fair≥2pp, scout_veto=off, max_bet=$5",
}

# ── Fetch real results from Sportmonks ───────────────────────────────────────

def fetch_result(fixture_id: int) -> dict:
    """
    Returns {"winner": "CODE"|"draw"|None, "result_info": str, "settled": bool}
    winner is None if match hasn't settled yet.
    """
    try:
        d = get_fixture(fixture_id)
    except Exception as e:
        return {"winner": None, "result_info": f"fetch error: {e}", "settled": False}

    result_info = d.get("result_info") or ""
    winner = None
    for p in d.get("participants", []):
        if p.get("meta", {}).get("winner"):
            winner = p.get("short_code")
            break
    if winner is None and "draw" in result_info.lower():
        winner = "draw"

    settled = result_info != "" and result_info is not None
    return {"winner": winner, "result_info": result_info, "settled": settled}

# ── Map outcome slot → winner code ──────────────────────────────────────────

def order_won(order: dict, winner: str | None) -> bool | None:
    """
    True if this order's outcome won, False if it lost, None if not settled.
    Orders are "YES" tokens on one outcome. They pay 1.0 if that outcome wins.
    """
    if winner is None:
        return None
    pick = order.get("pick", {})
    code = pick.get("code", "")
    # code == "draw" means the draw token
    if code == "draw":
        return winner == "draw"
    return code == winner

def order_pnl(order: dict, winner: str | None) -> float | None:
    """
    Simple P&L: stake × (1/entry_price - 1) if win, -stake if loss, None if pending.
    Binary market: YES token bought at entry_price. If it resolves YES → pays 1.0.
    Profit = stake × (1.0 / entry_price - 1)  [rough, ignores partial fills]
    Loss   = -stake
    """
    if winner is None:
        return None
    pick = order.get("pick", {})
    stake = pick.get("stake_usd", 0)
    entry = pick.get("entry_price", 0)
    won = order_won(order, winner)
    if won is None:
        return None
    if won:
        # payout = stake / entry_price; profit = payout - stake
        return round(stake * (1.0 / entry - 1), 2)
    else:
        return -round(stake, 2)

# ── Load stored data ─────────────────────────────────────────────────────────

state = json.loads(STATE_FILE.read_text())

forecasts = {}
with open(EVENTS_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            if e.get("type") == "forecast":
                forecasts[e["fixture_id"]] = e
        except Exception:
            pass

# ── Council reasoning trace from ledger files ────────────────────────────────

def get_council_trace(fixture_id, agent):
    pattern = f"pre_match_{fixture_id}_{agent}_"
    files = [f for f in os.listdir(LEDGER_DIR) if f.startswith(pattern)]
    if not files:
        return []
    fpath = LEDGER_DIR / sorted(files)[-1]
    try:
        d = json.loads(fpath.read_text())
    except Exception:
        return []
    trace = []
    for r in d.get("records", []):
        if r.get("behavior") != "Thinking":
            continue
        mi = r.get("model_invocation") or {}
        model = mi.get("model_name") or "deterministic"
        out = r.get("output_payload", "")
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except Exception:
                pass
        if isinstance(out, dict):
            snippet = {k: out[k] for k in
                       ("role", "probabilities", "confidence", "recommendation",
                        "summary", "action", "verdict") if k in out}
            trace.append((model, snippet))
        elif out:
            trace.append((model, str(out)[:200]))
    return trace

# ── Result badge ─────────────────────────────────────────────────────────────

def result_badge(won):
    if won is True:  return "WIN "
    if won is False: return "LOSS"
    return "PEND"

# ── Print report ─────────────────────────────────────────────────────────────

SEP = "═" * 80

print(SEP)
print("  WORLD CUP AGENT PORTFOLIO — ORDER, PREDICTION & RESULTS REPORT")
print(f"  {len(state['windows'])} windows across {len(forecasts)} forecasted fixtures")
print(SEP)

windows = list(state["windows"].items())
fixture_windows = {}
for wkey, wdata in windows:
    parts = wkey.split(":")
    fid = int(parts[0])
    if parts[1] == "PRE_MATCH":
        fixture_windows[fid] = wdata

# Pre-fetch all results
print("\n  Fetching results from Sportmonks...", flush=True)
results = {}
for fid in fixture_windows:
    results[fid] = fetch_result(fid)
    r = results[fid]
    w = r["winner"] or "pending"
    print(f"    {fid}: {w:10s}  {r['result_info'][:60]}")
print()

for fid, wdata in fixture_windows.items():
    fname = wdata.get("fixture_name", f"Fixture {fid}")
    status = wdata.get("status", "?")
    fc = forecasts.get(fid, {})
    probs = fc.get("probabilities", {})
    mids  = fc.get("mids", {})
    result = results[fid]
    winner = result["winner"]
    result_info = result["result_info"] or "not settled"

    print(f"\n{'─'*80}")
    print(f"  FIXTURE {fid}: {fname}  [{status.upper()}]")
    print(f"  Kickoff: {fc.get('kickoff', 'unknown')}")

    # Result line
    if winner == "draw":
        res_display = "RESULT: Draw"
    elif winner:
        res_display = f"RESULT: {winner} WON"
    else:
        res_display = "RESULT: Not settled yet"
    print(f"  {res_display}  ({result_info})")
    print(f"{'─'*80}")

    if probs:
        codes = list(probs.keys())
        home_code = codes[0]
        away_code = codes[2] if len(codes) > 2 else "?"
        print(f"\n  SHARED FORECAST (council_with_deterministic_v2):")
        for code, slot, market_key in [
            (home_code, "home", "home"),
            ("draw",    "draw", "draw"),
            (away_code, "away", "away"),
        ]:
            our_p   = probs.get(code, 0)
            mkt_p   = mids.get(market_key, "?")
            # check if prediction was correct
            correct = ""
            if winner is not None:
                if code == winner:
                    correct = " ← CORRECT"
                elif winner == "draw" and code == "draw":
                    correct = " ← CORRECT"
            print(f"    {code:8s} ({slot:4s}): our={our_p:.3f}  market={mkt_p}{correct}")
        top_pick = fc.get("outcome", "?")
        top_prob = fc.get("probability", 0)
        top_correct = " ✓" if (top_pick == winner or (winner == "draw" and top_pick == "draw")) else (" ✗" if winner else "")
        print(f"    Top pick: {top_pick} @ {top_prob:.3f}{top_correct}")

    agents_data = wdata.get("agents", {})
    if not agents_data:
        print(f"\n  [!] No agent data — run errored or was skipped")
        continue

    # Council reasoning trace
    trace = get_council_trace(fid, "monk")
    if trace:
        print(f"\n  COUNCIL REASONING TRACE (shared brain, monk ledger):")
        for model, snippet in trace:
            print(f"    [{model}]")
            if isinstance(snippet, dict):
                for k, v in snippet.items():
                    val = str(v)[:120]
                    print(f"      {k}: {val}")
            else:
                print(f"      {str(snippet)[:120]}")

    print(f"\n  PER-AGENT DECISIONS:")
    fixture_total_staked = 0.0
    fixture_total_pnl = 0.0
    fixture_orders = 0

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
        print(f"  │  Wallet: ${wallet:.2f}  |  Confidence: {conf}  |  Predicted: {outcome} @ {prob:.3f}")

        if orders:
            agent_staked = sum(o.get("pick", {}).get("stake_usd", 0) for o in orders)
            agent_pnl_list = [order_pnl(o, winner) for o in orders]
            agent_pnl = sum(x for x in agent_pnl_list if x is not None)
            fixture_total_staked += agent_staked
            fixture_total_pnl    += agent_pnl
            fixture_orders       += len(orders)

            print(f"  │  ORDERS ({len(orders)}):")
            for o, pnl in zip(orders, agent_pnl_list):
                pick   = o.get("pick", {})
                slot   = pick.get("slot", "?").upper()
                code   = pick.get("code", "?")
                stake  = pick.get("stake_usd", 0)
                entry  = pick.get("entry_price", 0)
                limit  = pick.get("limit_price", 0)
                our_p  = pick.get("our_prob", 0)
                fair_p = pick.get("fair_prob", 0)
                edge   = pick.get("edge_vs_fair", 0)
                ev     = pick.get("ev_per_dollar", 0)
                ostatus = o.get("status", "?")
                tx     = (o.get("tx_hash") or "")[:18]
                won    = order_won(o, winner)
                badge  = result_badge(won)
                pnl_str = f"${pnl:+.2f}" if pnl is not None else "pending"

                print(f"  │    [{badge}] [{slot}] {code}: ${stake:.2f} @ entry≤{entry:.4f} (limit {limit:.4f})")
                print(f"  │         our_p={our_p:.3f}  fair_p={fair_p:.4f}  edge_vs_fair={edge:+.4f}  ev/$={ev:.4f}")
                print(f"  │         P&L: {pnl_str}  |  status={ostatus}  tx={tx}…")

            net_str = f"${agent_pnl:+.2f}" if any(x is not None for x in agent_pnl_list) else "pending"
            print(f"  │  Net this fixture: {net_str} on ${agent_staked:.2f} staked")
        else:
            reason = "; ".join(skips) if skips else "no reason given"
            print(f"  │  NO ORDER — {reason}")

        print(f"  └{'─'*60}")

    print(f"\n  Fixture totals: {fixture_orders} orders, ${fixture_total_staked:.2f} staked, net P&L ${fixture_total_pnl:+.2f}")

# ── Portfolio summary ─────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  PORTFOLIO SUMMARY")
print(SEP)

all_orders_by_agent = {ag: [] for ag in AGENTS}

for wkey, wdata in windows:
    if ":PRE_MATCH" not in wkey:
        continue
    fid = int(wkey.split(":")[0])
    winner = results.get(fid, {}).get("winner")
    fname = wdata.get("fixture_name", "?")
    agents_data = wdata.get("agents", {})
    for agent in AGENTS:
        adata = agents_data.get(agent, {})
        if not adata or "error" in adata:
            continue
        for o in adata.get("orders", []):
            pick = o.get("pick", {})
            pnl = order_pnl(o, winner)
            all_orders_by_agent[agent].append({
                "fixture": fname,
                "code":  pick.get("code", "?"),
                "stake": pick.get("stake_usd", 0),
                "edge":  pick.get("edge_vs_fair", 0),
                "ev":    pick.get("ev_per_dollar", 0),
                "entry": pick.get("entry_price", 0),
                "won":   order_won(o, winner),
                "pnl":   pnl,
                "winner": winner,
            })

print(f"\n  Games errored  : 1 (Mexico vs South Africa — LedgerSession API bug, first run)")
print(f"  Games processed: {len(fixture_windows) - 1}")
print(f"  Results settled: {sum(1 for r in results.values() if r['settled'])} / {len(results)}")
print()

grand_staked   = 0.0
grand_pnl      = 0.0
grand_orders   = 0
grand_wins     = 0
grand_losses   = 0
grand_pending  = 0

for agent in AGENTS:
    orders = all_orders_by_agent[agent]
    staked  = sum(o["stake"] for o in orders)
    pnl     = sum(o["pnl"]   for o in orders if o["pnl"] is not None)
    wins    = sum(1 for o in orders if o["won"] is True)
    losses  = sum(1 for o in orders if o["won"] is False)
    pending = sum(1 for o in orders if o["won"] is None)
    n = len(orders)
    avg_edge = (sum(o["edge"] for o in orders) / n) if n else 0
    avg_ev   = (sum(o["ev"]   for o in orders) / n) if n else 0

    grand_staked  += staked
    grand_pnl     += pnl
    grand_orders  += n
    grand_wins    += wins
    grand_losses  += losses
    grand_pending += pending

    last_wallet = 0.0
    for wdata in reversed(list(state["windows"].values())):
        adata = wdata.get("agents", {}).get(agent, {})
        if adata and "wallet_available" in adata:
            last_wallet = adata["wallet_available"]
            break

    roi_str = f"{(pnl / staked * 100):+.1f}%" if staked > 0 else "—"
    pnl_str = f"${pnl:+.2f}" if n > 0 and pending < n else f"${pnl:+.2f} (+{pending} pending)"

    print(f"  {agent.upper():8s} | {n} orders | ${staked:.2f} staked | "
          f"{wins}W/{losses}L/{pending}P | {pnl_str} | ROI {roi_str} | "
          f"avg edge={avg_edge:+.4f} | wallet now ${last_wallet:.2f}")
    for o in orders:
        badge = result_badge(o["won"])
        pnl_s = f"${o['pnl']:+.2f}" if o["pnl"] is not None else "pending"
        print(f"           [{badge}] {o['fixture']:42s} {o['code']:5s} ${o['stake']:.2f}  "
              f"edge={o['edge']:+.4f}  {pnl_s}")

print()
roi_total = f"{(grand_pnl / grand_staked * 100):+.1f}%" if grand_staked > 0 else "—"
print(f"  {'TOTAL':8s} | {grand_orders} orders | ${grand_staked:.2f} staked | "
      f"{grand_wins}W/{grand_losses}L/{grand_pending}P | "
      f"net P&L ${grand_pnl:+.2f} | ROI {roi_total}")

# ── Upset / surprise analysis ─────────────────────────────────────────────────

print(f"\n{SEP}")
print("  UPSET & SURPRISE ANALYSIS")
print(SEP)
print()
print("  Fixture                          Our Top Pick  Market Fav   Actual Result  Surprise?")
print("  " + "─" * 76)

SURPRISE_THRESHOLD = 0.20  # if winner had < 20% probability, call it an upset

for fid, wdata in fixture_windows.items():
    fname = wdata.get("fixture_name", "?")
    fc = forecasts.get(fid, {})
    probs = fc.get("probabilities", {})
    mids  = fc.get("mids", {})
    result = results[fid]
    winner = result["winner"]

    if not probs or not winner:
        print(f"  {fname:35s} {'—':12s} {'—':12s} not settled     —")
        continue

    top_pick = fc.get("outcome", "?")
    top_prob = fc.get("probability", 0)
    # market favourite (highest mid)
    if mids:
        market_fav = max(mids, key=mids.get)
        market_fav_code = {"home": list(probs.keys())[0],
                           "draw": "draw",
                           "away": list(probs.keys())[2] if len(probs) > 2 else "?"}[market_fav]
    else:
        market_fav_code = "?"

    # winner probability according to us
    our_winner_prob = probs.get(winner, 0) if winner != "draw" else probs.get("draw", 0)
    surprise = "UPSET!" if our_winner_prob < SURPRISE_THRESHOLD else (
               "close"  if our_winner_prob < 0.40 else "")
    correct  = "✓" if top_pick == winner else "✗"

    print(f"  {fname:35s} {top_pick:5s}({top_prob:.2f}) {market_fav_code:5s}        "
          f"{winner:8s}  {correct} {surprise}")

# ── Order outcome analysis ─────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  ORDER OUTCOME ANALYSIS — Why orders won/lost")
print(SEP)

# Build unified order list
all_orders_flat = []
for agent in AGENTS:
    for o in all_orders_by_agent[agent]:
        all_orders_flat.append({**o, "agent": agent})

if all_orders_flat:
    settled_orders = [o for o in all_orders_flat if o["won"] is not None]
    print(f"\n  {len(settled_orders)} of {len(all_orders_flat)} orders settled")
    print()

    # Group by fixture for narrative
    by_fixture = {}
    for o in all_orders_flat:
        fx = o["fixture"]
        by_fixture.setdefault(fx, []).append(o)

    for fx, orders in by_fixture.items():
        fid_key = next((fid for fid, wd in fixture_windows.items()
                        if wd.get("fixture_name") == fx), None)
        res = results.get(fid_key, {}) if fid_key else {}
        winner = res.get("winner", "?")
        fc = forecasts.get(fid_key, {}) if fid_key else {}
        probs = fc.get("probabilities", {})

        print(f"  {fx}  →  actual: {winner}")
        for o in orders:
            badge  = result_badge(o["won"])
            pnl_s  = f"${o['pnl']:+.2f}" if o["pnl"] is not None else "pending"
            code   = o["code"]
            our_p  = probs.get(code, 0) if code != "draw" else probs.get("draw", 0)
            mkt_p  = o["entry"]
            edge   = o["edge"]
            # Narrative reason
            if o["won"] is True:
                reason = f"correctly spotted {code} underpriced (mkt {mkt_p:.3f} vs our {our_p:.3f})"
            elif o["won"] is False:
                reason = f"{code} lost; edge was real (+{edge:.3f} vs fair) but outcome didn't land"
            else:
                reason = "pending settlement"
            print(f"    [{badge}] {o['agent']:8s} {code:5s} ${o['stake']:.2f}  {pnl_s:8s}  {reason}")
        print()

print(SEP)
print("  BEHAVIORAL NOTES")
print(SEP)
print("""
  1. IDENTICAL PROBABILITIES — All 4 agents share one brain. Probability output is
     identical; only risk profiles (edge bars, size, scout veto) differ.

  2. ORDERS ≠ TOP PREDICTION — Agents bet on draws/underdogs even when their top
     pick is the favourite, because the EV engine measures edge vs the DE-VIGGED
     fair price. If the market overprices the favourite, the +EV bet is on the
     other side.
     Example — Germany vs Curacao: top pick = GER (84.7%), but orders = DRAW + CUW
     because GER market (94.4%) exceeds our estimate → GER has NEGATIVE edge.

  3. MONK (ORACLE) never fired — requires ≥10pp edge AND conf ≥ medium (0.55).
     No game cleared both bars simultaneously.

  4. BLITZ (SURGE) placed 8 orders on 6 games — the most active agent by design.
     It fires on ≥2pp edge with scout veto off.

  5. HUNTER (SAW) skipped Brazil vs Morocco despite a +4.45pp edge on MAR because
     of a high-severity scout flag; only blitz (scout_veto=False) took that bet.

  6. AUSTRALIA UPSET — biggest surprise: AUS was only 28% in our model (market 17.5%)
     yet won. All three agents that fired on AUS (anchor, hunter, blitz) won.
     The +10.41pp edge vs fair was the largest edge of any order placed.

  7. ALL HALFTIME WINDOWS SKIPPED — ht_window_not_open at check time for every game.
""")
print(SEP)

# ── Counterfactual: what if each agent had bet its top prediction? ────────────

print(f"\n{SEP}")
print("  COUNTERFACTUAL — WHAT EACH AGENT PREDICTED & WHAT WOULD HAVE HAPPENED")
print("  (If every agent had simply placed a max-profile-size bet on its top pick)")
print(SEP)

# Profile max bets (from harness/profiles.py)
PROFILE_MAX_BET = {"monk": 2.0, "anchor": 4.0, "hunter": 5.0, "blitz": 5.0}

# Summarise actual vs counterfactual per agent
cf_summary = {ag: {"staked": 0.0, "pnl": 0.0, "wins": 0, "losses": 0, "pending": 0}
              for ag in AGENTS}
actual_summary = {ag: {"staked": 0.0, "pnl": 0.0, "wins": 0, "losses": 0, "pending": 0}
                  for ag in AGENTS}

print()

for fid, wdata in fixture_windows.items():
    fname = wdata.get("fixture_name", f"Fixture {fid}")
    fc = forecasts.get(fid, {})
    probs = fc.get("probabilities", {})
    mids  = fc.get("mids", {})
    result = results[fid]
    winner = result["winner"]
    agents_data = wdata.get("agents", {})

    if not probs or not agents_data:
        continue

    # Slot → team code mapping from probs keys
    codes = list(probs.keys())   # [home_code, "draw", away_code]
    slot_to_code = {
        "home": codes[0],
        "draw": "draw",
        "away": codes[2] if len(codes) > 2 else "?",
    }
    # Market mid per code
    code_to_mid = {
        codes[0]: mids.get("home", 0),
        "draw":   mids.get("draw", 0),
        codes[2] if len(codes) > 2 else "?": mids.get("away", 0),
    }

    print(f"  {fname}  →  actual: {winner or 'pending'}")
    print(f"  {'Agent':8s}  {'Predicted':6s}  {'Our p':6s}  "
          f"{'Mkt p':6s}  {'CF stake':8s}  {'CF P&L':8s}  "
          f"{'Actual order':30s}  {'Actual P&L':10s}  Δ(CF-actual)")
    print("  " + "─" * 100)

    for agent in AGENTS:
        adata = agents_data.get(agent, {})
        if not adata or "error" in adata:
            continue

        pred    = adata.get("prediction", {})
        orders  = adata.get("orders", [])
        top_code = pred.get("outcome", "?")
        top_prob = pred.get("probability", 0)
        top_mid  = code_to_mid.get(top_code, 0)

        # Counterfactual: bet profile max on top pick at market price
        cf_stake = PROFILE_MAX_BET.get(agent, 2.0)
        cf_won = (top_code == winner) if winner else None
        if cf_won is True and top_mid > 0:
            cf_pnl = round(cf_stake * (1.0 / top_mid - 1), 2)
        elif cf_won is False:
            cf_pnl = -cf_stake
        else:
            cf_pnl = None

        cf_badge = result_badge(cf_won)
        cf_pnl_s = f"${cf_pnl:+.2f}" if cf_pnl is not None else "pending"

        # Actual order summary for this agent+fixture
        if orders:
            actual_staked = sum(o.get("pick", {}).get("stake_usd", 0) for o in orders)
            actual_pnl_list = [order_pnl(o, winner) for o in orders]
            actual_pnl = sum(x for x in actual_pnl_list if x is not None)
            actual_codes = "+".join(o.get("pick", {}).get("code", "?") for o in orders)
            actual_pnl_s = f"${actual_pnl:+.2f}"
            actual_desc = f"{actual_codes} ${actual_staked:.2f}"
        else:
            actual_pnl = 0.0
            actual_pnl_s = "$0.00 (no order)"
            actual_desc = "— no order placed"

        # Delta: counterfactual vs actual
        delta = (cf_pnl - actual_pnl) if cf_pnl is not None else None
        delta_s = f"${delta:+.2f}" if delta is not None else "pending"

        print(f"  {agent:8s}  {top_code:6s}  {top_prob:.3f}  "
              f"{top_mid:.4f}  ${cf_stake:.2f}      [{cf_badge}] {cf_pnl_s:8s}  "
              f"{actual_desc:30s}  {actual_pnl_s:10s}  {delta_s}")

        # Accumulate summaries
        if cf_pnl is not None:
            cf_summary[agent]["staked"] += cf_stake
            cf_summary[agent]["pnl"]    += cf_pnl
            if cf_won: cf_summary[agent]["wins"]   += 1
            else:      cf_summary[agent]["losses"] += 1
        else:
            cf_summary[agent]["pending"] += 1

        actual_won_count = sum(1 for o in orders if order_won(o, winner) is True)
        actual_loss_count = sum(1 for o in orders if order_won(o, winner) is False)
        actual_summary[agent]["staked"] += actual_staked if orders else 0
        actual_summary[agent]["pnl"]    += actual_pnl
        actual_summary[agent]["wins"]   += actual_won_count
        actual_summary[agent]["losses"] += actual_loss_count

    print()

# Overall counterfactual summary
print(f"{'─'*80}")
print(f"  {'':8s}  COUNTERFACTUAL TOTALS (bet top pick at market price each game)")
print(f"  {'Agent':8s}  {'CF stake':10s}  {'CF P&L':10s}  {'CF ROI':8s}  "
      f"{'CF W/L':8s}  ||  {'Actual stake':12s}  {'Actual P&L':10s}  {'Actual ROI':8s}")
print(f"  {'─'*90}")
for agent in AGENTS:
    cf  = cf_summary[agent]
    act = actual_summary[agent]
    cf_roi  = f"{(cf['pnl']  / cf['staked']  * 100):+.1f}%" if cf['staked']  > 0 else "—"
    act_roi = f"{(act['pnl'] / act['staked'] * 100):+.1f}%" if act['staked'] > 0 else "—"
    cf_wl  = f"{cf['wins']}W/{cf['losses']}L"
    print(f"  {agent:8s}  ${cf['staked']:8.2f}  ${cf['pnl']:+8.2f}  {cf_roi:8s}  "
          f"{cf_wl:8s}  ||  ${act['staked']:10.2f}  ${act['pnl']:+8.2f}  {act_roi:8s}")

print()
print("  NOTE: Counterfactual uses profile max bet at Polymarket mid price.")
print("  Actual orders used Kelly sizing vs de-vigged fair price.")
print("  A positive Δ(CF-actual) means top-pick betting would have outperformed EV betting.")
print(SEP)
