"""
Retrospective evaluation — what did the four agents actually do, and was the
brain any good?

Reads storage/live/events.jsonl + state.json and produces:
  - per-agent activity: windows, predictions, orders, fills, stake deployed
  - realized P&L per agent (entry price vs settled winner, paper-broker math;
    cross-checked against wallet snapshots when available)
  - forecast quality: Brier score of our distribution vs the de-vigged market
    baseline (STRATEGY.md measurement M-1), per window and overall
  - anchor-divergence outcomes (M-2): when we diverged from the market, who won

Writes storage/live/report/summary.md and prints the headline table.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

from live.metrics import read_events
from live.state import LiveState

REPORT_DIR = Path(__file__).resolve().parent.parent / "storage" / "live" / "report"


def _devig(mids: dict) -> dict | None:
    vals = {k: v for k, v in (mids or {}).items() if isinstance(v, (int, float))}
    if len(vals) != 3:
        return None
    tot = sum(vals.values())
    if tot <= 0:
        return None
    return {k: v / tot for k, v in vals.items()}


def _brier(probs: dict, winner_key: str) -> float | None:
    """Multiclass Brier over the 3 outcomes; winner_key must be a key of probs."""
    if not probs or winner_key not in probs:
        return None
    return sum((float(p) - (1.0 if k == winner_key else 0.0)) ** 2
               for k, p in probs.items())


def _winner_key(probs: dict, slot: str | None, code: str | None,
                home_code: str | None, away_code: str | None) -> str | None:
    """Map a settlement (slot/code) onto the forecast's probability keys."""
    if code and code in (probs or {}):
        return code
    if str(code).lower() == "draw" or slot == "draw":
        return "draw"
    if slot == "home" and home_code in (probs or {}):
        return home_code
    if slot == "away" and away_code in (probs or {}):
        return away_code
    return None


def _execution(order: dict) -> dict:
    return order.get("execution") or {}


def _actual_stake(order: dict) -> float:
    execution = _execution(order)
    filled = execution.get("filled_notional_usdc")
    if isinstance(filled, (int, float)) and filled > 0:
        return float(filled)
    pick = order.get("pick") or {}
    return float(pick.get("stake_usd") or 0.0)


def _actual_entry_price(order: dict) -> float:
    execution = _execution(order)
    avg = execution.get("actual_average_fill_price")
    if isinstance(avg, (int, float)) and 0 < avg < 1:
        return float(avg)
    pick = order.get("pick") or {}
    return float(pick.get("entry_price") or 0.0)


def _fees(order: dict) -> float:
    value = _execution(order).get("fees_usdc")
    return float(value) if isinstance(value, (int, float)) else 0.0


def build_report() -> str:
    events = read_events()
    state = LiveState()

    forecasts = [e for e in events if e["type"] == "forecast"]
    coverage_events = [e for e in events if e["type"] == "signal_coverage"]
    agent_windows = [e for e in events if e["type"] == "agent_window"]
    settlements = {str(e["fixture_id"]): e for e in events if e["type"] == "settlement"}
    agent_settle = [e for e in events if e["type"] == "agent_settlement"]
    errors = [e for e in events if e["type"] == "error"]
    version_keys = {
        (
            e.get("strategy_version"),
            e.get("forecast_pipeline_version"),
            e.get("model_version"),
            e.get("profile_configuration_hash"),
        )
        for e in events
        if e.get("strategy_version") or e.get("model_version")
    }
    if len(version_keys) > 1:
        latest = sorted(version_keys, key=str)[-1]
        events = [
            e for e in events
            if not (e.get("strategy_version") or e.get("model_version"))
            or (
                e.get("strategy_version"),
                e.get("forecast_pipeline_version"),
                e.get("model_version"),
                e.get("profile_configuration_hash"),
            ) == latest
        ]
        forecasts = [e for e in events if e["type"] == "forecast"]
        coverage_events = [e for e in events if e["type"] == "signal_coverage"]
        agent_windows = [e for e in events if e["type"] == "agent_window"]
        agent_settle = [e for e in events if e["type"] == "agent_settlement"]
        errors = [e for e in events if e["type"] == "error"]

    # ── forecast quality vs market (M-1, M-2) ─────────────────────────────
    briers_us, briers_mkt, divergences = [], [], []
    for f in forecasts:
        s = settlements.get(str(f["fixture_id"]))
        if not s:
            continue
        probs = f.get("probabilities") or {}
        wkey = _winner_key(probs, s.get("winner_slot"), s.get("winner_code"),
                           f.get("home_code"), f.get("away_code"))
        if not wkey:
            continue
        b_us = _brier(probs, wkey)
        slot_probs = _devig(f.get("mids") or {})
        b_mkt = None
        if slot_probs:
            slot_map = {"home": f.get("home_code"), "draw": "draw",
                        "away": f.get("away_code")}
            mkt_probs = {slot_map[k]: v for k, v in slot_probs.items() if slot_map.get(k)}
            b_mkt = _brier(mkt_probs, wkey)
            # divergence audit: our top pick vs market top pick
            our_top = max(probs, key=probs.get) if probs else None
            mkt_top = max(mkt_probs, key=mkt_probs.get) if mkt_probs else None
            if our_top and mkt_top and our_top != mkt_top:
                divergences.append({"fixture": f.get("fixture_name"),
                                    "window": f.get("window"),
                                    "ours": our_top, "market": mkt_top,
                                    "winner": wkey,
                                    "we_won": our_top == wkey,
                                    "market_won": mkt_top == wkey})
        if b_us is not None:
            briers_us.append(b_us)
        if b_mkt is not None:
            briers_mkt.append(b_mkt)

    # ── per-agent activity + realized P&L ─────────────────────────────────
    agents: dict[str, dict] = defaultdict(lambda: {
        "windows": 0, "predictions": 0, "orders": 0, "confirmed": 0,
        "stake": 0.0, "pnl": 0.0, "wins": 0, "losses": 0, "open": 0,
        "skips": defaultdict(int), "wallet_last": None,
        "recommendations": 0, "abstentions": 0, "duplicate_recommendations": 0,
        "rejected_recommendations": 0, "blitz_draw_candidates_removed": 0})
    for e in agent_windows:
        a = agents[e["agent"]]
        a["windows"] += 1
        if (e.get("prediction") or {}).get("outcome"):
            a["predictions"] += 1
        a["recommendations"] += int(e.get("recommendations") or 0)
        a["abstentions"] += int(e.get("abstentions") or 0)
        a["duplicate_recommendations"] += int(e.get("duplicate_recommendations") or 0)
        a["rejected_recommendations"] += int(e.get("rejected_recommendations") or 0)
        a["blitz_draw_candidates_removed"] += int(e.get("blitz_draw_candidates_removed") or 0)
        for r in (e.get("skip_reasons") or [])[:1]:
            a["skips"][r.split(":")[0][:48]] += 1
        s = settlements.get(str(e["fixture_id"]))
        winner = s.get("winner_code") if s else None
        wslot = s.get("winner_slot") if s else None
        for o in e.get("orders") or []:
            pick = o.get("pick") or {}
            a["orders"] += 1
            stake = _actual_stake(o)
            a["stake"] += stake
            if o.get("exec_status") == "confirmed":
                a["confirmed"] += 1
            if o.get("status") == "dry_run" or o.get("exec_status") != "confirmed":
                continue
            if s is None:
                a["open"] += 1
                continue
            price = _actual_entry_price(o)
            won = (str(pick.get("code")) == str(winner)) or (pick.get("slot") == wslot)
            if won and 0 < price < 1:
                a["pnl"] += stake * (1.0 / price - 1.0) - _fees(o)
                a["wins"] += 1
            else:
                a["pnl"] -= stake + _fees(o)
                a["losses"] += 1
    for e in agent_settle:
        if e.get("wallet"):
            agents[e["agent"]]["wallet_last"] = e["wallet"].get("available")

    # ── render ────────────────────────────────────────────────────────────
    L: list[str] = []
    L.append("# Live run — retrospective report\n")
    if version_keys:
        L.append("Version scope: aggregates use one compatible strategy, forecast "
                 "pipeline, model, and profile hash; incompatible historical runs "
                 "are not combined.\n")
    st = state.summary()
    L.append(f"Windows processed: {st['windows_total']}  ({st['by_status']})  "
             f"| fixtures settled: {st['settled']}  | errors logged: {len(errors)}\n")

    L.append("## Per-agent scorecard\n")
    L.append("| agent | windows | predictions | orders | confirmed | stake $ | "
             "realized P&L $ | return_on_staked_capital | W-L | open | wallet $ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, a in sorted(agents.items()):
        wallet = f"{a['wallet_last']:.2f}" if a["wallet_last"] is not None else "—"
        rosc = f"{(a['pnl'] / a['stake'] * 100):+.1f}%" if a["stake"] else "—"
        L.append(f"| {name} | {a['windows']} | {a['predictions']} | {a['orders']} | "
                 f"{a['confirmed']} | {a['stake']:.2f} | {a['pnl']:+.2f} | "
                 f"{rosc} | {a['wins']}-{a['losses']} | {a['open']} | {wallet} |")
    L.append("")

    L.append("## Forecast quality (Brier, lower is better)\n")
    if briers_us:
        L.append(f"- council forecast : **{sum(briers_us)/len(briers_us):.4f}**  "
                 f"(n={len(briers_us)} settled windows)")
    if briers_mkt:
        L.append(f"- de-vigged market : **{sum(briers_mkt)/len(briers_mkt):.4f}**  "
                 f"(n={len(briers_mkt)})")
    if briers_us and briers_mkt:
        edge = sum(briers_mkt)/len(briers_mkt) - sum(briers_us)/len(briers_us)
        L.append(f"- council − market : **{edge:+.4f}** "
                 f"({'council ahead' if edge > 0 else 'market ahead'}; "
                 f"pivot trigger fires at council worse by 0.02 over 16+ windows)")
    if not briers_us:
        L.append("- no settled forecasts yet")
    L.append("")

    if divergences:
        L.append("## Anchor divergences (we picked differently from the market)\n")
        we = sum(1 for d in divergences if d["we_won"])
        mk = sum(1 for d in divergences if d["market_won"])
        L.append(f"{len(divergences)} divergent windows — we were right {we}×, "
                 f"the market {mk}×, neither {len(divergences)-we-mk}×.\n")
        for d in divergences[:20]:
            L.append(f"- {d['fixture']} {d['window']}: ours={d['ours']} "
                     f"market={d['market']} → winner={d['winner']}")
        L.append("")

    # ── coordination: structured recommendations + portfolio dedup ────────
    coord = {n: a for n, a in agents.items()
             if any(a[k] for k in ("recommendations", "abstentions",
                                    "duplicate_recommendations",
                                    "rejected_recommendations",
                                    "blitz_draw_candidates_removed"))}
    if coord:
        L.append("## Coordination — recommendations & portfolio dedup\n")
        L.append("| agent | recommendations | abstentions | duplicate recs | "
                 "duplicate positions prevented | other rejections | "
                 "BLITZ draw candidates removed |")
        L.append("|---|---|---|---|---|---|---|")
        for name, a in sorted(coord.items()):
            dups = a["duplicate_recommendations"]
            other = max(0, a["rejected_recommendations"] - dups - a["abstentions"])
            L.append(f"| {name} | {a['recommendations']} | {a['abstentions']} | "
                     f"{dups} | {dups} | {other} | "
                     f"{a['blitz_draw_candidates_removed']} |")
        L.append("")

    # ── signal coverage: which inputs actually fed the council ────────────
    if coverage_events:
        sources = ("sportmonks", "supabase", "bzzoiro", "web_search", "reddit",
                   "grok_pulse", "polymarket")
        totals = {s: sum(1 for e in coverage_events if e.get(s)) for s in sources}
        n = len(coverage_events)
        L.append("## Signal coverage — what actually fed the council\n")
        L.append(f"Across {n} council windows, share of windows where each source "
                 f"carried content (BZZOIRO should be one of several, not the spine):\n")
        L.append("| source | windows used | usage rate |")
        L.append("|---|---|---|")
        for s in sources:
            L.append(f"| {s} | {totals[s]} | {totals[s]/n*100:.0f}% |")
        L.append("")

    L.append("## Top skip reasons per agent\n")
    for name, a in sorted(agents.items()):
        if a["skips"]:
            top = sorted(a["skips"].items(), key=lambda kv: -kv[1])[:3]
            L.append(f"- **{name}**: " + "; ".join(f"{k} ×{v}" for k, v in top))
    L.append("")

    if errors:
        L.append("## Recent errors\n")
        for e in errors[-10:]:
            L.append(f"- {e.get('ts')} [{e.get('agent') or e.get('scope') or '?'}] "
                     f"{str(e.get('error'))[:140]}")
        L.append("")

    text = "\n".join(L)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.md").write_text(text, encoding="utf-8")
    (REPORT_DIR / "agents.json").write_text(
        json.dumps({k: {**v, "skips": dict(v["skips"])} for k, v in agents.items()},
                   indent=2, default=str), encoding="utf-8")
    return text
