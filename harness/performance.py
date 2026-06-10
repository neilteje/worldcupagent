"""
Performance collection + plots.

Reads the session ledger, the cached predictions, and a results file, then writes:
  performance.csv   — one row per paper trade (for ad-hoc analysis / external plots)
  summary.json/.md  — per-agent P&L/ROI/win-rate + shared prediction calibration
  plots/*.png       — bankroll, P&L, cumulative P&L, and bet-count charts

Results file (`results.json`) maps fixture_code → winning slot ("home"|"draw"|
"away"). Use `results_template` to generate a blank one to fill in after matches.
"""
from __future__ import annotations
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path


def results_template(session_dir: str | Path, fixtures) -> Path:
    path = Path(session_dir) / "results.json"
    if not path.exists():
        blank = {f.fixture_code: {"result_slot": "", "home": f.home, "away": f.away,
                                  "note": "set result_slot to home|draw|away"} for f in fixtures}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blank, indent=2), encoding="utf-8")
    return path


def load_results(session_dir: str | Path) -> dict[str, str]:
    path = Path(session_dir) / "results.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for code, v in (raw or {}).items():
        slot = v.get("result_slot") if isinstance(v, dict) else v
        if slot in ("home", "draw", "away"):
            out[code] = slot
    return out


def _brier(probs_slots: dict, result_slot: str) -> float:
    return sum((probs_slots[s] - (1.0 if s == result_slot else 0.0)) ** 2
               for s in ("home", "draw", "away"))


def _log_loss(probs_slots: dict, result_slot: str) -> float:
    return -math.log(max(1e-9, min(1.0, probs_slots.get(result_slot, 0.0))))


def _pred_slots(pred: dict) -> dict:
    p = pred["probabilities"]
    return {"home": p[pred["home_code"]], "draw": p["draw"], "away": p[pred["away_code"]]}


def collect(session_dir: str | Path) -> dict:
    session_dir = Path(session_dir)
    ledger = json.loads((session_dir / "ledger.json").read_text(encoding="utf-8"))
    results = load_results(session_dir)

    # ── Per-agent trade metrics ──────────────────────────────────────────
    agents_summary = {}
    trade_rows = []
    for agent, book in ledger["agents"].items():
        trades = book["trades"]
        won = [t for t in trades if t["status"] == "won"]
        lost = [t for t in trades if t["status"] == "lost"]
        settled = won + lost
        staked = sum(t["stake"] for t in trades)
        pnl = round(sum(t["pnl"] for t in settled), 4)
        start = float(book["start_bankroll"])
        agents_summary[agent] = {
            "label": book.get("label", agent),
            "start_bankroll": start,
            "ending_bankroll": round(float(book["bankroll"]), 4),
            "n_bets": len(trades),
            "n_settled": len(settled),
            "n_won": len(won),
            "n_lost": len(lost),
            "win_rate": round(len(won) / len(settled), 4) if settled else None,
            "staked_total": round(staked, 4),
            "pnl_total": pnl,
            "roi": round(pnl / start, 4) if start else None,
            "avg_edge_vs_fair": round(sum(t["edge_vs_fair"] for t in trades) / len(trades), 4) if trades else None,
            "avg_ev_per_dollar": round(sum(t["ev_per_dollar"] for t in trades) / len(trades), 4) if trades else None,
        }
        for t in trades:
            trade_rows.append({**t, "result_slot": results.get(t["fixture_code"], "")})

    # ── Shared prediction calibration (same forecast for all agents) ──────
    pred_dir = session_dir / "predictions"
    pred_metrics = {"windows": {}, "overall": {}}
    all_b, all_l, all_hit, all_n = [], [], 0, 0
    if pred_dir.exists():
        bywin: dict[str, list] = {}
        for pf in sorted(pred_dir.glob("*.json")):
            pred = json.loads(pf.read_text(encoding="utf-8"))
            res = results.get(pred["fixture_code"])
            if not res:
                continue
            slots = _pred_slots(pred)
            b, l = _brier(slots, res), _log_loss(slots, res)
            hit = 1 if max(slots, key=slots.get) == res else 0
            bywin.setdefault(pred["window"], []).append((b, l, hit))
            all_b.append(b); all_l.append(l); all_hit += hit; all_n += 1
        for win, vals in bywin.items():
            n = len(vals)
            pred_metrics["windows"][win] = {
                "n": n,
                "brier": round(sum(v[0] for v in vals) / n, 4),
                "log_loss": round(sum(v[1] for v in vals) / n, 4),
                "accuracy": round(sum(v[2] for v in vals) / n, 4),
            }
        if all_n:
            pred_metrics["overall"] = {
                "n": all_n,
                "brier": round(sum(all_b) / all_n, 4),
                "log_loss": round(sum(all_l) / all_n, 4),
                "accuracy": round(all_hit / all_n, 4),
            }

    return {
        "session_dir": str(session_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_known": len(results),
        "agents": agents_summary,
        "predictions": pred_metrics,
        "_trade_rows": trade_rows,
    }


def write_reports(session_dir: str | Path) -> dict:
    session_dir = Path(session_dir)
    summary = collect(session_dir)
    trade_rows = summary.pop("_trade_rows")

    # CSV of every trade
    csv_path = session_dir / "performance.csv"
    if trade_rows:
        cols = ["agent", "fixture_code", "window", "outcome", "slot", "stake",
                "entry_price", "our_prob", "fair_prob", "edge_vs_fair",
                "ev_per_dollar", "market_source", "status", "pnl", "result_slot", "ts"]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(trade_rows)

    (session_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (session_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    plots = make_plots(session_dir, summary, trade_rows)
    return {"summary": summary, "csv": str(csv_path), "plots": plots}


def _markdown(summary: dict) -> str:
    lines = ["# Harness Performance", "",
             f"- Generated: {summary['generated_at']}",
             f"- Results known: {summary['results_known']} fixtures", "",
             "## Agents", "",
             "| Agent | Label | Bets | Won | Win% | Staked | P&L | ROI | End $ |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, a in summary["agents"].items():
        wr = f"{a['win_rate']*100:.0f}%" if a["win_rate"] is not None else "—"
        roi = f"{a['roi']*100:+.1f}%" if a["roi"] is not None else "—"
        lines.append(f"| {name} | {a['label']} | {a['n_bets']} | {a['n_won']} | {wr} "
                     f"| ${a['staked_total']:.2f} | ${a['pnl_total']:+.2f} | {roi} "
                     f"| ${a['ending_bankroll']:.2f} |")
    pm = summary["predictions"]
    if pm.get("overall"):
        o = pm["overall"]
        lines += ["", "## Shared prediction calibration", "",
                  f"- Overall: Brier {o['brier']}, log-loss {o['log_loss']}, "
                  f"accuracy {o['accuracy']*100:.0f}% (n={o['n']})"]
        for win, w in pm.get("windows", {}).items():
            lines.append(f"- {win}: Brier {w['brier']}, accuracy {w['accuracy']*100:.0f}% (n={w['n']})")
    return "\n".join(lines) + "\n"


def make_plots(session_dir: str | Path, summary: dict, trade_rows: list[dict]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    session_dir = Path(session_dir)
    out_dir = session_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    agents = list(summary["agents"].keys())
    if not agents:
        return []
    labels = [f"{n}\n({summary['agents'][n]['label']})" for n in agents]
    saved = []

    # 1) Start vs ending bankroll
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(agents))
    ax.bar([i - 0.2 for i in x], [summary["agents"][n]["start_bankroll"] for n in agents],
           width=0.4, label="start", color="#9fb3c8")
    ax.bar([i + 0.2 for i in x], [summary["agents"][n]["ending_bankroll"] for n in agents],
           width=0.4, label="end", color="#2e7d32")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_ylabel("Bankroll ($)")
    ax.set_title("Bankroll: start vs end"); ax.legend()
    p = out_dir / "bankroll.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    saved.append(str(p))

    # 2) P&L by agent
    fig, ax = plt.subplots(figsize=(7, 4))
    pnls = [summary["agents"][n]["pnl_total"] for n in agents]
    ax.bar(labels, pnls, color=["#2e7d32" if v >= 0 else "#c62828" for v in pnls])
    ax.axhline(0, color="black", linewidth=0.8); ax.set_ylabel("P&L ($)")
    ax.set_title("Net P&L by agent")
    p = out_dir / "pnl.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    saved.append(str(p))

    # 3) Cumulative P&L over settled trades (in time order)
    settled = [t for t in trade_rows if t["status"] in ("won", "lost")]
    if settled:
        settled.sort(key=lambda t: t["ts"])
        fig, ax = plt.subplots(figsize=(8, 4))
        for n in agents:
            running, cum = 0.0, []
            for t in settled:
                if t["agent"] == n:
                    running += t["pnl"]; cum.append(running)
            if cum:
                ax.plot(range(1, len(cum) + 1), cum, marker="o", label=n)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("settled bet #"); ax.set_ylabel("cumulative P&L ($)")
        ax.set_title("Cumulative P&L"); ax.legend()
        p = out_dir / "cumulative_pnl.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        saved.append(str(p))

    # 4) Bets vs wins
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - 0.2 for i in x], [summary["agents"][n]["n_bets"] for n in agents],
           width=0.4, label="bets", color="#9fb3c8")
    ax.bar([i + 0.2 for i in x], [summary["agents"][n]["n_won"] for n in agents],
           width=0.4, label="won", color="#2e7d32")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_ylabel("count")
    ax.set_title("Bets vs wins"); ax.legend()
    p = out_dir / "bets.png"; fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    saved.append(str(p))

    return saved
