"""Run the chronological walk-forward sweep + ablations and write a report.

Usage:  python3 scripts/run_walk_forward.py
Writes: storage/backtests/walk_forward_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.walk_forward import (
    load_matches, sweep, ablations, walk_forward, market_baseline,
)


def main() -> int:
    matches = load_matches()
    eval_n = sum(1 for m in matches if m.season == 2022)
    print(f"[wf] loaded {len(matches)} matches; evaluating {eval_n} (WC2022)")
    if eval_n == 0:
        print("[wf] no eval matches found — aborting")
        return 1

    ranked = sweep(matches)
    best = ranked[0]
    worst = ranked[-1]
    baseline = market_baseline(matches)
    abl = ablations(matches, best["params"])

    print("\n[wf] BEST config:")
    print(json.dumps({**best["params"], **{k: best[k] for k in ("logloss", "brier", "rps", "accuracy", "ece")}}, indent=2))
    print(f"[wf] base-rate prior baseline: logloss={baseline['logloss']} brier={baseline['brier']} acc={baseline['accuracy']}")
    print("\n[wf] ablations (logloss / brier / acc):")
    for name, r in abl.items():
        print(f"  {name:16} logloss={r['logloss']}  brier={r['brier']}  acc={r['accuracy']}")

    report = {
        "n_matches_total": len(matches),
        "n_eval": eval_n,
        "best": best,
        "worst": worst,
        "baseline_prior": baseline,
        "ablations": abl,
        "top5": ranked[:5],
        "grid_size": len(ranked),
    }
    out = ROOT / "storage" / "backtests" / "walk_forward_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[wf] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
