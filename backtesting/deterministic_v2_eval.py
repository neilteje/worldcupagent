"""
Deterministic v2 — 2022 World Cup evaluation & ablation runner.

Replays all 64 WC2022 fixtures (StatsBomb results + CheckBestOdds archive,
no future leakage — pre-match state only) through:

  - the MARKET baseline (de-vigged archive odds),
  - the OLD deterministic engine (models.probability pipeline),
  - the NEW deterministic v2 ensemble (models.deterministic_v2),
  - every ablated v2 component.

Writes a full report set under storage/backtests/deterministic_v2/.

Run:  python -m backtesting.deterministic_v2_eval [--limit N] [--clear-cache]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from agent.config import load_settings
from backtesting.runner import _from_historical_fixture
from backtesting.worldcup_2022 import build_worldcup_2022_history
from harness.backtest import _codes, _predict_row
from models.deterministic_v2 import EnsembleConfig, ablation_configs, predict_v2

MODEL_VERSION = "deterministic_v2.0"
OUTDIR = Path("storage/backtests/deterministic_v2")
ORDER = ("home", "draw", "away")


# ── metrics ──────────────────────────────────────────────────────────────────

def _slot_probs(probs: dict, hc: str, ac: str) -> dict:
    """Coerce a code- or slot-keyed dict to {home,draw,away}, normalized."""
    if "home" in probs and "away" in probs:
        h, d, a = probs.get("home", 0.0), probs.get("draw", 0.0), probs.get("away", 0.0)
    else:
        h, d, a = probs.get(hc, 0.0), probs.get("draw", 0.0), probs.get(ac, 0.0)
    s = (h + d + a) or 1.0
    return {"home": h / s, "draw": d / s, "away": a / s}


def evaluate(fixtures, predict_fn) -> dict:
    n = len(fixtures)
    acc = 0
    brier = ll = rps = dbrier = dll = 0.0
    fav_n = fav_hit = 0
    buckets: dict[int, list] = {}
    confusion: dict[tuple, int] = {}
    per_match = []
    for fx in fixtures:
        hc, ac = _codes(_from_historical_fixture(fx))
        p = predict_fn(fx, hc, ac)
        res = fx.result
        pick = max(ORDER, key=lambda k: p[k])
        hit = int(pick == res)
        acc += hit
        brier += sum((p[k] - (1.0 if k == res else 0.0)) ** 2 for k in ORDER)
        ll += -math.log(max(1e-9, p[res]))
        cp = co = 0.0
        for k in ORDER:
            cp += p[k]; co += 1.0 if k == res else 0.0; rps += (cp - co) ** 2
        dy = 1.0 if res == "draw" else 0.0
        dbrier += (p["draw"] - dy) ** 2
        dll += -(dy * math.log(max(1e-9, p["draw"])) + (1 - dy) * math.log(max(1e-9, 1 - p["draw"])))
        # favorite = the side the model makes most likely; "favorite acc" over non-draw picks
        if pick != "draw":
            fav_n += 1; fav_hit += hit
        conf = p[pick]
        b = min(9, int(conf * 10))
        buckets.setdefault(b, [0, 0, 0.0])
        buckets[b][0] += 1; buckets[b][1] += hit; buckets[b][2] += conf
        confusion[(pick, res)] = confusion.get((pick, res), 0) + 1
        per_match.append({
            "fixture": fx.fixture_code,
            "date": (fx.kickoff_utc or "")[:10],
            "home_team": fx.home_team,
            "away_team": fx.away_team,
            "stage": fx.stage,
            "result": res,
            "p_home": round(p["home"], 4),
            "p_draw": round(p["draw"], 4),
            "p_away": round(p["away"], 4),
            "pick": pick,
            "confidence": round(conf, 4),
            "hit": hit,
            "logloss": round(-math.log(max(1e-9, p[res])), 4),
            "brier": round(sum((p[k] - (1.0 if k == res else 0.0)) ** 2 for k in ORDER), 4),
        })
    cal = []
    cal_err = 0.0
    for b in sorted(buckets):
        cnt, h, cs = buckets[b]
        cal.append({"bucket": f"{b/10:.1f}-{(b+1)/10:.1f}", "n": cnt,
                    "avg_conf": round(cs / cnt, 4), "acc": round(h / cnt, 4)})
        cal_err += (cnt / n) * abs(cs / cnt - h / cnt)
    return {
        "n": n,
        "accuracy": round(acc / n, 4),
        "brier": round(brier / n, 4),
        "log_loss": round(ll / n, 4),
        "rps": round(rps / (2 * n), 4),
        "draw_brier": round(dbrier / n, 4),
        "draw_log_loss": round(dll / n, 4),
        "favorite_accuracy": round(fav_hit / fav_n, 4) if fav_n else None,
        "ece": round(cal_err, 4),
        "calibration": cal,
        "confusion": {f"{pk}->{rs}": c for (pk, rs), c in sorted(confusion.items())},
        "per_match": per_match,
    }


# ── predictors ─────────────────────────────────────────────────────────────--

def _market_predictor(fx, hc, ac):
    return _slot_probs(fx.market, hc, ac)


def _old_predictor(fx, hc, ac):
    row = _from_historical_fixture(fx)
    pred, _ = _predict_row(row, hc, ac, engine="deterministic")
    return _slot_probs(pred.probabilities, hc, ac)


def _is_knockout(stage: str) -> bool:
    s = (stage or "").lower()
    return bool(s) and "group" not in s


def _v2_predictor(cfg: EnsembleConfig):
    def fn(fx, hc, ac):
        out = predict_v2(fx.pre_state["home"], fx.pre_state["away"],
                         market_probs=(fx.market if cfg.use_market else None), cfg=cfg,
                         is_knockout=_is_knockout(fx.stage), match_week=fx.match_week)
        return out["probabilities"]
    return fn


def _v2_detailed_rows(fixtures, cfg: EnsembleConfig) -> list[dict]:
    """Full debug table for the v2 run: probs + expected goals + component probs."""
    rows = []
    for fx in fixtures:
        out = predict_v2(fx.pre_state["home"], fx.pre_state["away"], market_probs=fx.market, cfg=cfg,
                         is_knockout=_is_knockout(fx.stage), match_week=fx.match_week)
        p, eg = out["probabilities"], out["expected_goals"]
        comp = out["components"]
        mk = comp.get("market") or {}
        res = fx.result
        pick = out["pick"]
        rows.append({
            "fixture": fx.fixture_code, "date": (fx.kickoff_utc or "")[:10],
            "home_team": fx.home_team, "away_team": fx.away_team, "stage": fx.stage, "result": res,
            "p_home": round(p["home"], 4), "p_draw": round(p["draw"], 4), "p_away": round(p["away"], 4),
            "pick": pick, "confidence": out["confidence"], "hit": int(pick == res),
            "logloss": round(-math.log(max(1e-9, p[res])), 4),
            "brier": round(sum((p[k] - (1.0 if k == res else 0.0)) ** 2 for k in ORDER), 4),
            "lambda_home": eg["lambda_home"], "lambda_away": eg["lambda_away"],
            "exp_total": eg["expected_total"], "supremacy": eg["supremacy"],
            "rating_home": round(float(fx.pre_state["home"].get("live_rating", 0.0)), 4),
            "rating_away": round(float(fx.pre_state["away"].get("live_rating", 0.0)), 4),
            "elo_home": round(comp["elo"]["home"], 4), "elo_draw": round(comp["elo"]["draw"], 4),
            "elo_away": round(comp["elo"]["away"], 4),
            "poisson_home": round(comp["poisson"]["home"], 4), "poisson_draw": round(comp["poisson"]["draw"], 4),
            "poisson_away": round(comp["poisson"]["away"], 4),
            "market_home": round(mk.get("home", 0.0), 4), "market_draw": round(mk.get("draw", 0.0), 4),
            "market_away": round(mk.get("away", 0.0), 4),
        })
    return rows


# ── report writers ───────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def run(limit: int | None = None, clear_cache: bool = False) -> dict:
    settings = load_settings(True)
    cache_dir = settings.storage_dir / "backtests" / "cache" / "wc2022"
    if clear_cache:
        # Only clear derived prediction caches; keep the costly StatsBomb/odds archive.
        for stale in (OUTDIR,):
            if stale.exists():
                shutil.rmtree(stale)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    fixtures = build_worldcup_2022_history(cache_dir, limit=limit)
    cfg = EnsembleConfig()
    print(f"[{MODEL_VERSION}] evaluating {len(fixtures)} WC2022 fixtures")
    print(f"  config: w_elo={cfg.w_elo} w_poisson={cfg.w_poisson} w_market={cfg.w_market} "
          f"rho={cfg.rho} temp={cfg.temperature} base_shrink={cfg.base_rate_shrink}")

    market = evaluate(fixtures, _market_predictor)
    old = evaluate(fixtures, _old_predictor)
    v2 = evaluate(fixtures, _v2_predictor(cfg))

    # Ablation
    ablation = {}
    for name, acfg in ablation_configs(cfg).items():
        ablation[name] = evaluate(fixtures, _v2_predictor(acfg))

    headline = {"market_baseline": _headline(market), "old_deterministic": _headline(old),
                "new_deterministic_v2": _headline(v2)}

    # ── write outputs ──
    detailed = _v2_detailed_rows(fixtures, cfg)
    _write_csv(OUTDIR / "predictions.csv", detailed,
               ["fixture", "date", "home_team", "away_team", "stage", "result",
                "p_home", "p_draw", "p_away", "pick", "confidence", "hit", "logloss", "brier",
                "lambda_home", "lambda_away", "exp_total", "supremacy",
                "rating_home", "rating_away",
                "elo_home", "elo_draw", "elo_away",
                "poisson_home", "poisson_draw", "poisson_away",
                "market_home", "market_draw", "market_away"])
    worst = sorted(v2["per_match"], key=lambda r: r["logloss"], reverse=True)[:10]
    _write_csv(OUTDIR / "per_match_errors.csv",
               sorted(v2["per_match"], key=lambda r: r["logloss"], reverse=True),
               ["fixture", "date", "home_team", "away_team", "stage", "result",
                "pick", "confidence", "hit", "logloss", "brier"])
    _write_csv(OUTDIR / "calibration_buckets.csv", v2["calibration"],
               ["bucket", "n", "avg_conf", "acc"])

    summary = {
        "model_version": MODEL_VERSION,
        "config": v2["per_match"] and {
            "w_elo": cfg.w_elo, "w_poisson": cfg.w_poisson, "w_market": cfg.w_market,
            "rho": cfg.rho, "temperature": cfg.temperature,
            "base_rate_shrink": cfg.base_rate_shrink, "base_rate": cfg.base_rate,
            "strength": cfg.strength.__dict__,
        },
        "matches": len(fixtures),
        "result_distribution": dict(Counter(fx.result for fx in fixtures)),
        "headline": headline,
        "calibration_v2": v2["calibration"],
        "confusion_v2": v2["confusion"],
        "worst_10_v2": worst,
        "ablation": {k: _headline(v) for k, v in ablation.items()},
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTDIR / "report.md").write_text(_render_md(summary), encoding="utf-8")
    print(json.dumps(headline, indent=2))
    print(f"\nOutputs written to {OUTDIR}/")
    return summary


def _headline(m: dict) -> dict:
    return {k: m[k] for k in ("n", "accuracy", "brier", "log_loss", "rps",
                              "draw_brier", "draw_log_loss", "favorite_accuracy", "ece")}


def _render_md(s: dict) -> str:
    h = s["headline"]
    def row(name, key):
        d = h[key]
        return (f"| {name} | {d['accuracy']} | {d['brier']} | {d['log_loss']} | {d['rps']} | "
                f"{d['draw_brier']} | {d['draw_log_loss']} | {d['ece']} |")
    lines = [
        f"# Deterministic v2 — WC2022 Backtest ({s['matches']} matches)",
        "", f"Model version: `{s['model_version']}`  |  Result distribution: {s['result_distribution']}",
        "", "## Headline (lower is better except accuracy)",
        "", "| Model | Acc | Brier | LogLoss | RPS | DrawBrier | DrawLogLoss | ECE |",
        "|---|---|---|---|---|---|---|---|",
        row("Market baseline", "market_baseline"),
        row("Old deterministic", "old_deterministic"),
        row("**New deterministic v2**", "new_deterministic_v2"),
        "", "## Ablation (v2 components)",
        "", "| Config | Acc | Brier | LogLoss | RPS | DrawLogLoss | ECE |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, d in s["ablation"].items():
        lines.append(f"| {name} | {d['accuracy']} | {d['brier']} | {d['log_loss']} | "
                     f"{d['rps']} | {d['draw_log_loss']} | {d['ece']} |")
    lines += ["", "## v2 calibration buckets", "",
              "| Bucket | n | avg conf | acc |", "|---|---|---|---|"]
    for c in s["calibration_v2"]:
        lines.append(f"| {c['bucket']} | {c['n']} | {c['avg_conf']} | {c['acc']} |")
    lines += ["", "## Worst 10 v2 predictions (by log loss)", "",
              "| Match | Stage | Result | Pick | Conf | LogLoss |", "|---|---|---|---|---|---|"]
    for w in s["worst_10_v2"]:
        lines.append(f"| {w['home_team']} vs {w['away_team']} | {w['stage']} | {w['result']} | "
                     f"{w['pick']} | {w['confidence']} | {w['logloss']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m backtesting.deterministic_v2_eval")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of fixtures (default all 64).")
    ap.add_argument("--clear-cache", action="store_true", help="Clear derived v2 outputs before running.")
    args = ap.parse_args()
    run(limit=args.limit, clear_cache=args.clear_cache)


if __name__ == "__main__":
    main()
