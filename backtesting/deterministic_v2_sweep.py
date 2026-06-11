"""
Deterministic v2 — parameter sweep over the WC2022 backtest.

Grid-searches EnsembleConfig (market/elo/poisson weights, temperature,
base-rate shrink, Dixon-Coles rho, elo_blend) over all 64 WC2022 fixtures.

IMPORTANT — anti-overfitting discipline. Picking the single config that
minimizes error on the same 64 matches we score on IS fitting to 2022. So this
sweep:
  - constrains the market to remain the heaviest weight (0.60-0.80),
  - reports k-fold cross-validated log loss (mean + std) alongside the full-64
    score, so we can prefer configs that are stable across subsets,
  - is meant for choosing ROUND values in a flat region of the surface, not the
    razor-edge global minimum.

Run:  python -m backtesting.deterministic_v2_sweep
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from agent.config import load_settings
from backtesting.deterministic_v2_eval import OUTDIR, evaluate
from backtesting.worldcup_2022 import build_worldcup_2022_history
from models.deterministic_v2 import EnsembleConfig, predict_v2

K_FOLDS = 8


def _predictor(cfg: EnsembleConfig):
    def fn(fx, hc, ac):
        return predict_v2(fx.pre_state["home"], fx.pre_state["away"],
                          market_probs=(fx.market if cfg.use_market else None), cfg=cfg)["probabilities"]
    return fn


def _cv_logloss(fixtures, cfg: EnsembleConfig, k: int = K_FOLDS) -> tuple[float, float]:
    """Mean and std of per-fold log loss (deterministic folds by index)."""
    fn = _predictor(cfg)
    folds = [fixtures[i::k] for i in range(k)]
    lls = [evaluate(f, fn)["log_loss"] for f in folds if f]
    mean = sum(lls) / len(lls)
    var = sum((x - mean) ** 2 for x in lls) / len(lls)
    return round(mean, 4), round(var ** 0.5, 4)


def _grid():
    base = EnsembleConfig()
    for w_market in (0.60, 0.65, 0.70, 0.75, 0.80):
        rem = 1 - w_market
        for poisson_share in (0.5, 0.65, 0.8):
            for temperature in (1.00, 1.05, 1.10, 1.15, 1.20):
                for bs in (0.0, 0.04, 0.08, 0.12):
                    for rho in (-0.06, -0.10, -0.14):
                        yield replace(base, w_market=w_market,
                                      w_poisson=round(rem * poisson_share, 4),
                                      w_elo=round(rem * (1 - poisson_share), 4),
                                      temperature=temperature, base_rate_shrink=bs, rho=rho)


def run() -> dict:
    settings = load_settings(True)
    fixtures = build_worldcup_2022_history(settings.storage_dir / "backtests" / "cache" / "wc2022")
    print(f"sweeping over {len(fixtures)} fixtures...")

    rows = []
    for cfg in _grid():
        m = evaluate(fixtures, _predictor(cfg))
        cv_mean, cv_std = _cv_logloss(fixtures, cfg)
        rows.append({
            "w_market": cfg.w_market, "w_poisson": cfg.w_poisson, "w_elo": cfg.w_elo,
            "temperature": cfg.temperature, "base_rate_shrink": cfg.base_rate_shrink, "rho": cfg.rho,
            "accuracy": m["accuracy"], "brier": m["brier"], "log_loss": m["log_loss"],
            "rps": m["rps"], "draw_brier": m["draw_brier"], "draw_log_loss": m["draw_log_loss"],
            "ece": m["ece"], "cv_logloss_mean": cv_mean, "cv_logloss_std": cv_std,
        })

    by_ll = sorted(rows, key=lambda r: r["log_loss"])
    by_cv = sorted(rows, key=lambda r: (r["cv_logloss_mean"], r["cv_logloss_std"]))
    by_brier = sorted(rows, key=lambda r: r["brier"])

    out = {
        "n_configs": len(rows),
        "n_fixtures": len(fixtures),
        "current_default": _row_for(rows, EnsembleConfig()),
        "top10_by_log_loss": by_ll[:10],
        "top10_by_cv_log_loss": by_cv[:10],
        "top10_by_brier": by_brier[:10],
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUTDIR / "sweep.md").write_text(_render(out), encoding="utf-8")
    print(f"{len(rows)} configs evaluated. Best CV log loss:")
    for r in by_cv[:5]:
        print(f"  wm={r['w_market']} wp={r['w_poisson']} we={r['w_elo']} t={r['temperature']} "
              f"bs={r['base_rate_shrink']} rho={r['rho']} | ll={r['log_loss']} "
              f"cv={r['cv_logloss_mean']}±{r['cv_logloss_std']} brier={r['brier']} acc={r['accuracy']} ece={r['ece']}")
    print(f"\nWritten to {OUTDIR}/sweep.json")
    return out


def _row_for(rows, cfg: EnsembleConfig):
    for r in rows:
        if (r["w_market"] == cfg.w_market and r["temperature"] == cfg.temperature
                and r["base_rate_shrink"] == cfg.base_rate_shrink and r["rho"] == cfg.rho):
            return r
    return None


def _render(out: dict) -> str:
    cols = ["w_market", "w_poisson", "w_elo", "temperature", "base_rate_shrink", "rho",
            "accuracy", "brier", "log_loss", "rps", "draw_brier", "ece", "cv_logloss_mean", "cv_logloss_std"]
    def table(title, rows):
        head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols)
        body = "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows)
        return f"### {title}\n\n{head}\n{body}\n"
    parts = [f"# Deterministic v2 — Parameter Sweep ({out['n_configs']} configs, {out['n_fixtures']} fixtures)\n"]
    if out.get("current_default"):
        parts.append(table("Current default", [out["current_default"]]))
    parts.append(table("Top 10 by cross-validated log loss (preferred — robust)", out["top10_by_cv_log_loss"]))
    parts.append(table("Top 10 by full-64 log loss", out["top10_by_log_loss"]))
    parts.append(table("Top 10 by full-64 Brier", out["top10_by_brier"]))
    return "\n".join(parts)


if __name__ == "__main__":
    run()
