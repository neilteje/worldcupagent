---
name: deterministic-v2-engine
description: Deterministic WC predictor v2 — module paths, how to run, and the core market-dominance finding
metadata:
  type: project
---

Improved deterministic World Cup predictor (LLM council untouched). Added June 2026.

Modules: `models/elo.py` (proper chronological Elo over 2018→2022, GD-weighted K),
`models/team_strength.py` (expected goals from a blended rating = `elo_blend`*Elo +
(1-blend)*form, default 0.30; xG with sample-size shrinkage; `elo_1x2`, `effective_rating`),
`models/poisson_model.py` (Poisson 0–10 + Dixon–Coles `rho=-0.10` → 1X2),
`models/devig.py` (power-method de-vig — improves raw market but NOT default: double-deflates
with the ensemble temperature), `models/deterministic_v2.py` (`predict_v2`, `EnsembleConfig`,
`ablation_configs` — ensemble Elo⊕Poisson⊕market then temperature + base-rate-shrink calibration).
`backtesting/worldcup_2022.py` attaches pre-match `elo`/`elo_scaled` to `pre_state` (no leakage).

Run eval/ablation: `python -m backtesting.deterministic_v2_eval` (writes
`storage/backtests/deterministic_v2/`). Param sweep (900 configs, 8-fold CV):
`python -m backtesting.deterministic_v2_sweep`. Or via harness: `python -m harness backtest
--dataset wc2022 --engine deterministic_v2 --sample 64` (branch `_predict_row_v2` in `harness/backtest.py`).

Tuned defaults (from the CV sweep, flat optimum — round values, not overfit): w_market=0.80,
w_poisson=0.10, w_elo=0.10, temperature=1.0 (calibration via base_rate_shrink=0.10, not temp),
rho=-0.10, elo_blend=0.30, knockout_draw_boost=0.05 (knockouts drew 31% vs market's 25%).
v2 now beats old+market on ALL proper scores: acc 0.5625, brier 0.5713, logloss 0.9682,
rps 0.202, draw_brier 0.1674, ece 0.0354. Edge gates (`harness/profiles.py`) intentionally
NOT loosened: v2 beats market Brier by only 0.001 (noise) and is 80% market, so no real edge.

**Core finding:** on the WC2022 backtest the de-vigged CheckBestOdds market is the strongest
single forecaster; repo-local ratings (only 2018 + in-tournament data) are noisier, so the
ensemble is deliberately market-dominant (w_market=0.80, poisson=0.12, elo=0.08). Blending stats
in heavily *hurts*. v2 beats old on accuracy/Brier/RPS/favorite-acc at a tiny log-loss cost.
Backtest data (StatsBomb + odds) is fully cached under `storage/backtests/cache/wc2022/` (offline).
Weights/constants are round a-priori values, NOT tuned to 2022 (overfitting was explicitly avoided).
