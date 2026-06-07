# Repo State

## Current status

As reviewed on 2026-06-07, the repo contains two overlapping agent implementations:

- the current package-based agent under `agent/` (primary)
- an older top-level `agent.py` flow that reflects an earlier notebook-oriented design (legacy)

For new development and normal usage, the package-based path should be considered primary.

## What appears healthy

- `python -m agent.main` is a coherent CLI with once, daemon, backtest, mode-comparison, and dry-run modes.
- The default behavior is safety-first: `DRY_RUN=true`.
- Two forecast modes: `deterministic` (default) and `llm_central` (LLM synthesizes full distribution).
- The main decision path is deterministic, context-aware, and testable.
- The package agent is council-ready: external web/LLM council output can be reconciled through bounded evidence gates without bypassing deterministic risk controls.
- Anthropic integration is bounded to health check, claim extraction, analyst, central forecast, and critique roles.
- The repo writes rich artifacts under `storage/` (decisions, reviews, ledger runs, backtests, price history).
- The automated test suite currently passes.

Test result from this review:

```text
85 passed in 42.17s
```

## What may confuse a new contributor

- `agent.py` and root `config.py` look like runnable entrypoints, but they are legacy and not the cleanest current path.
- Some data modules still carry compatibility helpers and demo fallbacks, which is useful operationally but makes the codebase look more ambiguous than it is.
- The `ARCHITECTURE_REPORT.md` was outdated (pre-package refactor) but has now been updated.

## Architecture summary

The current package agent is:

- scheduler-driven
- deterministic-first by default, with optional `llm_central` mode
- dynamic source weighting by match/market archetype
- future-compatible council reconciliation and meta-arbiter contract
- fallback-tolerant (synthetic fixtures, demo defaults)
- artifact-heavy for auditability (ledger DAG, reviews, decisions)
- optionally LLM-augmented, but not LLM-authorized in deterministic mode
- supports mode-comparison backtesting (`--compare-modes`)

Its strongest internal boundary is that forecasting and risk remain local/deterministic even when Anthropic is enabled. In `llm_central` mode, the LLM provides the primary forecast but deterministic gates still police execution. External council output is treated as bounded evidence, not order authority.

## Model inventory (21 modules)

| Module | Purpose |
|--------|---------|
| `probability.py` | Pre-match and halftime blending entry points |
| `probability_blender.py` | Source weighting for deterministic blend |
| `archetype.py` | Match/market regime classification |
| `source_reliability.py` | Dynamic source-weight adjustment by regime |
| `council_reconciliation.py` | Bounded validation of external council output |
| `meta_arbiter.py` | Final deterministic/council forecast arbitration |
| `calibration.py` | Temperature scaling, market shrinkage, draw floor |
| `edge_engine.py` | Edge detection, tiering, and classification |
| `consensus.py` | Model-bookmaker-market agreement triangle |
| `halftime.py` | Score-line luck, xG adjustments, card distortions |
| `draw_model.py` | Draw-specific adjustments and sanity flags |
| `lineup_delta.py` | Lineup change impact scoring |
| `sanity_checks.py` | Global decision audit (risk/blocking flags) |
| `bet_sizing.py` | Kelly-criterion stake and limit price calculation |
| `market_stale.py` | Lagging market detection |
| `source_reconciliation.py` | Multi-source probability reconciliation |
| `signal_scoring.py` | Signal quality/freshness/corroboration scoring |
| `llm_decision.py` | Bounded LLM analyst (risk flags only) |
| `llm_central.py` | LLM-central forecast normalization and fallback |
| `critic_policy.py` | Anthropic critic review merging |

## Known caveats from code review

- Live-data fallbacks can mask integration problems if you only test without a real `ARENA_KEY`.
- The repo still has legacy files that should be clearly marked or eventually retired.
- Backtest comparison shows both deterministic and llm_central models underperformed the market baseline on the current 50-sample synthetic set — calibration tuning is needed.
- LLM-central blocked all predictions in the latest comparison backtest; the veto/skip logic may be too aggressive.
- Council integration is contract-ready but depends on main-branch council payloads being passed in under `fixture["council"]` / `fixture["council_output"]` or synthetic data.

## Suggested next cleanup steps

- Either trim or explicitly mark legacy entrypoints in the root docs.
- Add a small `.env.example` if the team wants onboarding to be less implicit.
- Add one explicit smoke-test section for live arena validation with a real key.
- Tune calibration parameters to beat the market baseline in backtests.
- Investigate LLM-central veto behavior to allow more predictions through.
- Add settled-result postmortems that update archetype-specific source reliability.
