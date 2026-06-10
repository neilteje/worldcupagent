# Architecture Overview and Capabilities

## 1. High-Level Structure

The repository is organized around a **trading-agent pipeline** that predicts World Cup 2026 match outcomes and optionally places play-money bets on Polymarket. The primary code lives under the `agent/` package, invoked via `python -m agent.main`. The top-level directories are:

| Directory | Purpose |
|-----------|---------|
| `agent/`          | Core CLI, scheduler, run cycle, and config (package-based agent). |
| `backtesting/`    | Utilities for running deterministic simulations of the agent. |
| `betting/`        | Kelly-criterion based bet-size calculator and related helpers. |
| `data/`           | API wrappers for Sportmonks, Polymarket, Supabase and synthetic fixtures. |
| `models/`         | Probabilistic models, calibration, consensus, edge detection, signal scoring, and LLM integration. |
| `reasoning/`      | LLM-driven reasoning, ledger building, review writing, and trace quality. |
| `storage/`        | Persistent artefacts – backtest results, price history, decisions, reviews, etc. |
| `tests/`          | Unit- and integration-tests for the above components. |
| `docs/`           | Human-readable documentation (README, architecture, running guide, etc.). |

Legacy top-level files (`agent.py`, `config.py`, `ledger/`) remain for historical reference but are not the active path.

## 2. Core Agent Loop (`agent/`)

1. **CLI parsing** (`agent/main.py`) – Parses flags (`--once`, `--daemon`, `--backtest`, `--dry-run`, `--decision-mode`, etc.) and loads env-backed settings.
2. **Scheduling** (`agent/scheduler.py`) – Dispatches to `run_once`, `run_daemon`, or delegates to `backtesting/runner.py`.
3. **Decision cycle** (`agent/run_cycle.py`) – The core path:
   - **Fixture discovery** – Sportmonks live lookup or synthetic fixtures.
   - **Data collection** – Fixture detail, market probabilities (Polymarket), bookmaker odds, Supabase priors, lineup data, halftime checkpoints.
   - **LLM claim extraction** (optional) – Structured claims from text sources via Anthropic.
   - **Deterministic probability modeling** – Pre-match and halftime blending with calibration, draw model, lineup delta, signal scoring, and source reconciliation.
   - **LLM analyst** (optional) – Bounded second-opinion risk flags from Anthropic.
   - **LLM-central forecast** (optional) – Full probability synthesis by Anthropic when `--decision-mode=llm_central`.
   - **Edge evaluation** – Compares model probs to market, tiers the edge, and determines `should_bet`.
   - **Risk gating** – Confidence, uncertainty, market completeness, lineup status, duplicate-order protection, dry-run, and any LLM-derived blocking flags.
   - **Bet sizing & limit price** – Kelly-criterion sizing with consensus modifier.
   - **Ledger & review** – DAG-shaped reasoning ledger and markdown/JSON review artefacts.

## 3. Backtesting Framework (`backtesting/`)

- `runner.py` orchestrates a sweep over synthetic fixtures, replaying the exact same pipeline deterministically with a $5 bankroll.
- Supports `--compare-modes` to run deterministic vs `llm_central` side-by-side and produce a comparison report.
- Results (predictions, trades, P&L, Brier scores, ROI) are written to `storage/backtests/` as JSON and markdown.
- Optional low-token Claude critique of the backtest summary via `--use-claude`.

## 4. Betting Utilities (`betting/`)

- `kelly.py` implements the Kelly-criterion to compute optimal bet size given an edge and bankroll.
- The module also contains helper functions for converting Polymarket token IDs into USD amounts and for handling idempotent order keys.

## 5. Data Layer (`data/`)

| Module | External source | Role |
|--------|----------------|------|
| `sportmonks.py` | Sportmonks API (proxied through the arena) | Schedule, fixture details, model predictions, odds, xG. |
| `polymarket.py` | Polymarket CLOB & Gamma APIs (via arena proxy) | Live market prices, condition IDs, token IDs. |
| `supabase_client.py` | Supabase PostgREST endpoint (public read-only) | Historical priors, catalog discovery, custom tables. |
| `supabase_data.py` | Supabase | Data helpers for arena aggregate tables. |
| `synthetic_fixtures.py` | Internal generator | Creates deterministic synthetic fixtures for stress-testing the pipeline. |
| `market_memory.py` | Local cache | Stores recent market snapshots to avoid redundant CLOB calls during backtests. |
| `lineup_monitor.py` | Sportmonks fixture payloads | Tracks lineup changes and missing players. |

## 6. Modeling (`models/`)

- **Bet sizing (`bet_sizing.py`)** – Translates a raw probability edge into a USD stake using Kelly; computes limit prices.
- **Calibration (`calibration.py`)** – Temperature scaling, market shrinkage, draw floor, probability normalization, and validation.
- **Consensus (`consensus.py`)** – Model-bookmaker-market agreement triangle with confidence and bet-size modifiers.
- **Signal scoring (`signal_scoring.py`)** – Assigns quality, freshness, corroboration, and impact weights to individual signals.
- **Edge detection (`edge_engine.py`)** – Finds outcomes where model probabilities disagree with Polymarket, labels edge source (model/bookmaker vs market, draw underpriced, lineup not priced in, stale market, etc.), and tiers the edge (none/soft/medium/strong).
- **Probability blending (`probability_blender.py`)** – Weighting logic for deterministic blend of bookmaker, Sportmonks, Polymarket, and Supabase sources.
- **Pre-match / halftime models (`probability.py`)** – Top-level pre-match and halftime probability blending entry points.
- **Draw model (`draw_model.py`)** – Draw-specific adjustments for low xG, small strength gap, level HT state, and market draw baselines; sanity flags for unexplained low-draw probabilities.
- **Halftime model (`halftime.py`)** – Score-line luck detection, xG-based adjustments, card distortions, and halftime confidence.
- **Lineup delta (`lineup_delta.py`)** – Compares expected vs confirmed starters and scores missing player importance.
- **Market stale (`market_stale.py`)** – Detects lagging Polymarket prices by comparing snapshots against bookmaker/signal movement.
- **Source reconciliation (`source_reconciliation.py`)** – Reconciles multiple probability sources into a coherent set.
- **Sanity checks (`sanity_checks.py`)** – Global decision audit producing risk flags and blocking flags.
- **LLM decision (`llm_decision.py`)** – Bounded LLM analyst that can add caution/veto flags but does not authorize orders.
- **LLM central (`llm_central.py`)** – Full probability synthesis by Anthropic when `--decision-mode=llm_central` is active; falls back to deterministic probs with blocking flags on failure.
- **Critic policy (`critic_policy.py`)** – Merges Anthropic critic review notes into decision artefacts without authorizing orders.

## 7. Reasoning Pipeline (`reasoning/`)

The `reasoning/` package contains the **LLM-centric** and **audit** components:

- **Prompt library (`prompts.py`)** – Centralized system prompts for each digest, prediction, and strategy step.
- **Anthropic review (`anthropic_review.py`)** – Optional health-check and critique using Claude.
- **Claim extraction (`claim_extraction.py`)** – Parses LLM output to extract verifiable statements for audit.
- **Central LLM (`central_llm.py`)** – Orchestrates the LLM-central forecast call and response parsing.
- **Ledger builder (`ledger_builder.py`)** – Constructs DAG-shaped reasoning ledger records (Observing, Planning, ToolCalling, Thinking, Acting, Reflecting) with parent links; saves locally first, then attempts arena batch submission.
- **Review writer (`review_writer.py`)** – Generates markdown / JSON run reviews with metadata, predictions, risk flags, and critique info.
- **Run report (`run_report.py`)** – Summarises a full run (metrics, artefacts).
- **Trace quality (`trace_quality.py`)** – Evaluates ledger trace completeness.
- **Post-mortem (`postmortem.py`)** – Generates a human-readable report after a run, summarizing edge, P&L, and any failures.

## 8. Storage (`storage/`)

- **Backtests** – `backtests/` holds comparison JSON and markdown files.
- **Price history** – `price_history/` contains raw Polymarket CLOB snapshots.
- **Decisions** – `decisions/` holds per-fixture/window decision JSON files.
- **Reviews** – `reviews/` stores LLM-generated run reviews (markdown + JSON).
- **Runs** – `runs/` holds local ledger DAG records.
- **Synthetic data** – `synthetic/` holds generated fixtures and priors for stress testing.

## 9. Tests (`tests/`)

A comprehensive test suite (51 tests) validates:
- API wrappers return expected schemas (`test_anthropic_review.py`, `test_consensus.py`).
- Backtesting logic produces deterministic results (`test_backtest_compare.py`, `test_arena_execution.py`).
- Ledger serialization conforms to the Reasoning-Ledger schema (`test_reasoning_trace.py`).
- Edge calculation and Kelly sizing behave correctly (`test_edge_engine.py`, `test_signal_draw_stale.py`).
- LLM decision and central forecast modules (`test_llm_decision.py`, `test_llm_central.py`, `test_llm_model_composition.py`).
- Halftime, lineup delta, probability sanity, and source reconciliation models.

## 10. Capabilities Summary

| Capability | Description |
|------------|-------------|
| **Live prediction & betting** | End-to-end pipeline that fetches live data, reasons with LLMs, and places demo bets on Polymarket. |
| **Predict-only mode** | If no market is available, the agent still records a prediction for scoring. |
| **Deterministic-first mode** | Default mode: all forecasting is local/deterministic; LLM is bounded to advisory roles. |
| **LLM-central mode** | `--decision-mode=llm_central`: Anthropic synthesizes the full probability distribution; deterministic gates still police execution. |
| **Backtesting** | Replay synthetic fixtures with stored market snapshots to evaluate strategy performance. |
| **Mode comparison** | `--compare-modes` runs deterministic vs llm_central side-by-side on the same sample. |
| **Synthetic stress testing** | Generate deterministic fixtures and priors to test edge cases (e.g., missing data, stale market, lineup shocks). |
| **Reasoning-Ledger audit trail** | Every action is recorded as a structured ledger entry, enabling transparent scoring by the arena. |
| **Configurable bankroll management** | Edge-based size rules, confidence-adjusted scaling, and Kelly-criterion integration. |
| **Extensible data sources** | New APIs can be added under `data/` and automatically incorporated into the pipeline. |
| **Comprehensive test coverage** | 51 unit and integration tests ensure reliability across all components. |

---

*This report documents the repository's architecture and the purpose of each sub-module. It lives in `docs/ARCHITECTURE_REPORT.md` for easy reference by developers and competition participants.*
