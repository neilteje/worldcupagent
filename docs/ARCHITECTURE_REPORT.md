# Architecture Overview and Capabilities

## 1. High‑Level Structure

The repository is organized around a **trading‑agent pipeline** that predicts World Cup 2026 match outcomes and optionally places play‑money bets on Polymarket.  The top‑level directories are:

| Directory | Purpose |
|-----------|---------|
| `worldcupagent/` | Core agent code, configuration, and the notebook walkthrough. |
| `backtesting/`   | Utilities for running historical simulations of the agent. |
| `betting/`       | Kelly‑criterion based bet‑size calculator and related helpers. |
| `data/`          | API wrappers for Sportmonks, Polymarket, Supabase and synthetic fixtures. |
| `ledger/`        | Structured logging (Reasoning‑Ledger) that the arena consumes for scoring. |
| `models/`        | Probabilistic models, calibration, consensus aggregation, and signal scoring. |
| `reasoning/`     | LLM‑driven reasoning steps (prompt templates, claim extraction, review writing). |
| `storage/`       | Persistent artefacts – backtest results, price history, synthetic data, etc. |
| `tests/`         | Unit‑ and integration‑tests for the above components. |
| `docs/`          | Human‑readable documentation (README, architecture, running guide, etc.). |

## 2. Core Agent Loop (`worldcupagent/`)

1. **Setup (notebook cell 2)** – Loads API keys, constants, and helper functions.
2. **Fixture discovery** – Calls the arena‑proxied Sportmonks schedule endpoint to obtain the list of matches for the WC‑2026 season.
3. **Data fetching** – For a chosen fixture the agent pulls:
   * Pre‑match signals from Sportmonks (model predictions, bookmaker odds, xG).
   * Live market data from Polymarket (mid‑prices, condition IDs, token IDs).
   * Historical team statistics from Supabase (style priors, set‑piece efficiency, etc.).
4. **LLM digestion** – Each raw payload is sent to Claude (via OpenRouter) with a strict system prompt that returns a **compact JSON digest**.  This keeps downstream prompts short and deterministic.
5. **Prediction** – A second LLM combines the Sportmonks and Supabase digests to produce the agent’s own win‑probability distribution.
6. **Strategy / Edge calculation** – The agent compares its probability to the market implied probability, computes an edge, and decides whether to trade (size, direction, limit price) based on a rule‑based bankroll‑manager.
7. **Order submission** – If a trade is warranted, a POST to `/arena/orders` is made; otherwise the run is *predict‑only*.
8. **Ledger recording** – Every step is encoded as a **Reasoning‑Ledger record** (Observing, ToolCalling, Thinking, Acting) and batch‑posted to the arena for audit and scoring.

The notebook (`worldcup‑arena‑sample‑agent.ipynb`) mirrors this flow cell‑by‑cell, making it an executable tutorial.

## 3. Backtesting Framework (`backtesting/`)

- `runner.py` orchestrates a sweep over historic fixtures, re‑playing the exact same pipeline but with **historical market snapshots** stored under `storage/price_history/`.
- Results (predictions, trades, P&L) are written to `storage/backtests/decisions/` as JSON files that can be visualized later.
- The framework supports **parameter sweeps** (e.g., different edge thresholds) via the `scheduler.py` utility.

## 4. Betting Utilities (`betting/`)

- `kelly.py` implements the Kelly‑criterion to compute optimal bet size given an edge and bankroll.
- The module also contains helper functions for converting Polymarket token IDs into USD amounts and for handling idempotent order keys.

## 5. Data Layer (`data/`)

| Module | External source | Role |
|--------|----------------|------|
| `sportmonks.py` | Sportmonks API (proxied through the arena) | Schedule, fixture details, model predictions, odds, xG. |
| `polymarket.py` | Polymarket CLOB & Gamma APIs (via arena proxy) | Live market prices, condition IDs, token IDs. |
| `supabase_client.py` | Supabase PostgREST endpoint (public read‑only) | Historical priors, catalog discovery, custom tables. |
| `synthetic_fixtures.py` | Internal generator | Creates deterministic synthetic fixtures for stress‑testing the pipeline. |
| `market_memory.py` | Local cache | Stores recent market snapshots to avoid redundant CLOB calls during backtests. |

## 6. Modeling (`models/`)

- **Bet sizing (`bet_sizing.py`)** – Translates a raw probability edge into a USD stake using Kelly.
- **Calibration (`calibration.py`)** – Fits a temperature‑scaled softmax to align LLM‑generated probabilities with observed outcomes.
- **Consensus (`consensus.py`)** – Merges multiple probability sources (Sportmonks, bookmaker consensus, Supabase priors) into a single distribution.
- **Signal scoring (`signal_scoring.py`)** – Assigns weights to individual signals (e.g., xG, set‑piece efficiency) based on historical performance.
- **Edge detection (`probability_blender.py`)** – Computes the final edge used by the strategy module.

## 7. Reasoning Pipeline (`reasoning/`)

The `reasoning/` package contains the **LLM‑centric** components:

- **Prompt library (`prompts.py`)** – Centralized system prompts for each digest, prediction, and strategy step.
- **Anthropic review (`anthropic_review.py`)** – Optional higher‑quality review of the agent’s reasoning using Claude’s “extended‑thinking” mode.
- **Claim extraction (`claim_extraction.py`)** – Parses LLM output to extract verifiable statements for audit.
- **Ledger builder (`ledger_builder.py`)** – Turns raw LLM responses into the structured ledger records used in step 8.
- **Post‑mortem (`postmortem.py`)** – Generates a human‑readable report after a run, summarizing edge, P&L, and any failures.

## 8. Storage (`storage/`)

- **Backtests** – `backtests/decisions/` holds per‑fixture decision JSON files.
- **Price history** – `price_history/` contains raw Polymarket CLOB snapshots used by the backtester.
- **Reviews** – `reviews/` stores LLM‑generated post‑mortems.
- **Synthetic data** – `synthetic/` holds generated fixtures and priors for stress testing.

## 9. Tests (`tests/`)

A comprehensive test suite validates:
- API wrappers return expected schemas (`test_anthropic_review.py`, `test_consensus.py`).
- Backtesting logic produces deterministic results (`test_backtesting.py`).
- Ledger serialization conforms to the Reasoning‑Ledger schema (`test_source_reconciliation.py`).
- Edge calculation and Kelly sizing behave correctly (`test_signal_draw_stale.py`).

## 10. Capabilities Summary

| Capability | Description |
|------------|-------------|
| **Live prediction & betting** | End‑to‑end pipeline that fetches live data, reasons with LLMs, and places demo bets on Polymarket. |
| **Predict‑only mode** | If no market is available, the agent still records a prediction for scoring. |
| **Backtesting** | Replay historic fixtures with stored market snapshots to evaluate strategy performance. |
| **Synthetic stress testing** | Generate deterministic fixtures and priors to test edge cases (e.g., missing data). |
| **Modular LLM digestion** | Each external payload is distilled into a JSON digest, keeping prompts short and reproducible. |
| **Reasoning‑Ledger audit trail** | Every action is recorded as a structured ledger entry, enabling transparent scoring by the arena. |
| **Configurable bankroll management** | Edge‑based size rules, confidence‑adjusted scaling, and Kelly‑criterion integration. |
| **Extensible data sources** | New APIs can be added under `data/` and automatically incorporated into the pipeline. |
| **Comprehensive test coverage** | Unit and integration tests ensure reliability across all components. |

---

*This report was generated to document the repository’s architecture and the purpose of each sub‑module.  It lives in `docs/ARCHITECTURE_REPORT.md` for easy reference by developers and competition participants.*