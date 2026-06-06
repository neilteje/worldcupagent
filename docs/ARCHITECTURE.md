# Architecture

## Current execution model

The maintained agent is the package under `agent/`, with `python -m agent.main` as the CLI entrypoint.

The agent now supports two forecast modes:

- `deterministic`: current deterministic-first blend with optional LLM support roles
- `llm_central`: the LLM synthesizes the full feature bundle into the primary forecast, while deterministic gates still police execution

At a high level:

1. `agent.main` parses CLI flags and loads env-backed settings.
2. `agent.scheduler` chooses `run_once`, `run_daemon`, or `run_backtest`.
3. `agent.run_cycle` performs one fixture/window decision cycle.
4. The run writes decision artifacts, ledger artifacts, and review artifacts to `storage/`.

## Decision pipeline

`agent.run_cycle.run_cycle()` is the core path.

### 1. Fixture selection

- live mode uses `data.sportmonks.discover_fixtures_safe()`
- synthetic mode uses `data.synthetic_fixtures.synthetic_fixtures()`
- fixture filtering can be forced with `--fixture-code`

### 2. Data collection

The cycle gathers:

- fixture detail from Sportmonks
- market probabilities from Polymarket mapping/Gamma/CLOB proxies
- bookmaker-derived probabilities parsed from Sportmonks detail
- prior probabilities from Supabase
- lineup information from the fixture payload
- halftime checkpoint data if the window is `HT`

If live data is missing, the code falls back to demo or synthetic defaults rather than crashing.

### 3. Optional LLM feature extraction

If `--use-llm-claims` is enabled:

- text sources are sent to Anthropic
- the result must parse as structured JSON
- claims are validated, capped, and overridden by official sources when needed

This path is constrained to feature extraction, not final forecasting.

### 4. Deterministic probability modeling

The probability stack is deterministic by default:

- `models.probability.pre_match_model()`
- `models.halftime.evaluate_halftime()`
- `models.probability.halftime_model()`
- `models.draw_model.apply_draw_model()`

Supporting model modules include:

- `models.lineup_delta`
- `models.consensus`
- `models.edge_engine`
- `models.market_stale`
- `models.source_reconciliation`
- `models.signal_scoring`
- `models.sanity_checks`
- `models.bet_sizing`

The overall behavior is:

- blend available sources
- calibrate and shrink toward market when appropriate
- score supporting and conflicting signals
- estimate confidence and uncertainty
- classify the edge versus market pricing

### 5. Optional LLM analyst

If `--use-llm-analyst` is enabled:

- Anthropic receives a compact decision context
- it returns a bounded JSON recommendation such as `BET`, `SKIP`, or `WATCH`
- `models.llm_decision.merge_llm_analysis_into_risk()` can add blocking flags

This means the LLM can add caution or veto, but it does not become the source of truth for order placement.

### 5a. LLM-central forecast mode

If `decision_mode=llm_central`:

- the run still gathers the same full feature set
- a central Anthropic forecast step synthesizes those features into the primary `home/draw/away` distribution
- the result includes confidence, uncertainty, supporting signals, contradicting signals, and extra risk flags
- if the LLM result is missing or invalid, the run falls back to the deterministic probabilities for artifact continuity, but it adds blocking flags so orders remain disallowed

This mode is designed for experimentation where the LLM is the main forecaster rather than a bounded sidecar.

### 6. Risk and order gating

The final gate combines:

- edge tier
- confidence
- uncertainty
- market completeness
- lineup status
- duplicate-order protection
- dry-run status
- any extra deterministic or LLM-derived blocking flags

Only after that does the agent compute:

- bet size
- limit price
- order payload

### 7. Ledger and review artifacts

Each run constructs a DAG-shaped reasoning ledger via `reasoning.ledger_builder`.

The standard trace includes:

- `Observing`
- `Planning`
- `ToolCalling`
- `Thinking`
- `Acting`
- `Reflecting`

The ledger is always saved locally first. In dry-run mode it remains local-only.

`reasoning.review_writer` then creates markdown/json review artifacts summarizing:

- run metadata
- data availability
- predictions
- risk flags
- ledger status
- optional Anthropic health/critic info

## Module layout

### Runtime

- `agent/main.py`: CLI
- `agent/scheduler.py`: once/daemon orchestration
- `agent/run_cycle.py`: one decision cycle
- `agent/config.py`: env-backed settings

### Data adapters

- `data/sportmonks.py`
- `data/polymarket.py`
- `data/supabase_data.py`
- `data/market_memory.py`
- `data/lineup_monitor.py`
- `data/synthetic_fixtures.py`

### Models

- `models/probability.py`
- `models/probability_blender.py`
- `models/calibration.py`
- `models/edge_engine.py`
- `models/consensus.py`
- `models/halftime.py`
- `models/draw_model.py`
- `models/lineup_delta.py`
- `models/sanity_checks.py`
- `models/bet_sizing.py`
- `models/market_stale.py`
- `models/source_reconciliation.py`
- `models/signal_scoring.py`
- `models/llm_decision.py`

### Reasoning and reporting

- `reasoning/ledger_builder.py`
- `reasoning/review_writer.py`
- `reasoning/anthropic_review.py`
- `reasoning/claim_extraction.py`
- `reasoning/run_report.py`
- `reasoning/trace_quality.py`

### Simulation and tests

- `backtesting/runner.py`
- `tests/`

## Legacy path

The repo also contains an older top-level implementation:

- `agent.py`
- `config.py`
- `ledger/`
- `betting/`
- older `reasoning/llm.py` and prompt-driven notebook flow

That path is still useful as historical context, but it is not the cleanest representation of the currently tested package agent.
