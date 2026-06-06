# Running The Agent

## Requirements

- Python environment with the dependencies in `requirements.txt`
- optional `ARENA_KEY` for live Stair arena access
- optional `ANTHROPIC_KEY` for Anthropic health checks, claim extraction, analyst, or critique features

## Install

From `worldcupagent/`:

```bash
pip install -r requirements.txt
```

The repo uses `python-dotenv`, so a local `.env` file is a practical way to set keys.

## Important environment variables

The package agent reads settings from `agent/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `DRY_RUN` | `true` | Prevents live order submission. This is the safe default. |
| `ARENA_KEY` or `STAIR_API_KEY` | empty | Required for live arena reads and writes. |
| `SUPABASE_URL` | shared staging URL | Supabase REST base URL. |
| `SUPABASE_PUBLISHABLE_KEY` or `SUPABASE_KEY` | shared staging key | Supabase read key. |
| `MAX_ORDER_USD` | `4.0` | Hard per-order cap, additionally capped to `$5`. |
| `MIN_EDGE_TO_BET` | `0.06` | Minimum edge used by the decision logic. |
| `TIME_IN_FORCE_SECONDS` | `30` | Limit-order time in force. |
| `STORAGE_DIR` | `storage` | Where run artifacts are written. |
| `ANTHROPIC_KEY` or `ANTHROPIC_API_KEY` | empty | Enables optional Anthropic features. |
| `LLM_SIGNAL_DELTA_CAP` | `0.02` | Caps LLM-derived signal influence. |
| `BACKTEST_LLM_BUDGET_USD` | `5.0` | Budget guard for optional backtest critique. |
| `DECISION_MODE` | `deterministic` | Forecasting mode: deterministic-first or `llm_central`. |

## Main commands

Single safe run:

```bash
python -m agent.main --once --dry-run
```

Single fixture/window:

```bash
python -m agent.main --once --fixture-code WC2026-GS-M1 --window PRE_MATCH --dry-run
python -m agent.main --once --fixture-code WC2026-GS-M1 --window HT --dry-run
```

Synthetic fixtures:

```bash
python -m agent.main --once --dry-run --use-synthetic-fixtures --verbose
```

Daemon mode:

```bash
python -m agent.main --daemon --interval-seconds 300 --dry-run
python -m agent.main --daemon --interval-seconds 60 --dry-run --max-iterations 2 --use-synthetic-fixtures
```

Backtest:

```bash
python -m agent.main --backtest --backtest-sample 50
```

Optional Anthropic checks and augmentations:

```bash
python -m agent.main --once --dry-run --anthropic-health-check
python -m agent.main --once --dry-run --use-llm-claims
python -m agent.main --once --dry-run --use-llm-analyst
python -m agent.main --once --dry-run --use-anthropic-critic
```

LLM-central mode:

```bash
python -m agent.main --once --dry-run --decision-mode llm_central
python -m agent.main --once --dry-run --decision-mode llm_central --use-llm-claims
```

In `llm_central` mode, the LLM becomes the primary forecaster and synthesizes the full feature bundle into the main probability forecast. Deterministic gates still handle edge checks, sizing, and order safety.

## Safety model

- `DRY_RUN=true` is the default.
- A prediction cycle still runs in dry-run mode and writes decision, ledger, and review artifacts.
- Orders remain blocked unless deterministic gates pass.
- Even with Anthropic enabled, the LLM path cannot directly authorize an order.

Live orders require all of the following:

- `DRY_RUN=false`
- valid market and model probabilities
- tradable edge after deterministic checks
- no blocking risk flags
- positive bet size and valid limit price

## Output folders

The package agent writes to `storage/` by default:

- `storage/decisions/`: final decision JSON per fixture/window
- `storage/price_history/`: market snapshots
- `storage/runs/`: local ledger session payloads
- `storage/reviews/`: markdown/json run reviews and Anthropic review artifacts
- `storage/backtests/`: backtest summaries and decisions

## Validation

Current automated validation command:

```bash
pytest -q
```

Useful runtime smoke checks:

```bash
python -m agent.main --once --dry-run
python -m agent.main --once --dry-run --use-synthetic-fixtures --verbose
python -m agent.main --backtest --backtest-sample 50
```
