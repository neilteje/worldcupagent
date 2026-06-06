# World Cup Agent Docs

This folder documents the current package-based agent implementation in this repo.

The active entrypoint is `python -m agent.main`.

The older top-level files `agent.py` and `config.py` still exist, but they describe an earlier notebook-style flow and should be treated as legacy unless you are intentionally maintaining that path.

## Quick start

From the `worldcupagent/` directory:

```bash
pip install -r requirements.txt
python -m agent.main --once --dry-run
```

If you want deterministic offline-ish coverage without relying on live arena data:

```bash
python -m agent.main --once --dry-run --use-synthetic-fixtures --verbose
```

## What this agent does

The current agent:

- discovers fixtures from Sportmonks or synthetic fixtures
- fetches market, bookmaker, prior, lineup, and optional halftime data
- blends probabilities with deterministic models
- detects trading edges versus Polymarket
- applies hard risk gates before any order is allowed
- writes a reasoning-ledger DAG and local review artifacts for each run
- optionally uses Anthropic for bounded claim extraction, signal analysis, and critique

## Doc map

- [RUNNING.md](./RUNNING.md): setup, commands, environment variables, outputs
- [ARCHITECTURE.md](./ARCHITECTURE.md): execution flow, modules, and data path
- [REPO_STATE.md](./REPO_STATE.md): current repo status, tests, legacy paths, caveats
