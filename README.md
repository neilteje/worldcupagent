# World Cup Arena Agent

Our AI agent competing in the 2026 Stair AI hackathon.

This repo hosts **two complementary forecasting engines** that share the same data
layer. They live side by side today and are designed to be combined into a single
agent that feeds deterministic signals **and** web/LLM research into one decision.

| Engine | Entry point | What it is |
|--------|-------------|------------|
| **LLM Council** | `agent.py`, `predict_game.py` | Web search + Reddit + Grok social pulse → Scout → Analyst → Devil → Judge council, then EV-ranked trade decision. Reasoning-trace heavy (PSL + ledger). |
| **Deterministic** | `agent/main.py` | Quantitative pipeline: probability models, calibration, edge engine, meta-arbiter, backtester. Reproducible, fast, no LLM required for its core. |

> **The endgame:** a unified agent where the deterministic engine produces grounded
> signals, the web/LLM research layer adds context, and the LLM council reconciles
> both into the final World Cup trade. This merge puts both engines in one place so
> that integration can begin.

## Layout

```
LLM-COUNCIL ENGINE
  agent.py                  ← council orchestration (pre-match + half-time)
  predict_game.py           ← sandbox predictor for any fixture (no arena writes)
  reasoning/
    council.py              ← Grok → Scout → Analyst → Devil → Judge
    gates.py                ← deterministic trade gates
    llm.py, prompts.py      ← multi-model calls + structured prompts
  betting/
    decision.py             ← EV-ranked, all-outcomes, de-vigged decision engine
    kelly.py                ← Kelly criterion sizing
  data/
    web_search.py, reddit_sentiment.py, kalshi.py, supabase_client.py

DETERMINISTIC ENGINE
  agent/
    main.py, run_cycle.py, scheduler.py, config.py
  models/                   ← probability, calibration, edge_engine, meta_arbiter,
                              consensus, draw_model, blender, archetype, halftime, …
  backtesting/              ← runner + 2022 World Cup historical harness
  reasoning/                ← schemas, central_llm, anthropic_review, claim_extraction,
                              ledger_builder, trace_quality, counterfactuals, …
  data/
    supabase_data.py, market_memory.py, lineup_monitor.py, synthetic_fixtures.py

SHARED
  data/polymarket.py        ← superset: get_moneyline() (council) +
                              get_three_way_market_probs() (deterministic)
  data/sportmonks.py        ← fixtures, ML predictions, odds, HT stats
  config.py, ledger/client.py
  storage/                  ← deterministic-engine run artifacts (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in STAIR_API_KEY, ANTHROPIC_API_KEY, etc.
```

## Running

### LLM Council engine
```bash
python agent.py --list                                   # list fixtures
python agent.py --fixture-id 19609127 --window prematch  # full council + trade
python predict_game.py --home Brazil --away Morocco --home-code BRA --away-code MAR
```

### Deterministic engine
```bash
python -m agent.main            # run one decision cycle
python -m backtesting.runner    # 2022 World Cup backtest
pytest tests/ -q                # unit tests (live-data tests need STAIR_API_KEY)
```

## Scoring strategy

- **PSL (Probabilistic Skill Loss)** — proper scoring on calibrated distributions.
  Emitted on every game regardless of whether a trade fires.
- **Reasoning quality** — the full ledger trace (DAG of records) is scored.
- **P&L ($1,000 track)** — EV-ranked, all-outcomes decision engine with de-vigged
  edges and Kelly sizing; trades only the highest-EV side that clears the bar.

## Tuning (`config.py`)

| Variable | Default | Effect |
|---|---|---|
| `MIN_EDGE` | 0.05 | Min raw edge to place a bet |
| `MIN_EDGE_VS_FAIR` | 0.03 | Min edge vs the de-vigged fair price |
| `MAX_KELLY_FRACTION` | 0.20 | Max % of wallet per bet |
| `MAX_BET_USD` | 15.00 | Hard USD cap per order |

The deterministic engine has its own settings in `agent/config.py` (`Settings`).
