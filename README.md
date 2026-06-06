# World Cup Arena Agent

An autonomous AI agent competing in the [Stair AI World Cup Agent Arena](https://stair-ai.com).
Built to win the **Highest Stair AI Score** ($2,000) by combining well-calibrated
probabilistic predictions with rich, structured reasoning traces.

## Architecture

```
agent.py                ← orchestration (pre-match + half-time flows)
config.py               ← env-backed configuration
data/
  sportmonks.py         ← Sportmonks proxy (fixtures, ML predictions, HT stats)
  supabase_client.py    ← StatsBomb priors + live checkpoints via Supabase
  polymarket.py         ← Market prices via arena proxy
reasoning/
  prompts.py            ← structured prompts for pre-match + HT windows
  llm.py                ← Claude extended thinking + Gemini ensemble
ledger/
  client.py             ← 7-behavior reasoning trace builder + batch submit
betting/
  kelly.py              ← Kelly criterion bet sizing
```

## Setup

```bash
# 1. Clone / open in your IDE
cp .env.example .env
# Fill in ARENA_KEY, SUPABASE_KEY, ANTHROPIC_KEY, GEMINI_KEY

# 2. Install dependencies
pip install -r requirements.txt

# 3. Register with Stair AI
#    staging.stair-ai.com → Launch → create API key → post to Discord #registration

# 4. Verify connection
python -c "from data.polymarket import get_listings; print(get_listings()[:2])"
```

## Running the Agent

```bash
# Single fixture, pre-match window
python agent.py --fixture WC2026-GS-M1 --window prematch

# Single fixture, half-time window
python agent.py --fixture WC2026-GS-M1 --window halftime

# Auto-scan all active fixtures
python agent.py --scan
python agent.py --scan --window halftime
```

## Scoring Strategy

### Stair AI Score (Primary Target — $2,000)
- **PSL (Probabilistic Skill Loss)**: proper scoring rule rewarding calibrated
  probability distributions, not just binary winners
- **Reasoning quality**: the full ledger trace is scored — we emit 9+ records
  per session covering every behavior type with rich Claude extended-thinking chains

### P&L (Secondary Target — $1,000)
- Kelly Criterion sizing: bet only when `|model_p - market_p| > 5%`
- Half-Kelly to limit drawdown
- HT window is the main alpha source: live xG/score divergence

### Key Design Decisions
1. **Polymarket prices as prior**: never fight the market without evidence
2. **Claude extended thinking**: captures full internal chain-of-thought for ledger
3. **Gemini ensemble**: 70/30 blend with Gemini 2.5 Pro for calibration
4. **HT Bayesian update**: explicit likelihood update given live xG vs score

## Tuning Parameters (`config.py`)
| Variable | Default | Effect |
|---|---|---|
| `THINKING_BUDGET` | 8000 | Claude thinking tokens — higher = richer trace |
| `MIN_EDGE` | 0.05 | Minimum edge to place a bet |
| `MAX_KELLY_FRACTION` | 0.20 | Max % of wallet per bet |
| `MAX_BET_USD` | 15.00 | Hard USD cap per order |

## Implemented strategy

This repo now includes a deterministic, dry-run-safe arena agent package that can discover available fixtures, compute calibrated win/draw/loss probabilities, compare them with market prices, write a reasoning-ledger DAG, and optionally submit orders only when deterministic risk checks pass.

### Environment variables

| Variable | Purpose | Safe default |
|---|---|---|
| `ARENA_KEY` / `STAIR_API_KEY` | Stair AI arena API key sent as `x-api-key`. | Empty; live submissions disabled. |
| `SUPABASE_URL` | Supabase REST base URL for arena aggregate tables. | Shared staging URL from the Builder Guide. |
| `SUPABASE_PUBLISHABLE_KEY` / `SUPABASE_KEY` | Supabase publishable key. | Shared staging key if present in repo config. |
| `ANTHROPIC_KEY` / `ANTHROPIC_API_KEY` | Optional Claude key for low-cost backtest critique. | Empty; deterministic backtest still runs. |
| `DRY_RUN` | If true, orders are never submitted. | `true`. |
| `MAX_ORDER_USD` | Hard cap for any single order. | `4.00`, capped to the $5 dev wallet. |
| `MIN_EDGE_TO_BET` | Minimum edge used by order logic. | `0.06`. |
| `AGENT_NAME` | Human-readable local agent label. | `worldcupagent`. |
| `BACKTEST_LLM_BUDGET_USD` | Optional Claude backtest-review budget guard. | `5.00`, capped to $5. |

### How to run

```bash
# One safe dry-run cycle. This creates local decision, price-history, and ledger files.
python -m agent.main --once --dry-run

# One cycle for a specific fixture/window.
python -m agent.main --once --fixture-code WC2026-GS-M1 --window PRE_MATCH --dry-run
python -m agent.main --once --fixture-code WC2026-GS-M1 --window HT --dry-run

# Daemon mode. DRY_RUN defaults to true unless DRY_RUN=false is explicitly exported.
python -m agent.main --daemon --interval-seconds 300

# Deterministic synthetic backtest.
python -m agent.main --backtest --backtest-sample 50

# Optional low-token Claude critique of the backtest summary. The deterministic backtest does not depend on Claude.
python -m agent.main --backtest --backtest-sample 50 --use-claude
```

### DRY_RUN and order safety

`DRY_RUN=true` is the default. In dry-run mode the agent still builds predictions, evaluates edges, saves decisions, and constructs local ledger records, but it records `dry_run_enabled` as a blocking risk flag and does not send `POST /api/v1/orders`. Live orders require all of the following:

1. `DRY_RUN=false`.
2. Valid calibrated probabilities for `home`, `draw`, and `away`.
3. Complete market data for the selected outcome.
4. Edge engine says `should_bet=true`.
5. No blocking risk flags from the sanity audit.
6. Bet size is greater than zero and no larger than `MAX_ORDER_USD`.
7. Limit price is inside `[0, 1]` and capped below model fair value.

### Top modules

- **Market disagreement engine** (`models/edge_engine.py`) finds outcomes where model probabilities disagree with Polymarket, labels the edge source, tiers the edge, and blocks low-confidence or high-uncertainty bets.
- **Consensus triangle** (`models/consensus.py`) compares model, bookmaker, and Polymarket top picks, returning agreement cases plus confidence and bet-size modifiers.
- **Half-time scoreline luck model** (`models/halftime.py`) uses score state, xG, shots, and cards to detect deserved leads, lucky leads, dominant draws, dead matches, volatility, and red-card distortions.
- **Lineup delta model** (`models/lineup_delta.py`, `data/lineup_monitor.py`) compares expected vs confirmed starters, scores missing player importance, caps lineup-driven probability moves, and flags unconfirmed lineups.
- **Reasoning Ledger DAG** (`reasoning/ledger_builder.py`) builds Observing, Planning, ToolCalling, Thinking, Acting, and Reflecting records with parent links, saves them locally first, and attempts arena batch submission using the Builder Guide wire endpoint.
- **Backtesting** (`backtesting/runner.py`) runs a deterministic $5-bankroll simulation over synthetic historical-like fixtures, computes Brier scores, ROI, bet counts, and can optionally request a concise Claude review without making agent decisions dependent on the model.

### Output locations

- `storage/price_history/{fixture_code}.jsonl` — each Polymarket/midpoint fetch snapshot.
- `storage/runs/{session_id}.json` — local ledger DAG records, saved before submission.
- `storage/runs/{session_id}.failed.json` — failed ledger submissions after retries.
- `storage/decisions/{fixture_code}-{window}.json` — final decision report for a fixture/window.
- `storage/backtests/backtest-*.json` — backtest summaries and per-match decisions.

### Validation

Run all checks with:

```bash
pytest -q
python -m agent.main --once --dry-run
python -m agent.main --backtest --backtest-sample 50
```

### Self-improvement, synthetic runs, and review artifacts

The package CLI now supports deterministic review-oriented runs:

```bash
# Print extra decision metadata and write storage/reviews artifacts.
python -m agent.main --once --dry-run --verbose

# Exercise ten deterministic synthetic fixtures when live data is unavailable.
python -m agent.main --once --dry-run --use-synthetic-fixtures --verbose

# Validate daemon automation without running forever.
python -m agent.main --daemon --interval-seconds 60 --dry-run --max-iterations 2 --use-synthetic-fixtures
```

Every successful prediction run writes:

- `storage/reviews/run_YYYYMMDD_HHMMSS.md` and `.json` with run metadata, API/data status, prediction summaries, critique, implemented ideas, and next priorities.
- `storage/reviews/latest_review.md` pointing to the newest markdown review.
- `storage/reviews/iteration_log.md` with append-only iteration notes.
- `storage/reviews/comparison.md` with simple before/after run metrics.

Additional quality and safety modules added for iteration:

- **Signal scoring** (`models/signal_scoring.py`) assigns source quality, freshness, corroboration, impact, and final weight to lineup, draw, halftime, stale-market, and weak web signals. Weak web/rumor/sentiment deltas are capped before they can affect confidence or ledger explanations.
- **Draw-specialist model** (`models/draw_model.py`) directly adjusts draw probability for low projected xG, small strength gaps, level HT states, and market/bookmaker draw baselines, with a draw sanity flag for unexplained very-low draw probabilities.
- **Market stale detector** (`models/market_stale.py`) compares current and previous market snapshots with bookmaker/signal movement to flag possible lagging Polymarket prices.
- **Synthetic fixtures** (`data/synthetic_fixtures.py`) cover model/bookmaker-vs-market, bookmaker+market-against-model, HT low-xG draw, favorite trailing with xG dominance, missing goalkeeper, red-card comeback, draw-underpriced, stale-market, weak web-claim, and missing-market scenarios.
- **Duplicate-order prevention** checks the existing decision marker for the same fixture/window and adds a blocking `duplicate_order` risk flag before any non-dry-run order can be sent.
