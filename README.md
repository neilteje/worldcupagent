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
