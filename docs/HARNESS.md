# Paper-Trading Harness

A live dress rehearsal for the World Cup. It runs the **same kickoff + half-time
windows** the arena rules define, produces one shared prediction per window, and
lets **multiple differently-tuned agents** paper-trade it. Nothing touches the
arena — the broker is a demo "calc sheet" that fills against real Polymarket mids
when available and a clearly-labeled synthetic reference when a friendly has no
market.

## The two agents (tomorrow)

Both consume the **same forecast**; they differ only in aggressiveness, so the
head-to-head is a clean A/B. Tune any knob in `harness/profiles.py` (or via a
`--profiles` JSON override).

| Agent | Label | Edge bar (vs fair) | Kelly | Max bet | Conf floor | Style |
|---|---|---|---|---|---|---|
| `conservative` | anchor | 6.0pp | 0.25× | $3 | medium+ | bets only clear edges, low variance |
| `aggressive` | blitz | 2.5pp | 0.60× | $5 | low+ | acts on thin edges, both windows, 2 picks |

## Run it tomorrow

```bash
PY=/opt/homebrew/opt/python@3.11/bin/python3.11   # interpreter with deps

# 1) Initialize the session (snapshots fixtures + prints the window schedule)
$PY -m harness init

# 2) Go live. Waits for each window's trigger time, then predicts + paper-trades.
#    PRE_MATCH fires ~5 min before kickoff; HT fires ~50 min after kickoff.
$PY -m harness run

# 3) After matches end, fill the winners in:
#    storage/harness/2026-06-10/results.json   (set result_slot = home|draw|away)

# 4) Settle + generate per-agent P&L, ROI, calibration, CSV, and plots
$PY -m harness settle
```

Outputs land in `storage/harness/<date>/`:
`ledger.json`, `performance.csv`, `summary.md`, `summary.json`, `plots/*.png`.

## Useful variations

```bash
# Run a single window right now (manual / testing)
$PY -m harness now --fixture FRD-POR-NGA --window PRE_MATCH

# Smoke-test the whole day instantly (ignores wait times)
$PY -m harness run --start-now --engine market --market synthetic

# Skip LLM calls (use the deterministic engine)
$PY -m harness run --engine deterministic

# Force the demo market even if a real Polymarket slug exists
$PY -m harness run --market synthetic

# Regenerate plots/CSV from the current state without re-trading
$PY -m harness report
```

## How it decides (per agent, per window)

1. **Predict once** (`--engine council` by default; falls back to
   `deterministic` → `market` if keys/network are unavailable). Cached so both
   agents and any re-run share the identical forecast.
2. **Market snapshot** — real Polymarket mids by slug, else a seeded synthetic
   reference (`market_source` is recorded on every trade).
3. **EV decision** — `betting/decision.py` de-vigs the market, ranks all three
   outcomes by EV, and sizes with fractional Kelly.
4. **Profile policy** — each agent applies its edge bar, EV floor, confidence
   floor, scout-flag veto, stake caps, and max-bets-per-window.
5. **Settle** at results time → P&L, ROI, win rate; predictions are scored with
   Brier / log-loss / accuracy.

## Notes

- Engine `council` calls real LLMs (Grok → Scout → Analyst → Devil → Judge) and
  needs the usual keys; without them it degrades gracefully.
- Friendlies usually have **no Polymarket market**, so expect `synthetic_demo`
  pricing — useful to exercise the full pipeline, but the P&L is illustrative,
  not real alpha. When a real slug exists, drop it into the fixtures override and
  the broker trades live mids.
- Everything under `storage/` is gitignored; only the harness source is tracked.
