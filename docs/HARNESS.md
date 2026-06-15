# Paper-Trading Harness

A live dress rehearsal for the World Cup. It runs the **same kickoff + half-time
windows** the arena rules define, produces one shared prediction per window, and
lets **all four agent profiles** paper-trade it side by side. Nothing touches the
arena — the broker is a demo "calc sheet" that fills against real Polymarket mids
when available and a clearly-labeled synthetic reference when a fixture has no
market.

## The four agents

All four consume the **same forecast**; they differ only in trading policy, so
the head-to-head is a clean A/B/C/D. Profiles live in `harness/profiles.py` —
the **single source of truth** shared with the arena agent (`agent.py
--profile` / `AGENT_PROFILE` env). Tune via a `--profiles` JSON override.

| Agent | Mandate (STRATEGY.md §4) | Edge bar (vs fair) | EV floor | Conf floor | Max entry price | Kelly | Max bet | Bets/window | Scout veto | Synthetic |
|---|---|---|---|---|---|---|---|---|---|---|
| `monk` | ORACLE — forecast specialist | 10.0pp | 8%/$ | medium+ (0.55) | — | 0.20× | $2 | 1 | ON | never |
| `anchor` | KEEL — disciplined EV accumulator | 4.5pp | 2%/$ | low+ (0.40) | — | 0.35× | $4 | 1 | ON | never |
| `hunter` | SAW — skew harvester (draws/dogs) | 3.0pp | 1%/$ | low+ (0.40) | **0.40** | 0.75× | $5 | 2 | ON | never |
| `blitz` | SURGE — event-driven aggression | 2.0pp | any +EV | 0.35 | — | 0.65× | $5 | 2 | OFF | ×0.25 size |

Expected fire rates on real Polymarket mids for typical WC group games:
monk a handful all tournament (deliberate — it competes on the Score track),
anchor ~20–40%, hunter on skewed prices only, blitz ~40–60%. Only `monk` may
legitimately sit near 0% — if anchor or blitz does, retune its bars
(STRATEGY.md §6 pivot triggers).

**Synthetic market honesty:** demo prices are derived from our own forecast +
noise, so "edge" against them is noise. Profiles default to `trade_synthetic:
false` (they HOLD); blitz may opt in but is sized down ×0.25 and every such
trade carries `synthetic_warning: true` in the ledger.

## Run it on match day

```bash
PY=.venv/bin/python   # interpreter with deps

# 1) Initialize the session (snapshots fixtures + profiles, prints the schedule)
$PY -m harness init

# 2) Go live. Waits for each window's trigger time, then predicts + paper-trades
#    all four agents. PRE_MATCH fires ~5 min before kickoff; HT ~50 min after.
$PY -m harness run

# 3) After matches end, fill the winners in:
#    storage/harness/<date>/results.json   (set result_slot = home|draw|away)

# 4) Settle + generate per-agent P&L, ROI, calibration, CSV, and plots
$PY -m harness settle
```

Outputs land in `storage/harness/<date>/`:
`ledger.json`, `performance.csv`, `summary.md`, `summary.json`, `plots/*.png`.

## Useful variations

```bash
# Run a single window right now (manual / testing)
$PY -m harness now --fixture FRD-POR-NGA --window PRE_MATCH

# Force a fresh council prediction (ignore the cache)
$PY -m harness now --fixture FRD-POR-NGA --window PRE_MATCH --engine council --refresh

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
   `deterministic` → `market` if keys/network are unavailable). The council is
   fed the SAME structured grounding as the arena path: Sportmonks digest (when
   the fixture has a `sportmonks_fixture_id`) + Supabase priors (resolved by
   team name — works for friendlies too) via `data/fixture_bundle.py`, plus
   web/Reddit/Grok research. Council output passes through
   `reasoning/grounding.py` (anchor sanity checks + documented low-confidence
   shrink toward bookmaker consensus). Cached so all agents and re-runs share
   the identical forecast.
2. **Market snapshot** — real Polymarket mids by slug, else a seeded synthetic
   reference (`market_source` is recorded on every trade).
3. **EV decision** — `betting/decision.py` de-vigs the market, ranks all three
   outcomes by EV, and sizes with the profile's fractional Kelly. The profile's
   `min_edge_vs_fair` is the ONLY edge bar (no second bar in gates).
4. **Profile policy** — EV floor, confidence floor, scout-flag veto, stake
   caps, max-bets-per-window, synthetic-market policy.
5. **Settle** at results time → P&L, ROI, win rate; predictions are scored with
   Brier / log-loss / accuracy.

## Notes

- Engine `council` calls real LLMs (Grok → Scout → Analyst → Devil → Judge) and
  needs the usual keys; without them it degrades gracefully (and loudly logs
  which role failed).
- Fixtures without a Polymarket market get `synthetic_demo` pricing — useful to
  exercise the pipeline, but by default no profile bets it (see above). When a
  real slug exists, drop it into the fixtures override (`pm_slug`) and the
  broker trades live mids. Add `sportmonks_fixture_id` to the override to give
  the council real ML/odds/xG grounding.
- Everything under `storage/` is gitignored; only the harness source is tracked.
