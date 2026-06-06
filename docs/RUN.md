# RUN.md — Upgraded World Cup Arena Agent

This document explains what was built in this upgrade, how the pieces fit, and
exactly how to run and verify the agent.

---

## TL;DR — how to run

```bash
# 1. Install deps (use the same interpreter you run the agent with)
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pip install -r requirements.txt

# 2. Make sure .env is filled (see .env.example for every key)

# 3. Smoke-test connectivity (no orders, no ledger submit)
python agent.py --test-connection

# 4. Run a single fixture pre-match (THIS submits a prediction + ledger,
#    and MAY place a real order — this is the live run)
python3.11 agent.py --fixture-id 19609127 --window prematch

# 5. Half-time window for the same fixture
python3.11 agent.py --fixture-id 19609127 --window halftime
```

> Use the interpreter that has the deps installed. During development that was
> `/opt/homebrew/opt/python@3.11/bin/python3.11`.

---

## What this upgrade added

The agent went from a single market-blind LLM prediction to a **multi-model
reasoning council fed by six data sources, gated by a deterministic risk layer**,
and emitting a richly linked ledger DAG.

### New data sources (`data/`)

| File | Purpose | Keys | Failure mode |
|------|---------|------|--------------|
| `web_search.py` | Targeted injury / lineup / preview search (Serper, DuckDuckGo fallback). Skips lineup queries when Sportmonks already has confirmed lineups. | `SERPER_API_KEY` (optional) | Returns empty bundle |
| `reddit_sentiment.py` | r/soccer thread comments for crowd sentiment. Falls back to `site:reddit.com` web search when Reddit blocks direct JSON. | none | Returns empty bundle |
| `kalshi.py` | Kalshi moneyline as a second market to triangulate vs Polymarket. Strict both-team matching to avoid the elections host's geopolitical noise. | `KALSHI_API_KEY` (optional) | Returns all-None mids |

All three **fail soft**: any network/parse error returns an empty result and the
run continues.

### The reasoning council (`reasoning/council.py` + prompts)

Four roles, four distinct models, one ledger record each:

```
Scout      claude-haiku-4-5     Planning   ← web + reddit + sportmonks
   │  flags injuries / lineups / crowd lean (severity-tagged)
   ▼
Analyst    claude-sonnet-4-5    Thinking   ← sportmonks + supabase + scout
   │  MARKET-BLIND base probability (no prices shown — prevents anchoring)
   ▼
Devil      deepseek-reasoner    Thinking   ← analyst
   │  strongest counter-case; raw chain-of-thought → richest internal_reasoning
   ▼
Judge      claude-sonnet-4-5    Thinking   ← analyst + devil + polymarket + kalshi
        final calibrated probability; ONLY here do market prices enter
```

The Judge's output is the PSL-scored prediction. The DAG converges
(Scout→Analyst→Devil→Judge→Prediction), which is exactly the multi-step,
cross-linked trace the reasoning-quality rubric rewards.

### Deterministic gates (`reasoning/gates.py`)

A pure function applied **after** the council, **before** Kelly sizing. It can
veto a trade or scale its size:

1. **Wallet floor** — no trade below `MIN_WALLET_USD` ($2).
2. **Market exists** — needs a usable Polymarket mid.
3. **Minimum edge** — skip if `model_prob − pm_mid < MIN_EDGE` (5pp).
4. **Cross-market consensus** — Polymarket vs Kalshi agree within 3pp → size ×1.25; diverge >8pp → size ×0.50.
5. **Scout veto** — a high-severity flag on our predicted side kills the trade.
6. **Confidence scaling** — low council confidence ×0.5, high ×1.2.

Final size = `kelly_usd(prob, pm_mid, wallet) × bet_multiplier`, capped by
`MAX_BET_USD` and wallet, with a $1 minimum.

### Ledger upgrades (`ledger/client.py`)

- New `planning()` (Scout + gate decision) and `reflecting()` (closing
  retrospective) behaviors, both schema v0.3.
- DAG linkage flows through `upstream_record_id` (the field the arena actually
  scores — there is no `parent_ids` in the live schema, confirmed against the
  sample notebook).

### Pre-match ledger trace (~21 records, 22 with an order)

```
Observing   trigger
ToolCalling schedule, pm_slug, sm_fixture, web_search, reddit,
            pm_event, pm_mids, kalshi, sb_catalog, sb_priors
Thinking    sm_digest, pm_digest, sb_digest, analyst, devil, judge
Planning    scout, gates
Acting      prediction  (+ order, only if gates pass)
Reflecting  closing retrospective
```

---

## Important operational notes

- **DeepSeek needs balance.** The provided `DEEPSEEK_API_KEY` currently returns
  `402 Insufficient Balance`. The council automatically falls back to Claude for
  the Devil's Advocate, so everything still runs — but to get DeepSeek's *raw*
  chain-of-thought (the richest `internal_reasoning` for scoring), top up the
  DeepSeek account. Verified: with no balance, the run completes on Claude.

- **Serper key unlocks research.** Without `SERPER_API_KEY`, web search and the
  Reddit search-fallback return empty (DuckDuckGo's Instant Answer API is too
  thin for match news). The council still runs — the Scout just has fewer
  external flags. Add a free Serper key for full signal coverage.

- **Reddit blocks datacenter IPs.** Direct `reddit.com/*.json` returns 403 from
  many hosts now; the module falls back to `site:reddit.com` via the search
  backend, so sentiment still flows *if* a Serper key is set.

- **Kalshi** rarely has a clean market for a far-future fixture, so all-None
  mids are normal — the cross-market gate simply abstains and the trade proceeds
  on Polymarket alone.

- **Half-time flow** still uses the original predict/strategy path; only the
  pre-match flow was converted to the council. It remains fully functional and
  can be upgraded to a council the same way later.

---

## How it was verified (no live arena writes)

- `gates.evaluate_gates` unit-checked across all five branches (edge veto,
  consensus boost, contested size-down, scout veto, wallet floor).
- `data/{reddit_sentiment,web_search,kalshi}` exercised live; confirmed graceful
  degradation and the Reddit/Kalshi fixes.
- `reasoning/council.run_council` run end-to-end with mocked digests: real
  Claude calls for Scout/Analyst/Judge, DeepSeek attempted then Claude fallback
  for Devil; produced a calibrated, normalized probability map with ~2.7k-char
  reasoning chains captured per role.
- `ledger` `planning()` / `reflecting()` records dumped and validated against
  schema v0.3 (correct `behavior`, `upstream_record_id`, stringified payloads).

The only thing **not** run is the final `python agent.py --fixture-id ...`,
which performs the live prediction/order/ledger submission — left for you.
