# RUN.md — World Cup Arena Agent

How to pick any World Cup game, run the agent on it, and understand what it does.

---

## 0. One-time setup

```bash
# Install deps into the interpreter you'll run the agent with.
/opt/homebrew/opt/python@3.11/bin/python3.11 -m pip install -r requirements.txt
```

Fill `.env` (see `.env.example` for every key). Required: `STAIR_API_KEY`,
`ANTHROPIC_API_KEY`. Strongly recommended: `SERPER_API_KEY` (web + social
research), `DEEPSEEK_API_KEY` with balance (devil's-advocate raw reasoning),
`XAI_API_KEY` (Grok live social pulse). Optional: `KALSHI_API_KEY`.

> Throughout, `python` means the interpreter that has the deps installed. In dev
> that was `/opt/homebrew/opt/python@3.11/bin/python3.11`.

---

## 1. Pick any game

List every WC2026 fixture with its id:

```bash
python agent.py --list
```

You'll get a table like:

```
 Fixture ID   Kickoff (UTC)         Match
 19609127     2026-06-11 19:00:00   Mexico vs South Africa
 19609143     2026-06-16 19:00:00   France vs Senegal
 19609166     2026-06-17 17:00:00   Portugal vs Congo DR
 ...                                (72 fixtures)
```

Copy the **Fixture ID** of the game you want.

---

## 2. Run the agent on that game

```bash
# Pre-match window (forms the prediction, runs the council, may place an order)
python agent.py --fixture-id 19609143 --window prematch

# Half-time window (Bayesian update on the live match state)
python agent.py --fixture-id 19609143 --window halftime
```

That's the whole loop. Swap the id for any game from `--list`.

```bash
# Verify every data source + which LLM providers are configured (writes nothing)
python agent.py --test-connection

# Run the full loop across ALL fixtures, one after another
python agent.py --scan --window prematch
```

---

## 3. What happens during a pre-match run

```
[1] Sportmonks schedule  → discover the fixture
[2] Polymarket slug      → map fixture → market
[3] Sportmonks fixture   → ML probs, odds, xG  → Claude digest
[4] Web + Reddit research→ injuries / lineups / previews across ~30 sources
[5] Polymarket moneyline → live CLOB mids       → Claude digest
[6] Kalshi moneyline     → second market for cross-checking
[7] Supabase priors      → H2H, set-piece, KO/stage records → Claude digest
[8] Reasoning council    → Pulse → Scout → Analyst → Devil → Judge
[9] Deterministic gates  → edge / consensus / scout-veto / confidence → Kelly size
    → prediction (PSL-scored) + optional order + full ledger trace
```

### The reasoning council (the brain)

```
Grok        grok-4.3            ToolCalling ← live X/Twitter + news social pulse
   │  breaking injuries, lineup leaks, fan mood (xAI's training edge)
   ▼
Scout       claude-haiku-4-5    Planning    ← web + reddit + grok + sportmonks
   │  consolidates everything into severity-tagged flags
   ▼
Analyst     claude-sonnet-4-5   Thinking    ← sportmonks + supabase + scout
   │  MARKET-BLIND base probability (never sees prices → no anchoring)
   ▼
Devil       deepseek-reasoner   Thinking    ← analyst
   │  strongest counter-case; raw chain-of-thought = richest ledger reasoning
   ▼
Judge       claude-sonnet-4-5   Thinking    ← analyst + devil + polymarket + kalshi
        final calibrated probability; ONLY here do market prices enter
```

The Judge's output is the PSL-scored prediction. The records link into a DAG
(Grok→Scout→Analyst→Devil→Judge→Prediction→Reflection) via `upstream_record_id`,
which is what the reasoning-quality rubric rewards.

### Deterministic gates (the risk layer)

Applied after the council, before Kelly sizing — can veto or scale a bet:

1. Wallet floor (no trade below $2)
2. Market must exist (usable Polymarket mid)
3. Minimum edge (skip if `model_prob − pm_mid < 5pp`)
4. Cross-market consensus (Polymarket vs Kalshi agree within 3pp → ×1.25; diverge >8pp → ×0.50)
5. Scout veto (high-severity flag on our pick kills the trade)
6. Confidence scaling (low ×0.5, high ×1.2)

Final size = `kelly_usd(prob, pm_mid, wallet) × multiplier`, capped by
`MAX_BET_USD` and wallet, $1 minimum.

---

## 4. Data sources (all fail soft)

| Source | What it gives | Key | Notes |
|--------|---------------|-----|-------|
| Sportmonks | fixtures, ML probs, odds, xG, lineups | `STAIR_API_KEY` | via arena proxy |
| Polymarket | event slug, condition/token ids, live mids | `STAIR_API_KEY` | Gamma + CLOB proxies — **verified live** |
| Supabase | StatsBomb priors + live HT checkpoints | shared key (built in) | **verified live** |
| Web search | injury/lineup/preview across BBC, ESPN, Guardian, FIFA, Forebet, … | `SERPER_API_KEY` | ~70 results / ~30 domains per fixture |
| Reddit | r/soccer crowd takes | none | direct API 403s → auto-falls back to Serper `site:reddit.com` |
| Kalshi | second prediction market | optional | strict both-team match; far-future fixtures often have none |
| Grok | live X/Twitter + news social pulse | `XAI_API_KEY` | xAI model trained on X data |

### Supabase country-id fix (important)

The priors tables key on **StatsBomb** `country_id` (Mexico = 147), which differs
from **Sportmonks** team ids (Mexico = 458). The agent now resolves team name →
StatsBomb id automatically by building a name→id map from `ads_a_h2h_country`
(which carries both). Result: priors fetch returns the **2 relevant rows** for
the two teams instead of dumping all 71 on the LLM. Verify with:

```bash
python agent.py --test-connection   # shows "Country-id resolver (Mexico): 147"
```

---

## 5. Operational notes

- **Grok role.** Grok isn't a redundant 5th voter — it's the *social-pulse
  scout*. Because xAI trains on X/Twitter, it's the right tool for live breaking
  news / fan mood, which then feeds the Scout. Without `XAI_API_KEY` the pulse is
  skipped and the council runs on web + Reddit alone (`--test-connection` warns
  you if the key is missing).

- **DeepSeek needs balance.** If the DeepSeek account is empty it returns
  `402`; the Devil's Advocate then falls back to Claude automatically (summarized
  thinking instead of raw). Top up to get the richest `internal_reasoning`.

- **Serper unlocks both web AND Reddit.** Reddit blocks unauthenticated JSON from
  many IPs, so the Reddit module falls back to `site:reddit.com` via Serper — so
  one Serper key powers both. Confirmed: 70 results / 28 sources for MEX–RSA.

- **Half-time flow** still uses the original predict/strategy path; only
  pre-match runs the full council. It can be upgraded the same way later.

---

## 6. Verified live (this session)

- Polymarket: slug `fifwc-mex-rsa-2026-06-11`, mids home 0.685 / draw 0.205 / away 0.105.
- Supabase: catalog (14 tables), priors fetch returns exactly country_ids 147 & 211.
- Country-id resolver: Mexico→147, South Africa→211, USA→241, Brazil→31.
- Web search: 70 results across 28 domains via Serper.
- Council: end-to-end with real Claude + DeepSeek; calibrated normalized probabilities.
- `--list` enumerates all 72 fixtures with ids.

Not auto-run: the final live `--fixture-id … --window prematch` (it submits to
the arena and may place an order) — that one is yours to fire.
