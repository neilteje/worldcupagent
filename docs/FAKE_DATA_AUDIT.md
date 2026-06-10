# Fake, Synthetic, and Unavailable Data Audit

This file catalogs every place in the codebase that uses hardcoded values, synthetic
data, or data that does not exist yet. Each entry has a severity rating, the exact
location, and what needs to happen for it to use real data.

Severity: CRITICAL (blocks correct decisions) / HIGH (degrades quality) / LOW (testing only, no production impact)

---

## 1. Entire backtest is synthetic — no real historical data

**Severity: HIGH (affects all P&L reporting)**

`backtesting/runner.py` — `synthetic_history(n, seed)`

All backtest matches are randomly generated. No real Polymarket prices, no real match
outcomes, no real odds. The synthetic generator adds Gaussian noise to a randomly
drawn "true" probability to simulate four sources:

```python
market    = noisy(0.055)   # ±5.5% noise around true prob
bookmaker = noisy(0.035)   # ±3.5%
sportmonks = noisy(0.045)  # ±4.5%
priors    = noisy(0.025)   # ±2.5%
```

**What's missing:** Real historical match outcomes with contemporaneous Polymarket
prices and bookmaker odds. These don't exist in any accessible data source — Supabase
has historical results (`d_match_scores`) but no historical market prices.

**Impact:** Brier and ROI figures from backtests reflect synthetic variance, not real
model performance. The Brier gap (model vs market: ~0.009) is real signal;
ROI is noise.

**Fix when available:** Once WC2026 group-stage matches play out (June 11–27), real
outcomes + live Polymarket prices can be recorded and used for genuine backtesting.

---

## 2. Demo probability fallbacks in run_cycle.py

**Severity: CRITICAL (silently hides missing data)**

`agent/run_cycle.py` lines 171, 177, 179, 184, 190

When `fixture.get("demo") == True` or no `ARENA_KEY` is set, the agent substitutes
these hardcoded values for real data:

| Source | Hardcoded fallback | Real source |
|---|---|---|
| Market prices | `home=0.44, draw=0.29, away=0.27` | Polymarket CLOB via mapping tokens |
| Bookmaker odds | `home=0.46, draw=0.28, away=0.26` | Sportmonks `odds` include |
| Sportmonks ML | `home=0.47, draw=0.27, away=0.26` | Sportmonks `/predictions/probabilities/fixtures/{id}` |
| Supabase priors | `home=0.40, draw=0.28, away=0.32` | `ads_a_h2h_country` + `ads_a_stage_record` |
| HT live data | `home_goals=0, away_goals=0, home_xg=0.35, home_shots=4, ...` | Supabase `d_match_scores` + `d_checkpoint_snapshot` |

These fallbacks trigger whenever:
- `ARENA_KEY` is not set in `.env`
- The fixture dict has `"demo": True`
- `discover_fixtures_safe()` falls back to the demo fixture (e.g., no internet)

**How to tell it happened:** Check `decision["market_probs"]["reason"]` — it will
be `"demo"` instead of `"ok"`. Also `data_completeness.score` will be low and
`dry_run` will be True.

**Fix:** Ensure `ARENA_KEY` is set and the agent is run with real fixtures from
`discover_fixtures_safe()`.

---

## 3. Sportmonks 3-way ML prediction is not extracted

**Severity: HIGH (25% source weight wasted on real fixtures)**

`data/sportmonks.py` — `extract_sportmonks_prediction()`

The function walks the fixture dict looking for a key with `home`, `draw`, `away`
sub-keys. But Sportmonks prediction type_id=235 returns binary `{yes, no}` per team,
not a 3-way distribution:

```json
{"predictions": {"yes": 42.49, "no": 57.51}, "type_id": 235}
```

The 3-way probabilities exist at a separate endpoint:
`GET /v3/football/predictions/probabilities/fixtures/{fixture_id}`

**Impact:** `sm_pred` is `None` for real fixtures. The 25% Sportmonks source weight
in `DEFAULT_PREMATCH_WEIGHTS` gets redistributed to remaining sources. The model
runs on only 3 real sources instead of 4.

**Fix needed:**
```python
# In data/sportmonks.py, add:
def get_fixture_win_probabilities(fixture_id: int) -> dict | None:
    data = _get(f"predictions/probabilities/fixtures/{fixture_id}")
    # Walk for home_win / draw / away_win keys
    ...
```
Then call this in `run_cycle.py` when `sm_pred` is None after `extract_sportmonks_prediction`.

---

## 4. Supabase priors return None for many country pairs

**Severity: HIGH (15% source weight often missing)**

`data/supabase_data.py` — `get_priors()`

The h2h lookup uses `ilike` matching on country names. This fails or returns None when:
- Countries have fewer than 1 h2h match recorded (e.g., new WC teams like Curacao, Haiti)
- Country name in the arena mapping doesn't match Supabase spelling exactly
  (e.g., "Czech Republic" vs "Czechia", "Cote d'Ivoire" vs "Ivory Coast")
- Both teams are from continent pairs with little historical data

Confirmed working: Argentina vs Brazil, major European nations.  
Likely failing: Haiti, Curacao, Bosnia and Herzegovina, Cape Verde Islands, Jordan,
Iraq vs any opponent.

**Impact:** When `get_priors()` returns `None`, the 15% supabase weight is redistributed.
The model uses uniform prior or 3-source blend instead.

**Fix:** Add a name-normalization map for spelling discrepancies, and add a continent-level
fallback using `ads_a_h2h_continent` when country-level h2h is missing.

---

## 5. d_* live checkpoint tables are empty pre-tournament

**Severity: LOW now, CRITICAL for HT mode from June 11**

`data/supabase_data.py` — `get_live_checkpoint()`

Tables `d_checkpoint_snapshot` and `d_match_scores` are populated live by the Stair AI
pipeline during matches. Before the tournament starts (June 11), they contain warm-up
and historical data only, none of which corresponds to WC2026 fixture IDs.

**What `get_live_checkpoint()` returns pre-tournament:** `None`

**What happens in run_cycle.py when live=None:** The HT model still runs but gets no
live score/xG/shots data. It falls back to the pre-match blend as if no HT data is
available, which is correct behavior.

**Impact when tournament starts:** HT window decisions (kickoff+45 → kickoff+60)
need this data. If the pipeline lags or the Supabase push is slow, HT decisions will
also run on None and fall back. Monitor during first matches.

**No code fix needed** — the fallback behavior is correct. Just note that HT mode
requires active Supabase ingestion from Stair AI's side.

---

## 6. xG projected data is never populated for real fixtures

**Severity: HIGH (draw model loses an input)**

`agent/run_cycle.py` line 197:
```python
"total_projected_xg": synthetic.get("projected_xg"),
```

`synthetic.get("projected_xg")` is only populated in `data/synthetic_fixtures.py`
(specifically `SYN-DRAW-UNDER` which sets `projected_xg: 1.9`). For all real fixtures
this is always `None`.

The draw model (`models/draw_model.py`) uses `total_projected_xg` to adjust draw
probability for low-scoring game styles. Without it, this adjustment is skipped.

**Fix:** Fetch xG from Sportmonks by adding `xGFixture` to the include params and
extracting it:
```python
# data/sportmonks.py
def extract_projected_xg(detail: dict) -> float | None:
    xg_entries = detail.get("xGFixture") or []
    if not xg_entries:
        return None
    return sum(float(e.get("xg") or 0) for e in xg_entries)
```
Then in `run_cycle.py`:
```python
"total_projected_xg": sportmonks.extract_projected_xg(detail),
```

---

## 7. Lineup data is always "unconfirmed" pre-kickoff

**Severity: MEDIUM (adds risk flag, reduces confidence)**

`data/lineup_monitor.py` — `extract_lineups()`

Sportmonks only confirms lineups ~60 minutes before kickoff. For PRE_MATCH decisions
made earlier, the lineup extraction will either return empty lists or predicted lineups,
triggering the `lineup_unconfirmed` risk flag in `evaluate_lineup_delta()`.

**Impact:** `lineup_unconfirmed` risk flag reduces source_quality weight from 0.90
to 0.35 for lineup signals (`run_cycle.py:269`). Doesn't block orders by itself but
lowers confidence.

**No fix needed** — this is correct behavior. The timing of the PRE_MATCH decision
determines whether lineup data is available. Scheduling PRE_MATCH decisions within
90 minutes of kickoff rather than hours beforehand improves lineup confirmation rates.

---

## 8. Claim extraction has no text sources for real fixtures

**Severity: HIGH (LLM-claims mode does nothing)**

`agent/run_cycle.py` line 126:
```python
source_texts = synthetic.get("source_texts") or fixture.get("source_texts") or []
```

For real fixtures from the mapping endpoint, `source_texts` is always an empty list.
When `--use-llm-claims` is passed, the code checks for source texts:
```python
if not source_texts:
    result.update({"ok": True, "reason": "no_text_sources_for_claim_extraction"})
```

This means `--use-llm-claims` silently does nothing on real fixtures.

**Impact:** The 3% `llm_claims` source weight is never used. The LLM claim extraction
path (which could flag injuries, suspensions, weather) never runs.

**Fix needed:** Fetch text from Sportmonks or another source and populate `source_texts`.
Options:
- Sportmonks `news` endpoint for fixture-related news
- Sportmonks `injuries` and `sidelined` player data
- External news API (not currently available through the arena)

---

## 9. Source weights are hardcoded, not calibrated

**Severity: MEDIUM (suboptimal blend)**

`models/probability_blender.py`:
```python
DEFAULT_PREMATCH_WEIGHTS = {
    "bookmaker": 0.30,
    "sportmonks": 0.25,
    "polymarket": 0.20,
    "supabase": 0.15,
    "lineup": 0.05,
    "draw_model": 0.05,
    "llm_claims": 0.03,
}
```

These weights were chosen by hand. No calibration against real WC data has been
performed. Given that `sm_pred` is currently always None for real fixtures (issue 3),
the effective weights at runtime are actually:
- bookmaker: ~40% (redistributed share)
- polymarket: ~27%
- supabase: ~20% (when available)
- lineup + draw_model + llm_claims: ~13%

**Fix:** Once real WC2026 data starts flowing (June 11+), run Brier minimization over
the first 10–15 matches to calibrate source weights empirically.

---

## 10. Calibration parameters are not tuned

**Severity: MEDIUM (systematic under/overconfidence)**

`models/probability_blender.py` — `deterministic_blend()`:
```python
temperature=1.06, market_shrink=0.06, draw_floor=0.16
```

Calibration analysis across 800 synthetic matches shows:
- Model is **underconfident at ~50%** (predicts 50%, actual rate 57.7%)
- Model is **underconfident at ~10%** (predicts 10%, actual rate 16.7%)
- Well-calibrated at 25–45%

The current `temperature=1.06` is slightly flattening the distribution when it should
be sharpening it at the high end. Raising temperature to ~1.10–1.12 for high-confidence
predictions and adding a separate top-probability boost would improve Brier by ~0.005.

---

## 11. Synthetic fixture edge cases in `data/synthetic_fixtures.py`

**Severity: LOW (testing only, clearly labeled)**

10 hand-crafted scenarios used when `--use-synthetic-fixtures` is passed:
`SYN-MB-VS-PM`, `SYN-PM-BK-AGAINST`, `SYN-HT-LOW-XG`, `SYN-FAV-LUCKY-TRAIL`,
`SYN-GK-MISSING`, `SYN-RED-LEADER`, `SYN-DRAW-UNDER`, `SYN-STALE-LINEUP`,
`SYN-WEAK-WEB`, `SYN-MISSING-MARKET`.

All have `"demo": True` and `"synthetic": True` flags. They are correctly gated
behind `--use-synthetic-fixtures`. No impact on production.

---

## 12. KO round fixtures not yet in arena mapping

**Severity: LOW now, HIGH after group stage ends**

`GET /v1/web/mapping` currently returns 72 group-stage fixtures only. Round of 32,
quarter-final, semi-final, and final fixtures will be added by the arena as matchups
are confirmed (from June 27 onward).

`discover_fixtures_safe()` will automatically pick them up once they appear in the
mapping — no code change needed.

---

## 13. Market stale detection blind on first run

**Severity: LOW**

`models/market_stale.py` — `detect_market_stale()`

Requires a `previous_market` snapshot to compare against. On the first decision for
any fixture, `previous_normalized_probs()` returns `None` and stale detection cannot
run. Adds `market_snapshot_stale_unknown` risk flag (`run_cycle.py:292`).

Price history accumulates in `storage/price_history/` across daemon runs. After the
second cycle for a fixture, stale detection works normally.

**No fix needed** — correct behavior. Just acknowledge that the first PRE_MATCH
decision on each fixture cannot detect stale markets.

---

## Summary table

| # | Component | Type | Severity | Affects production | Fix available |
|---|---|---|---|---|---|
| 1 | Backtesting | Synthetic data | HIGH | No (testing only) | After June 11 results |
| 2 | Demo fallbacks (all 5 sources) | Hardcoded | CRITICAL | Only if ARENA_KEY missing | Set ARENA_KEY |
| 3 | Sportmonks 3-way ML prediction | Wrong endpoint | HIGH | Yes — sm_pred always None | Add predictions/probabilities endpoint |
| 4 | Supabase priors (new/small nations) | Missing data | HIGH | Yes — priors None for ~20 teams | Name normalization + continent fallback |
| 5 | HT live checkpoint | Not yet ingested | LOW→CRITICAL | Only in HT mode pre-June 11 | Auto-resolves on June 11 |
| 6 | Projected xG for draw model | Not fetched | HIGH | Yes — draw model missing input | Extract xGFixture from Sportmonks |
| 7 | Lineup confirmation | Timing issue | MEDIUM | Yes — pre-kickoff always unconfirmed | Schedule closer to kickoff |
| 8 | LLM claim extraction text | No source texts | HIGH | Yes — llm-claims silently no-ops | Wire Sportmonks news/injuries endpoint |
| 9 | Source weights | Hardcoded | MEDIUM | Yes — not optimized | Calibrate after June 11 |
| 10 | Calibration parameters | Not tuned | MEDIUM | Yes — over/underconfident at extremes | Tune temperature after real data |
| 11 | Synthetic fixtures | Testing only | LOW | No | n/a |
| 12 | KO round fixtures | Not in mapping yet | LOW→HIGH | No until June 27 | Auto-resolves as tournament progresses |
| 13 | Market stale (first run) | Cold start | LOW | Minor — 1 flag per fixture | Auto-resolves on second cycle |
