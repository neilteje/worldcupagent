# Data Sources Reference

All data used by the agent. Verified against live endpoints 2026-06-06.

---

## 1. Arena API

**Base URL:** `https://staging.stair-ai.com/api`  
**Auth:** `x-api-key: <ARENA_KEY>` on every request  
**Config:** `agent/config.py` → `Settings.arena_api`, `Settings.headers`

### 1.1 Agent Identity

```
GET /v1/arena/agents/me
```

Returns agent profile and wallet.

```json
{
  "agent_id": "1d100fe4-...",
  "display_name": "CHI CHI CHILE",
  "lifecycle_phase": "active",
  "wallet": {
    "available_balance_usdc": "5",
    "locked_balance_usdc": "0",
    "polymarket_profile_url": "https://polymarket.com/profile/0x..."
  }
}
```

### 1.2 Open Positions

```
GET /v1/arena/exposure
```

Returns all currently open positions and locked USD. `positions: []` when no bets are placed.

### 1.3 Fixture Mapping

```
GET /v1/web/mapping               -- returns ALL 72 group-stage fixtures
GET /v1/web/mapping?fixture_id=N  -- returns single fixture (often empty; use no-param form)
```

**This is the primary fixture discovery mechanism.** Each entry includes:

| Field | Example | Use |
|---|---|---|
| `sportmonks_fixture_id` | `"19609127"` | Sportmonks API calls, Supabase match_id |
| `sportmonks_match_name` | `"Mexico vs South Africa"` | Display |
| `sportmonks_kickoff_utc` | `1781204400000` | Unix ms UTC, compare to now for scheduling |
| `home_country` / `away_country` | `"Mexico"` / `"South Africa"` | Supabase priors lookup |
| `home_short_code` / `away_short_code` | `"MEX"` / `"ZAF"` | Order `team_code` field |
| `polymarket_event_slug` | `"fifwc-mex-rsa-2026-06-11"` | Gamma API (deprecated), reference |
| `polymarket_home_token_yes` | `"207790..."` | CLOB midpoint lookup |
| `polymarket_draw_token_yes` | `"116341..."` | CLOB midpoint lookup |
| `polymarket_away_token_yes` | `"115307..."` | CLOB midpoint lookup |
| `match_confidence` | `"high"` | Mapping reliability flag |

**Code:** `data/polymarket.py` → `get_all_mappings()`

### 1.4 Order Submission

```
POST /v1/arena/orders
```

```json
{
  "fixture_code": "WC2026-GS-M1",
  "team_code": "MEX",
  "usd_size": "4.00",
  "limit_price": 0.55,
  "time_in_force_seconds": 30,
  "idempotency_key": "<uuid>"
}
```

`team_code` = `home_short_code`, `away_short_code`, or `"DRAW"`.  
`limit_price` = max price willing to pay (0–1). Use CLOB mid as ceiling.

**Code:** `agent/run_cycle.py` → `_safe_order()`

### 1.5 Order Status

```
GET /v1/arena/orders/{order_id}
```

**Status: Live.** Returns `{"code": "NOT_FOUND", "status": 404}` for unknown UUIDs.  
The polling loop in `run_cycle.py:56-75` works correctly once a real order is submitted.

### 1.6 Reasoning Ledger

```
POST /v1/arena/ledger/records/batch
```

Submits the reasoning trace (9+ records per session). Scored for the leaderboard.  
**Code:** `ledger/client.py` → `LedgerSession.submit()`

---

## 2. Sportmonks (via Arena Proxy)

**Proxy prefix:** `GET /v1/data/proxy/sportmonks/v3/football/<path>`  
**Same auth:** `x-api-key: <ARENA_KEY>`  
**Response envelope:** every response wraps the real payload in `body.data`

```python
envelope["body"]["data"]  # real payload
```

**Code:** `data/sportmonks.py` → all functions

### 2.1 Fixture Detail (primary data source)

```
GET /v1/data/proxy/sportmonks/v3/football/fixtures/{fixture_id}
    ?include=predictions;odds;participants
```

Returns the full fixture record including:

| Include | Content | Used for |
|---|---|---|
| `participants` | Team name, short_code, country_id, location (home/away) | Team identification |
| `predictions` | Binary win/no-win by type_id | Sportmonks ML signal |
| `odds` | 2000+ bookmaker 1X2 entries | Bookmaker probability extraction |
| `xGFixture` | Pre-match xG projections (if available) | xG signal |

**Predictions note:** type_id 235 returns `{yes, no}` (binary, not 3-way). For 3-way ML prediction, use `predictions/probabilities/fixtures/{id}` endpoint or extract from odds.

**Code:** `data/sportmonks.py` → `get_fixture()`, `extract_bookmaker_probs()`, `extract_sportmonks_prediction()`

### 2.2 Live Scores

```
GET /v1/data/proxy/sportmonks/v3/football/livescores/inplay
    ?per_page=25
```

Returns fixtures currently in-play. Used for HT window detection.

### 2.3 Season Metadata

```
GET /v1/data/proxy/sportmonks/v3/football/seasons/26618
```

WC2026 season: starts 2026-06-11, ends 2026-07-19, `is_current: true`.  
**WC2026 season_id = 26618** (hardcoded in `config.py`).

### 2.4 Fixtures by Season (filter format)

```
GET /v1/data/proxy/sportmonks/v3/football/fixtures
    ?filter=seasonId:26618&per_page=200
```

**Note:** The correct param is `filter` (singular), not `filters`. Filter format is `seasonId:NNNN`.

---

## 3. Polymarket (via Arena Proxy)

**Gamma proxy:** `/v1/data/proxy/polymarket-gamma/...`  
**CLOB proxy:** `/v1/data/proxy/polymarket-clob/...`

### 3.1 CLOB Midpoint (primary — use this)

```
GET /v1/data/proxy/polymarket-clob/midpoint?token_id=<YES_token_id>
```

Response envelope: `body.mid` (float, 0–1). Returns `body.error` for unknown tokens.

**Best practice:** Get token IDs from the mapping endpoint (Section 1.3), then call CLOB directly. This avoids the deprecated Gamma API entirely.

**Code:** `data/polymarket.py` → `get_three_way_from_tokens()`, `get_three_way_market_probs(tokens=...)`

### 3.2 Gamma Events (deprecated — avoid)

```
GET /v1/data/proxy/polymarket-gamma/events?slug=<slug>
```

Returns a deprecation header: `warning: 299 - "use /events/keyset"`. Often returns empty `[]`.  
Use the mapping token IDs and CLOB directly instead.

---

## 4. Supabase (direct, no arena proxy)

**Base URL:** `https://ezvbmtvrvzageqixvdak.supabase.co/rest/v1`  
**Auth:** `apikey: <SUPABASE_KEY>` + `Authorization: Bearer <SUPABASE_KEY>`  
**Shared key:** `sb_publishable__m8bOkD05ToFwATpaWST5w_2-3fGS7V` (hardcoded, no setup needed)

Two schemas:
- **`public`** — catalog tables (`catalog_tables`, `catalog_full`, `catalog_columns`); no `Accept-Profile` header
- **`world_cup_arena`** — all data tables; send `Accept-Profile: world_cup_arena`

**Code:** `data/supabase_data.py`

### 4.1 Catalog

```
GET /catalog_full?limit=100       (public schema, no Accept-Profile)
```

Returns all tables with descriptions and row counts. Use this to explore the schema.

### 4.2 Historical Priors (ads_a_* tables)

All keyed by StatsBomb internal `country_id` (NOT Sportmonks country_id).  
Use `country_name_a` / `country_name_b` string matching (`ilike`) for lookups.

#### ads_a_h2h_country (1484 rows)

Head-to-head records between every pair of countries. The primary prior signal.

| Column | Type | Description |
|---|---|---|
| `country_id_a` | int | StatsBomb country ID of team A |
| `country_id_b` | int | StatsBomb country ID of team B |
| `country_name_a` | text | Team A name (use for ilike lookup) |
| `country_name_b` | text | Team B name |
| `match_scope` | text | `"all"` = all competitions |
| `total_matches` | int | Raw count of matches |
| `wins_a_raw` | int | Raw wins for team A |
| `draws_raw` | int | Raw draws |
| `wins_a_weighted` | float | Recency-weighted wins (more weight to recent) |
| `draws_weighted` | float | Recency-weighted draws |
| `losses_a_weighted` | float | Recency-weighted losses |
| `total_weight` | float | Sum of weights (denominator) |
| `win_rate_a_weighted` | float | wins_a_weighted / total_weight |
| `last_meeting_date` | date | Date of most recent h2h match |

**Query example:**
```
GET /ads_a_h2h_country
    ?country_name_a=ilike.Mexico&country_name_b=ilike.South Africa
    &match_scope=eq.all&select=wins_a_weighted,draws_weighted,...
    Accept-Profile: world_cup_arena
```

#### ads_a_stage_record (180 rows)

Win/draw/loss rates for each country at each tournament stage.

| Column | Type | Description |
|---|---|---|
| `country_id` | int | StatsBomb country ID |
| `stage_canonical` | text | `group`, `round_of_32`, `round_of_16`, `quarter_final`, `semi_final`, `final` |
| `matches` | int | Total matches at this stage |
| `wins` | int | |
| `draws` | int | |
| `losses` | int | |
| `win_rate` | float | wins / matches |

#### ads_a_country_style (71 rows)

Goal-scoring and style metrics.

| Column | Type | Description |
|---|---|---|
| `country_id` | int | |
| `set_piece_shots` | int | Set-piece shot count |
| `set_piece_goals` | int | Set-piece goal count |
| `conversion_rate` | float | set_piece_goals / set_piece_shots |
| `group_gpg` | float | Goals per game at group stage (may be null) |
| `ko_gpg` | float | Goals per game at KO stage (may be null) |

#### ads_a_ko_pattern (71 rows)

Knockout stage exit patterns.

| Column | Type | Description |
|---|---|---|
| `country_id` | int | |
| `tournaments_reached_ko` | int | |
| `first_ko_wins` | int | |
| `first_ko_loss_rate` | float | |
| `modal_exit_stage` | text | Most common stage of elimination |

#### ads_a_special_match (36 rows)

Extra time and penalty shootout history.

| Column | Type | Description |
|---|---|---|
| `country_id` | int | |
| `et_matches` | int | Extra time appearances |
| `et_win_rate` | float | |
| `pen_shootouts` | int | Penalty shootout appearances |
| `pen_win_rate` | float | |

#### ads_a_manager (206 rows)

Manager-level coaching stats.

| Column | Type | Description |
|---|---|---|
| `manager_id` | int | |
| `manager_name` | text | |
| `tenure_years` | float | |
| `tournaments_coached_count` | int | |
| `ko_matches` | int | |
| `ko_win_rate` | float | |

### 4.3 Live Checkpoint Data (d_* tables)

Populated live during the tournament. Keyed by Sportmonks fixture integer ID (`match_id`).  
Pre-tournament these tables contain warm-up/historical data only.

#### d_match_scores (206 rows, grows live)

| Column | Type | Description |
|---|---|---|
| `match_id` | int | Sportmonks fixture integer ID |
| `checkpoint_code` | text | `HT`, `FT`, `ET1`, `ET2` |
| `home_team_id` | int | Sportmonks team ID |
| `away_team_id` | int | Sportmonks team ID |
| `home_goals` | int | Goals at checkpoint |
| `away_goals` | int | Goals at checkpoint |
| `pulled_at` | timestamp | When data was ingested |

**Query for HT score:**
```
GET /d_match_scores?match_id=eq.19609127&checkpoint_code=eq.HT&select=home_goals,away_goals
Accept-Profile: world_cup_arena
```

#### d_checkpoint_snapshot (260 rows, grows live)

Rich per-team stats at each checkpoint. Two rows per match (one home, one away).

| Column | Type | Description |
|---|---|---|
| `match_id` | int | Sportmonks fixture ID |
| `checkpoint_code` | text | `HT` / `FT` / `ET1` / `ET2` |
| `team_id` | int | Sportmonks team ID |
| `is_home` | bool | True = home team |
| `cum_xg` | float | Cumulative xG (may be null) |
| `raw_cum_shots_total` | int | Total shots |
| `raw_cum_shots_on_target` | int | Shots on target |
| `raw_cum_possession_pct` | int | Possession percentage |
| `cum_red_cards` | int | Red cards |
| `cum_yellow_cards` | int | Yellow cards |
| `cum_subs_used` | int | Substitutions used |
| `seg_xg` | float | xG this segment only |

#### dim_match (65 rows)

Match dimension table with full metadata.

| Column | Type | Description |
|---|---|---|
| `match_id` | int | Sportmonks fixture ID |
| `kickoff_at` | timestamp | Match start time |
| `competition_name` | text | e.g. `"World Cup"` |
| `team_id_a` | int | Sportmonks team ID |
| `team_name_a` | text | |
| `country_id_a` | int | Sportmonks country ID |
| `stage_canonical` | text | `group`, `round_of_16`, etc. |
| `is_knockout` | bool | |

---

## 5. Data Flow Summary

```
Arena mapping (no params)
    |
    +--> sportmonks_fixture_id  --> Sportmonks proxy --> fixture detail
    |                                                        |
    |                                                        +--> bookmaker odds
    |                                                        +--> predictions
    |                                                        +--> participants (team names)
    |
    +--> home_country/away_country --> Supabase ads_a_h2h_country --> h2h prior
    |                              --> Supabase ads_a_stage_record --> stage win rates
    |
    +--> home/draw/away token IDs --> Polymarket CLOB --> live market prices
    |
    +--> sportmonks_kickoff_utc --> scheduling logic (PRE_MATCH vs HT window)
```

---

## 6. Known Gaps & Limitations

| Gap | Impact | Workaround |
|---|---|---|
| Gamma API deprecated (`/events?slug=`) | Returns empty | Use mapping tokens + CLOB directly |
| No order read endpoint history | Can't audit past orders via API | Check `storage/decisions/*.json` locally |
| Supabase country IDs ≠ Sportmonks IDs | Can't join ads_a_* directly with Sportmonks fixture data | Use `country_name` string match with `ilike` |
| d_* tables pre-tournament have warm-up data only | `get_live_checkpoint()` returns None pre-kickoff | Expected behavior; HT mode only runs post-kickoff |
| Sportmonks predictions type_id=235 is binary | Not a 3-way forecast | Use bookmaker odds (`extract_bookmaker_probs`) or dedicated endpoint `predictions/probabilities/fixtures/{id}` |
| No KO-round fixtures in mapping yet | 72 group-stage fixtures mapped; KO fixtures added as tournament progresses | Re-call `get_all_mappings()` each session |
