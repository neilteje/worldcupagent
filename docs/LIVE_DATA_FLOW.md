# Live Data Flow

This document describes the current live path used by:

```bash
python -m live run
```

The same core path is also used by:

```bash
python -m live once --fixture-id <id> --window PRE_MATCH
python -m live once --fixture-id <id> --window HT
```

## Top-Level Loop

```mermaid
flowchart TD
    A["python -m live run"] --> B["live.__main__.main"]
    B --> C["load_roster()"]
    C --> D["LiveRunner(agents, dry_run).run_forever()"]
    D --> E["refresh_schedule(): Sportmonks season schedule"]
    E --> F["pending_windows(): PRE_MATCH and HT windows"]
    F --> G{"Window due?"}
    G -- "no" --> H["sleep until next trigger"]
    H --> D
    G -- "yes" --> I["window_open(): Arena match window check"]
    I --> J{"Open / allowed?"}
    J -- "no" --> K["mark skipped/missed/failed in LiveState"]
    J -- "yes" --> L["run_window_cycle(fixture_id, window, agents)"]
    L --> M["mark window done or failed"]
    M --> N["settle_pass(): settlement + wallet/order snapshots"]
    N --> D
```

## Window Cycle Shape

`run_window_cycle()` runs one shared forecast once per fixture/window, then each agent gets its own deterministic execution tail.

```mermaid
flowchart TD
    A["run_window_cycle(fixture_id, window, agents)"] --> B{"window"}
    B -- "PRE_MATCH" --> C["gather_prematch()"]
    B -- "HT" --> D["gather_halftime()"]
    C --> E["Forecast object"]
    D --> E
    E --> F["metrics.log_event('forecast')"]
    F --> G["for each LiveAgent"]
    G --> H["act_for_agent(agent, forecast)"]
    H --> I["policy + gates + wallet cap"]
    I --> J["optional orders"]
    J --> K["agent ledger session"]
    K --> L["agent summary"]
    L --> M["run_window_cycle result"]
```

The shared `Forecast` is the handoff object. Its important normalized outputs are:

| Field | Meaning |
| --- | --- |
| `probabilities` | Full 3-way probability map: `{home_code, draw, away_code}`. |
| `outcome` | Highest-probability outcome. |
| `probability` | Probability of `outcome`. |
| `confidence` | `low`, `medium`, or `high`. |
| `engine` | Forecast engine label, e.g. `council_with_deterministic_v2` or `ht_bayesian_llm`. |
| `grounding` | Council grounding plus deterministic context for PRE_MATCH. |
| `summary` | Human-readable rationale from the final model step. |
| `moneyline` / `mids` | Market data used by policy/order selection. |

## PRE_MATCH Flow

PRE_MATCH is the main integrated path. The deterministic engine runs before the LLM council. Its output and generated signals are passed into the council roles. The final forecast is the council result, not a separate arbiter.

```mermaid
flowchart TD
    A["gather_prematch(fixture_id)"] --> B["Sportmonks fixture"]
    B --> C["participants: home/away names + codes"]
    C --> D["_fetch_market(): Polymarket slug, moneyline, mids"]
    C --> E["web_search.gather_research()"]
    C --> F["reddit_sentiment.get_sentiment_bundle()"]
    C --> G["kalshi.get_moneyline()"]
    C --> H["fixture_bundle.build_context()"]
    H --> I["sportmonks_digest"]
    H --> J["supabase_digest"]
    D --> K["llm.digest_polymarket() or no_market fallback"]

    B --> L["_deterministic_context_for_council()"]
    D --> L
    I --> L
    L --> M["deterministic_context"]

    I --> N["council.run_council()"]
    J --> N
    K --> N
    E --> N
    F --> N
    G --> N
    M --> N

    N --> O["CouncilResult"]
    O --> P["Forecast.probabilities/outcome/confidence"]
    M --> Q["Forecast.grounding.deterministic_context"]
    P --> R["Forecast.engine = council_with_deterministic_v2"]
```

### Deterministic Engine Inputs

The deterministic path is built in `live.cycle._deterministic_v2_model()`.

| Input | Source | Use |
| --- | --- | --- |
| Polymarket mids | `_fetch_market()` | Preferred market prior if available. |
| Sportmonks bookmaker probabilities | `sportmonks.extract_bookmaker_probs()` | Fallback prior if no Polymarket prior. |
| Sportmonks ML probabilities | `sportmonks.extract_ml_probabilities()` | Fallback prior if no bookmaker prior. |
| Neutral prior | Hard-coded fallback | Used when no prior exists. |
| Stage name | Sportmonks fixture | Sets knockout/group context for deterministic_v2. |

### Deterministic Engine Outputs

The deterministic model output is stored on `Forecast.deterministic_model` and is also transformed into `deterministic_context` for the council.

| Output | Meaning |
| --- | --- |
| `probabilities` | Deterministic H/D/A probability distribution. |
| `expected_goals` | Deterministic expected goals signal. |
| `components` | Component model distributions/signals from deterministic_v2. |
| `weights` / `component_weights` | Blend weights used by deterministic_v2. |
| `blended_raw` | Raw blended distribution before final context wrapping. |
| `confidence` | Deterministic confidence score. |
| `uncertainty` | Derived uncertainty band. |
| `risk_flags` | Flags such as neutral cold start. |
| `prior_source` / `prior_hda` | Which prior was used and its normalized H/D/A map. |
| `home_state` / `away_state` | Cold-start team state passed into deterministic_v2. |
| `config` | Deterministic_v2 ensemble config. |

### LLM Council Integration

```mermaid
flowchart LR
    A["deterministic_context"] --> B["Scout"]
    A --> C["Analyst"]
    A --> D["Devil"]
    A --> E["Judge"]

    F["Sportmonks digest"] --> B
    F --> C
    F --> D
    G["Supabase digest"] --> C
    G --> D
    H["Web research"] --> B
    I["Reddit bundle"] --> B
    J["Grok social pulse"] --> B
    B --> C
    C --> D
    C --> E
    D --> E
    K["Polymarket digest"] --> E
    L["Kalshi moneyline"] --> E
    E --> M["Final CouncilResult"]
```

Role outputs:

| Role | Inputs | Output |
| --- | --- | --- |
| Social pulse | Fixture, teams, kickoff | Live/news/social pulse JSON. |
| Scout | Sportmonks digest, deterministic context, web, Reddit, social pulse | Flags, crowd lean, data quality. |
| Analyst | Sportmonks digest, Supabase digest, scout flags, anchor, deterministic context | Market-blind 3-way probabilities. |
| Devil | Analyst output, Sportmonks/Supabase digests, deterministic context | Counter-case and counter-probabilities. |
| Judge | Analyst, Devil, Polymarket, Kalshi, deterministic context | Final calibrated probabilities. |

The judge output becomes the PRE_MATCH forecast:

```text
Forecast.probabilities = CouncilResult.probabilities
Forecast.outcome       = CouncilResult.outcome
Forecast.probability   = CouncilResult.probability
Forecast.confidence    = CouncilResult.confidence
Forecast.engine        = "council_with_deterministic_v2"
```

## HT Flow

HT uses the existing half-time Bayesian LLM update path. It does not run the PRE_MATCH council again.

```mermaid
flowchart TD
    A["gather_halftime(fixture_id, prematch_note)"] --> B["Sportmonks fixture"]
    B --> C["extract_ht_stats()"]
    A --> D["Supabase HT snapshot"]
    A --> E["Supabase HT score"]
    A --> F["_fetch_market(): Polymarket moneyline/mids"]
    F --> G["llm.digest_polymarket() or no_market fallback"]
    C --> H["llm.ht_predict()"]
    D --> H
    E --> H
    G --> H
    I["prematch_note from state"] --> H
    H --> J["single outcome + probability"]
    J --> K["Rebuild 3-way map from residual mass"]
    K --> L["Forecast.engine = ht_bayesian_llm"]
```

HT forecast outputs are the same normalized `Forecast` fields consumed by policy and orders.

## Per-Agent Tail

After the shared forecast is built, every configured agent runs the same tail with its own profile, wallet, API key, orders, and ledger.

```mermaid
flowchart TD
    A["Forecast"] --> B["act_for_agent(agent, fx)"]
    B --> C["ArenaClient.wallet()"]
    A --> D["betting.policy.select_picks()"]
    C --> D
    D --> E["gates.evaluate_gates()"]
    E --> F["wallet/cycle cap"]
    F --> G{"picks?"}
    G -- "no" --> H["predict-only ledger"]
    G -- "yes" --> I{"dry_run?"}
    I -- "yes" --> J["simulated orders"]
    I -- "no" --> K["ArenaClient.place_order() + poll_order()"]
    H --> L["LedgerSession records"]
    J --> L
    K --> L
    L --> M["session.submit() or validate()"]
    M --> N["agent summary"]
```

Policy inputs:

| Input | Source |
| --- | --- |
| Forecast probabilities | Council+deterministic PRE_MATCH or HT Bayesian update. |
| Market moneyline/mids | Polymarket fetch. |
| Agent profile | `live.roster` / `harness.profiles`. |
| Wallet balance | Arena wallet endpoint. |
| Confidence | Forecast confidence converted with `confidence_to_num()`. |
| Scout flags | PRE_MATCH council scout flags, empty for HT unless populated upstream. |

Agent outputs:

| Output | Meaning |
| --- | --- |
| `prediction` | Outcome, probability, full probabilities, confidence, engine. |
| `orders` | Placed/simulated order summaries. |
| `skip_reasons` | Why no trade or why picks were filtered. |
| `ledger` | Ledger submit/validate result. |
| `session_id` | Per-agent reasoning-ledger session id. |

## Ledger DAG

PRE_MATCH ledger records include deterministic_v2 as an explicit upstream reasoning step.

```mermaid
flowchart TD
    A["trigger: live-runner"] --> B["planning"]
    A --> C["tool: sportmonks"]
    A --> D["tool: polymarket"]
    A --> E["tool: kalshi"]
    C --> F["thinking: DETERMINISTIC_V2"]
    D --> F
    A --> G["tool: web_search"]
    A --> H["tool: reddit"]
    F --> I["thinking: SCOUT"]
    G --> I
    H --> I
    C --> J["thinking: ANALYST"]
    F --> J
    I --> J
    J --> K["thinking: DEVIL"]
    J --> L["thinking: JUDGE final forecast"]
    K --> L
    D --> L
    E --> L
    F --> L
    L --> M["acting_prediction"]
    L --> N["thinking: deterministic policy/gates/sizing"]
    D --> N
    N --> O["acting_order records"]
    L --> P["reflecting"]
    N --> P
    P --> Q["ledger submit/validate"]
```

HT ledger records use the HT predictor as the final forecast step:

```mermaid
flowchart TD
    A["trigger + planning"] --> B["tool: sportmonks"]
    A --> C["tool: polymarket"]
    A --> D["tool: kalshi"]
    B --> E["thinking: HT_PREDICT_SYS"]
    C --> E
    D --> E
    E --> F["acting_prediction"]
    E --> G["thinking: deterministic policy/gates/sizing"]
    G --> H["orders if selected"]
    E --> I["reflecting"]
    G --> I
```

## Metrics And State Outputs

| Output | Writer | Contents |
| --- | --- | --- |
| `metrics.log_event("forecast")` | `run_window_cycle()` | Fixture, window, engine, probabilities, confidence, market source, grounding, summary. |
| `metrics.log_event("agent_window")` | `act_for_agent()` | Agent profile, engine, prediction, wallet, orders, skip reasons, ledger result. |
| `storage/live/state.json` | `LiveState` | Window status, agent summaries, retries, settlement status. |
| Ledger API | `LedgerSession` | Per-agent reasoning DAG and action trace. |
| Arena orders | `ArenaClient.place_order()` | Live or simulated orders depending on `--dry-run`. |

## Current Engine Summary

| Window | Forecast engine | Deterministic_v2 used? | Final forecast owner |
| --- | --- | --- | --- |
| `PRE_MATCH` | `council_with_deterministic_v2` | Yes. Full deterministic context is passed to Scout, Analyst, Devil, and Judge. | LLM council Judge result after grounding. |
| `HT` | `ht_bayesian_llm` | No separate deterministic_v2 run in the current HT path. | HT Bayesian LLM update, then residual 3-way reconstruction. |

