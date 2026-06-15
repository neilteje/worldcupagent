# File-by-File Report

## Core Agent Infrastructure
- `agents/contracts.py`: Contains immutable data contracts (e.g. `FixtureDataSnapshot`, `AgentDataView`, `AgentForecast`, `MarketContext`, `DirectionalSignal`, `TradeCandidate`).
- `agents/base.py`: Defines the `AgentStrategy` Protocol specifying the interface that all agents must implement (`build_data_view`, `build_forecast`, `generate_candidates`, etc.).

## Concrete Agent Strategies
- `agents/monk.py`: Implements `MonkStrategy`, which strictly isolates market features and serves as the fundamental baseline using `models.deterministic_v2`.
- `agents/anchor.py`: Implements `AnchorStrategy`, inheriting the core `MonkStrategy` base forecast but expanding candidate generation to hunt deep value against the market.
- `agents/hunter.py`: Implements `HunterStrategy`, inheriting the `AnchorStrategy` but applying specific tail models (`models/tails/`) to seek extreme skew values instead of general value.
- `agents/blitz.py`: Implements `BlitzStrategy`, serving as a thin adapter layer over the legacy MatchForecast workflow to preserve original behavior while filtering out draw candidates.

## Live Engine Refactoring
- `live/cycle.py`: Completely overhauled to iterate through the configured Agent profiles and directly instantiate their respective `AgentStrategy` classes. Replaced the generic `select_picks` loop with `strategy.generate_candidates` and `strategy.generate_recommendations`.
- `betting/portfolio.py`: Extended to handle joint portfolio allocation (`PortfolioCoordinator`), consuming `AgentRecommendation`s from MONK, ANCHOR, and HUNTER to deduplicate exposures across strategies based on a generated `correlation_key`.
- `ledger/client.py`: Refactored to properly delineate shared fixture traces (API calls and data payloads) from individual agent traces (data views, forecasts, recommendations, allocations, and gating).

## Data and Modeling
- `data/bzzoiro.py`, `data/bzzoiro_mapper.py`, `data/bzzoiro_schemas.py`: Form the complete integration suite for fetching and securely parsing real-world football features from BZZOIRO.
- `models/team_state_builder.py`, `models/chronological_elo.py`, `models/rolling_form.py`, `models/lineup_strength.py`, `models/context_adjustments.py`: Segregated models that previously lived implicitly within the Analyst LLM prompt or the `deterministic_v2` model, allowing agents to fetch these independent statistical primitives deterministically.
- `models/live_state.py`, `models/live_update.py`: Implements a deterministic Bayesian updater for halftime forecasts, utilizing pre-match xG and live xG.
- `models/tails/upset_model.py`, `models/tails/draw_model.py`: Contain the statistical heuristics for artificially inflating extreme tail events for the HUNTER agent.

## Reasoning Council Fixes
- `reasoning/council.py`, `reasoning/prompts.py`: The fundamental prompts dictating the `Analyst` behavior were cleansed of all fields associated with `polymarket`, `kalshi`, and `sportmonks_bookmaker`, enforcing a pure baseline generation. Market context is now only exposed explicitly at the `Judge` layer.
