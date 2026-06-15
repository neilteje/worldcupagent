# BZZOIRO API Integration

## Endpoints Used
The system utilizes the BZZOIRO v3 endpoints to fetch pre-match probabilities, in-game momentum, and expected goals via the `/v3/predictions` and `/v3/live` endpoints. All fetching is handled through the robust client implemented in `data/bzzoiro.py`.

## Schemas
Schemas have been strictly defined in `data/bzzoiro_schemas.py`, utilizing Pydantic to ensure all data contracts are respected. Missing expected fields will be flagged but won't crash the pipeline, providing graceful degradation.

## Authentication and Configuration
The client authenticates using a Bearer token stored in the `.env` file under `BZZOIRO_API_KEY`. It respects exponential backoff and timeouts defined in `config.py`.

## Caching
All historical and future upcoming BZZOIRO requests are cached locally on disk in `/cache/bzzoiro` using the file persistence mechanism in `data/bzzoiro_snapshots.py`. This ensures we don't spam the external provider API during historical walk-forward backtesting.

## Mapping
Event mapping between BZZOIRO events and Sportmonks fixtures is handled in `data/bzzoiro_mapper.py`, comparing canonical entity names and aligning kickoff times.

## Leakage Safeguards
The integration actively strips post-match live fields from being accidentally appended to the `PRE_MATCH` snapshots. The deterministic team state builder drops any data that falls chronologically after the `as_of_timestamp`.

## Shadow-Model Behavior
By default, the `BZZOIRO_MODEL_SHADOW_ONLY` flag is set to `True`. BZZOIRO predictions will only be collected and appended to the data-view trace, but the actual determinist model weights remain 0 unless explicitly activated via config.
