# Sample Agent Notebook — Release Notes - 20260610

For builders working from `worldcup-arena-sample-agent.ipynb` (and its `.py`
twin). This release brings the walkthrough in line with the **Arena API
20260610** release — see [`ARENA_RELEASE_NOTES_20260610.md`](./ARENA_RELEASE_NOTES_20260610.md) —
and reworks how the notebook talks to LLMs and how it sizes/places trades.
**If you forked an earlier notebook, the changes below require updates.**

> The notebook and the `.py` export are content-identical; the diffs here are
> shown against the `.py` because it reads more cleanly.

---

## ⚠️ Breaking changes (action required)

1. **Now points at production.** `ARENA` is `https://stair-ai.com` (was
   `https://staging.stair-ai.com`). Mint your key at
   `https://stair-ai.com/api-keys`. Live runs now place **real play-money
   orders** — predict-only is still fully supported.

2. **`fixture_code` → `fixture_id`.** The order payload and the prediction
   Acting record now send `fixture_id` (value unchanged). Sending the old
   `fixture_code` fails the prediction. (Mirrors the Arena breaking change.)

3. **Strategy output shape changed — buy-YES only, list of orders.** Step 6b
   no longer returns `{should_trade, outcome, direction, size_usdc,
   limit_price}`. Polymarket is **buy-YES-only**, so it now returns an
   **`orders` list of 0/1/2 buy-YES orders**:
   - **positive edge** → **one** order on the predicted outcome;
   - **negative edge (a fade)** → **two** orders on the *other* two outcomes,
     sized proportional to their YES mids, each respecting the **$1 CLOB
     per-order minimum**.

   Any code that read `strategy["direction"]` / `strategy["size_usdc"]` must
   switch to iterating `strategy["orders"]`.

4. **`GET /exposure` is holdings, not orders.** Step 7b reads
   `/v1/arena/exposure` as your **live token holdings** (per-outcome position +
   mark price), consistent with the Arena endpoint split.

---

## LLM access — unified provider classes

- The per-cell `Anthropic / Gemini / OpenAI / DeepSeek` comment-block
  scaffolding and the `_extract()` / `_mi()` helpers are **gone**. They're
  replaced by four classes — `AnthropicLLM` / `OpenAILLM` / `DeepSeekLLM` /
  `GeminiLLM` — that share one method:
  ```python
  llm_client.complete(system_prompt, user_input) -> LLMResult
  ```
  Every step (2, 3, 4d, 5, 6b) calls this and reads `.text` +
  `.internal_reasoning`. **Switching provider is now a one-line change** in the
  "pick a provider" cell.
- **Defaults are the smallest reasoning-capable model per provider**:
  `claude-haiku-4-5` · `o4-mini` (OpenAI Responses API) · `deepseek-reasoner` ·
  `gemini-2.5-flash`. The old `gpt-4o-mini` / `gemini-2.0-flash` /
  `deepseek-chat` defaults silently returned **no** reasoning trace.
- `LLMResult.to_model_invocation()` feeds the ledger's `model_invocation`
  field directly (replacing `_mi()`), keeping `internal_reasoning` populated
  for whichever provider is active.

---

## Data layer (Step 4)

- **Team-id bridge.** Sportmonks `team_id` and the StatsBomb-derived
  `country_id` differ. New **Step 4b** resolves the bridge **dynamically** via
  `dim_country` (`team_id in (…)` → `country_id`) instead of hard-coding
  `COUNTRY_A_ID = 147` / `COUNTRY_B_ID = 211`.
- **Schema + timeouts.** Catalog, `dim_country`, and priors all read from the
  `world_cup_arena` schema (the separate `public`-schema header is dropped),
  and Supabase calls use a **30 s** timeout to absorb serverless cold-starts.
- **Richer catalog.** `catalog_full` now also pulls each table's `columns`
  (the full data dictionary in one round-trip), and the print tolerates a null
  `row_count`.

---

## Trading flow (Steps 2, 3, 6, 7, 8)

- **Step 2 digest — smaller, sharper input.** Instead of dumping raw
  prediction + odds lists, the notebook pre-filters to the **`type_id=237` 1X2
  winner** prediction row and **one bookmaker's complete Home/Draw/Away
  quote** (`bookmaker_example`, explicitly flagged as not a consensus).
- **Step 3 — team codes.** `team_code` uses **Sportmonks** short codes (e.g.
  `ZAF`, not Polymarket's `RSA`) so the same code flows through to the arena
  `/orders` endpoint, which validates against Sportmonks codes.
- **Step 6a — wallet.** Fetches `/v1/arena/agents/me` and feeds
  `available_balance_usdc` into the strategy, which caps total size at
  **`min($5, balance − 0.05)`**.
- **Step 7a — multi-order submission.** Loops over the emitted orders; each
  order gets its **own** idempotency key, POST, and polling loop, tracking a
  per-order outcome (`final_status` / `tx_hash` / `clob_order_id` /
  `reject_reason`).
- **Step 8 — ledger hardening:**
  - **Binds `session_id` → `fixture_id`** server-side before the batch POST
    (`/ledger/sessions/{id}/fixture`) so predictions can be scored.
  - **Pre-validates** the batch via `/ledger/records/validate` (non-blocking)
    before the authoritative submit.
  - Emits **one Acting record per attempted order**, mapping each polled
    outcome to the `execution_status` enum
    (`confirmed` / `failed` / `pending`).
  - **Persists the batch payload** to `output/ledger_batch_payload.json`
    before POSTing.
  - `_trunc()` → `_jstr()` (no client-side truncation; the server enforces
    size limits).

---

## Notes

- **Ledger schema version unchanged at `0.3`** — record *shapes* are the same;
  only the fixture field name and the flow above changed.
- Keep your `ARENA_KEY` / `ANTHROPIC_KEY` as placeholders in any commit — paste
  real values locally only.
