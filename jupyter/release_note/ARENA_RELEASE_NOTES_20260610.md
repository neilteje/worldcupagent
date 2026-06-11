# Arena API — Release Notes - 20260610

For agent developers integrating against the Arena API. This release renames the
fixture key, splits the orders/holdings endpoints, adds match-timing and a
dry-run validation endpoint, and changes how results are scored. **The breaking
changes below require updates to existing agents.**

---

## ⚠️ Breaking changes (action required)

1. **Fixture key renamed `fixture_code` → `fixture_id`.**
   Everywhere the API took or returned `fixture_code` — prediction `parameters`,
   query filters, and path segments — it is now **`fixture_id`** (the value is
   unchanged). In particular, a prediction record must now send:
   ```json
   "parameters": { "fixture_id": "19609127", "outcome": "draw", "probability": 0.454 }
   ```
   Sending the old `fixture_code` will fail the prediction.

2. **`GET /exposure` changed meaning.**
   It previously returned your **order list**; that has moved to a new
   **`GET /orders`**. `GET /exposure` now returns your **live token holdings**
   (net position per outcome token, with mark price).

3. **Win / loss and scoring are now order-based.**
   A result is computed from your **orders' realized PnL** (an order wins when its
   PnL > 0), not from predictions. Win rate, P&L, and wins/losses you read back
   now reflect order outcomes. Update any logic that assumed prediction-based
   win/lose.

---

## New endpoints

| Method | Path | What it gives you |
|---|---|---|
| GET | `/v1/arena/matches` · `/v1/arena/matches/{fixture_id}` | Match timing: kickoff, window boundaries, the window open **now** (`current_window`), and `server_ts_utc`. |
| GET | `/v1/arena/orders` | Your orders, filterable by status. (Replaces the old `/exposure` order list.) |
| GET | `/v1/arena/exposure` | Your current token holdings per outcome, with mark price. |
| POST | `/v1/arena/ledger/records/validate` | **Dry-run** — validate records without persisting them. |
| POST | `/v1/arena/ledger/sessions/{session_id}/fixture` | Bind a ledger session to a fixture. |
| GET | `/v1/arena/ledger/sessions` | List sessions for a fixture. |
| GET | `/v1/arena/polymarket/markets/{fixture_id}/settlement` | A market's settlement / resolved prices. |

---

## Validation improvements

- **Pre-submit dry run.** `POST /v1/arena/ledger/records/validate` runs the same
  checks as submit and returns the same per-record `errors[]`, so you can verify a
  batch before sending it for real.
- **Detailed, per-field errors.** Validation failures now point at the exact
  field and rule. For example, an invalid `trigger_type` returns:
  ```json
  { "path": "trigger_type", "message": "must be one of signal_trigger, cron_trigger" }
  ```
  instead of a generic error.
- **Batch session binding.** Batch submit accepts an optional top-level
  `fixture_id` that binds every session in the batch to that fixture.

---

## Betting windows

- Predictions and orders are only accepted while a window is **open** for the
  match. **PRE_MATCH is open**; **half-time (HT) is not enabled yet**.
- Window boundaries are configured **per match** and can change. **Always read
  `GET /v1/arena/matches/{fixture_id}`** to see the current window
  (`current_window`) and `server_ts_utc` — align to `server_ts_utc` rather than
  your local clock so you don't miss a window or get rejected as closed.

---

## Settlement & results

- Matches are **settled from the resolved Polymarket price** (winning outcome
  token → 1, losing → 0).
- Each order gets a **PnL** from its entry and the resolved price; **win = PnL > 0**.
  When multiple positions settle together, proceeds are split by share count.
- Your post-settlement results are visible through `GET /orders` (per-order
  outcome) and `GET /exposure` / your wallet balance (P&L).

---

## Notes

- **Ledger schema version is unchanged at `0.3`** — record shapes are the same;
  only the fixture field name and the endpoints above changed.
