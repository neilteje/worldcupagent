# LIVE.md — running the 4-agent portfolio through the World Cup

This is the runbook for the **live arena runner** (`live/`): one process that
carries all four agents (monk/ORACLE, anchor/KEEL, hunter/SAW, blitz/SURGE)
from the opening match to the final, on a single VM, with kill-anytime
resumability and full retrospective logging.

---

## What changed with the 2026-06-10 arena release (already implemented)

The organizers' notes in `jupyter/release_note/` contain **breaking changes**;
this codebase has been updated for all of them:

| Change | Where we handle it |
|---|---|
| `fixture_code` → **`fixture_id`** in prediction params + order payloads (old name = rejected) | `ledger/client.py`, `live/arena_client.py`, `agent.py` |
| Wallet shape: `agents/me` → `wallet.available_balance_usdc` / `locked_balance_usdc` | `ArenaClient.wallet()` (legacy fallback kept) |
| `GET /exposure` = **holdings**; orders moved to `GET /orders` | `ArenaClient.exposure()` / `.orders()` |
| Win/loss + scoring now **order-based** (PnL > 0 = win) | settlement watcher + `live report` |
| New `GET /v1/arena/matches/{id}` — window state + `server_ts_utc` | runner verifies every window server-side before firing |
| **HT window not enabled yet** | runner polls; if HT never opens it records `skipped` and moves on — and starts trading HT automatically the day the arena enables it |
| Ledger: dry-run `…/records/validate`, batch `fixture_id` session binding, per-order Acting records, payload persisted before POST | `LedgerSession.submit()` + `live/cycle.py` |
| Schema v0.3 strictness: Planning needs `goal`+`steps`, Reflecting needs `inputs`, ModelInvocation needs non-empty provider/model | `ledger/client.py` (scout/gate records are now Thinking; every session has a real Planning record) |
| Sizing: cap at **min($5, balance − $0.05)**, $1 CLOB per-order minimum | `live/cycle.py` + `betting/policy.py` |

## How one window runs

```
trigger (kickoff−45' or kickoff+46')
   └─ confirm window open via GET /v1/arena/matches/{id}   (server time)
   └─ SHARED BRAIN — runs ONCE per window:
        Sportmonks fixture + digests · Polymarket slug/mids · Kalshi
        web research · Reddit · Grok pulse
        council (Scout → Analyst → Devil → Judge) + grounding layer
   └─ PER AGENT (×4, each under its own API key):
        wallet → betting/policy.select_picks(profile) → gates overlay
        → orders POSTed + polled to terminal state
        → full ledger DAG (Planning + ToolCalling + Thinking + Acting
          prediction + Acting per order + Reflecting) → validate → batch submit
   └─ state.json marked done  ·  events.jsonl appended
```

The four agents deliberately share one forecast (the prediction is identical;
the *policy* differs). Cost: ~8 LLM calls per window total, not per agent.

## Setup (once, on the DigitalOcean VM)

```bash
git clone <repo> && cd worldcupagent
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`.env` — mint **4 production keys** at https://stair-ai.com/api-keys
(1 key = 1 arena agent; multi-agent is explicitly sanctioned):

```bash
AGENT_KEY_MONK=...      # oracle — forecast specialist (Score track)
AGENT_KEY_ANCHOR=...    # keel   — disciplined EV accumulator
AGENT_KEY_HUNTER=...    # saw    — skew harvester (draws/dogs ≤ 0.40)
AGENT_KEY_BLITZ=...     # surge  — event-driven aggression
ANTHROPIC_API_KEY=...   # + any other LLM keys you use (XAI_KEY, DEEPSEEK_KEY…)
# STAIR_API_KEY=...     # optional — only needed for single-agent runs without
                        # AGENT_KEY_* set. If omitted, data proxies (Sportmonks,
                        # Polymarket) borrow AGENT_KEY_ANCHOR automatically.
```

Verify everything before match day:

```bash
python -m live test                 # all 4 keys + schedule + matches endpoint
python -m live once --fixture-id 19609127 --dry-run   # full pipeline, no writes
```

`--dry-run` runs data + council + policy and *builds* the ledgers (validating
them via the dry-run endpoint) but places no orders and submits nothing.

## Run it

```bash
python -m live run
```

That's the whole tournament. For a VM, run it under systemd so it survives
SSH disconnects and reboots:

```ini
# /etc/systemd/system/worldcup.service
[Unit]
Description=World Cup 4-agent arena runner
After=network-online.target

[Service]
WorkingDirectory=/root/worldcupagent
ExecStart=/root/worldcupagent/.venv/bin/python -m live run
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable --now worldcup
journalctl -u worldcup -f          # tail the logs
```

(`tmux` / `nohup python -m live run >> storage/live/runner.log 2>&1 &` work
too — the loop itself doesn't care how it's hosted.)

### Stop / restart semantics

State lives in `storage/live/state.json`, written atomically after every
window. Kill the process at any moment; on restart it:

- skips every `(fixture, window)` already `done` / `missed` / `skipped`,
- retries `failed` windows (up to 3 attempts) **while their window is still
  open**, marking them `missed` once it isn't,
- continues the settlement watcher for anything unresolved.

It never restarts from game 1. To reset deliberately, delete
`storage/live/state.json` (and `events.jsonl` if you want clean metrics).

## Watching it

```bash
python -m live status      # done / pending counts + next fixtures
python -m live report      # retrospective (also writes storage/live/report/)
tail -f storage/live/events.jsonl | python -m json.tool   # raw event stream
```

## Retrospective evaluation

Everything needed to improve the agents later is logged as it happens to
`storage/live/events.jsonl`:

| event | contents |
|---|---|
| `forecast` | full 3-way distribution, confidence, grounding audit (anchor, shrink λ, sanity flags), market mids at decision time, scout flags |
| `agent_window` | per agent: picks **and skip reasons**, wallet, order outcomes (status/fills/tx), ledger result |
| `settlement` / `agent_settlement` | resolved winner, per-agent orders + wallet snapshot after each match |
| `missed_window` / `skipped_window` / `error` | every gap, with reasons |

`python -m live report` turns that into the STRATEGY.md §6 measurements:
per-agent P&L scorecard, **council Brier vs de-vigged market Brier (M-1)**,
**anchor-divergence win rate (M-2)**, fire rates, and top skip reasons — with
the written pivot triggers in `docs/STRATEGY.md` deciding what to change.

## Knobs

| What | Where |
|---|---|
| PRE_MATCH lead (default kickoff − 45') | `live/runner.py::PRE_LEAD_MIN` |
| HT trigger / deadline | `HT_OFFSET_MIN` / `HT_DEADLINE_MIN` |
| Agent policies (edge bars, Kelly, skew filter, caps) | `harness/profiles.py` — single source of truth for harness + arena + live |
| Run a subset of agents | `python -m live run --agents anchor,hunter` |
| Per-cycle spend ceiling | `config.MAX_BET_USD` ($5, arena rule) + wallet buffer in `live/cycle.py` |
