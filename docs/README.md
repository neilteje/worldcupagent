# World Cup Agent — Docs

This repo runs a **4-agent portfolio** in the Stair AI World Cup Arena. Every
agent shares one brain — data ingestion → LLM council (grounded by a
deterministic ensemble) → EV decision engine → risk gates → order — and differs
only in its trading *profile* (`monk`, `anchor`, `hunter`, `blitz`).

## Entry points

| Command | Purpose |
|---|---|
| `python -m live run` | Run the full 4-agent portfolio end-to-end, resumable, until the tournament ends. **This is the deployment.** |
| `python -m live once --fixture <id> --window PRE_MATCH` | Run one fixture/window for all agents. |
| `python -m live report` | Retrospective: P&L per agent, Brier vs market, fire rates, skip reasons. |
| `python predict_game.py` | Inspect a single forecast (no trading). |
| `python -m harness ...` | Paper-trading rehearsal of the live policy. |

## Doc map

- [LIVE.md](./LIVE.md) — running the 4-agent portfolio on a VM (setup, systemd, restart, monitoring).
- [STRATEGY.md](./STRATEGY.md) — how we win the arena: roster mandates and sizing ladder.
- [RUN.md](./RUN.md) — single-agent / ad-hoc cycle reference.
- [HARNESS.md](./HARNESS.md) — paper-trading harness and profile table.
- [DATA_SOURCES.md](./DATA_SOURCES.md) — Sportmonks / Polymarket / Kalshi feed reference.
- [GUIDE.md](./GUIDE.md) / [CHALLENGE.md](./CHALLENGE.md) — the Stair AI Arena rules and scoring.

## Forecast brain

The council (`reasoning/council.py`) runs Scout → Analyst → Devil → Judge. All
four roles now receive a `deterministic_context` from the calibrated ensemble
(`models/deterministic_v2.py`: Elo + Poisson/Dixon-Coles + de-vigged market
prior) and must reconcile their view against it, alongside web search, Reddit,
Grok social pulse, bookmaker anchors, and the tradable markets. The deterministic
signal is logged as its own node in the reasoning ledger DAG.
