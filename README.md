# World Cup Arena Agent

Our AI agent competing in the 2026 Stair AI World Cup Agent Arena.

We deploy a **4-agent portfolio** (`monk`, `anchor`, `hunter`, `blitz`) that shares
**one brain** and differs only in trading aggressiveness. The brain fuses a
deterministic quantitative ensemble with web/LLM research and lets an LLM council
reconcile everything into the final, calibrated forecast and trade.

```
data → deterministic_v2 ensemble ─┐
       web + Reddit + Grok pulse ──┼─► LLM council (Scout→Analyst→Devil→Judge)
       bookmaker/market anchors ───┘        → calibrated 1X2 distribution
                                            → EV-ranked decision + risk gates
                                            → order (≤ $5, long only)
                                            → reasoning-ledger DAG
```

The deterministic ensemble (`models/deterministic_v2.py`: Elo + Poisson/Dixon-Coles
+ de-vigged market prior, calibrated) is injected into **every** council role as
`deterministic_context` — a quantitative cross-check the LLMs must reconcile with,
not blindly copy. It is logged as its own node in the ledger trace.

## Layout

```
ENTRY POINTS
  live/                     ← the deployment (4 agents, one resumable process)
    runner.py               ← forever-loop: schedule → windows → settle
    cycle.py                ← run the shared brain once, then 4 per-agent tails
    arena_client.py         ← per-key arena API (2026-06-10 contract)
    roster.py, state.py     ← 4 agents from env keys; kill-anytime state store
    report.py, metrics.py   ← events.jsonl + retrospective evaluation
  agent.py                  ← single-agent council cycle (pre-match + half-time)
  predict_game.py           ← sandbox predictor for any fixture (no arena writes)
  harness/                  ← paper-trading rehearsal of the live policy

BRAIN
  reasoning/
    council.py              ← Grok pulse → Scout → Analyst → Devil → Judge
    grounding.py            ← bookmaker/ML anchors + sanity checks
    gates.py                ← deterministic trade gates (risk overlay)
    llm.py, prompts.py      ← multi-model calls + structured prompts
  models/
    deterministic_v2.py     ← calibrated Elo + Poisson + market ensemble
    team_strength.py, poisson_model.py, calibration.py
  betting/
    decision.py             ← EV-ranked, all-outcomes, de-vigged decision engine
    policy.py               ← single profile→orders policy (harness + live)
    kelly.py                ← Kelly criterion sizing
  harness/profiles.py       ← the 4 agent profiles (shared by harness + live)

DATA
  data/sportmonks.py        ← fixtures, ML predictions, odds, HT stats
  data/polymarket.py        ← tradable market mids
  data/kalshi.py, web_search.py, reddit_sentiment.py, supabase_client.py

SHARED
  config.py, ledger/client.py, agent/config.py (Settings)
  storage/                  ← run artifacts incl. live state/metrics (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # AGENT_KEY_MONK/ANCHOR/HUNTER/BLITZ, ANTHROPIC_API_KEY, etc.
```

## Running

```bash
# Live tournament — all 4 agents in one resumable process (see docs/LIVE.md)
python -m live test             # check all 4 agent keys + endpoints
python -m live run              # resumable loop until the final
python -m live report           # retrospective evaluation

# Single forecast / ad-hoc cycle
python predict_game.py --home Brazil --away Morocco --home-code BRA --away-code MAR
python agent.py --fixture-id 19609127 --window prematch

# Paper-trading rehearsal of the live policy
python -m harness now --fixture FRD-POR-NGA --window PRE_MATCH

pytest tests/ -q                # unit tests
```

## Scoring strategy

- **PSL (Probabilistic Skill Loss)** — proper scoring on the calibrated distribution,
  emitted every game whether or not a trade fires.
- **Reasoning quality** — the full ledger DAG (deterministic node + Scout→Analyst→
  Devil→Judge) is scored.
- **P&L** — EV-ranked, all-outcomes engine with de-vigged edges and Kelly sizing;
  trades only the highest-EV side that clears each agent's bar.

## Tuning (`config.py`)

| Variable | Default | Effect |
|---|---|---|
| `MIN_EDGE` | 0.05 | Min raw edge to place a bet |
| `MIN_EDGE_VS_FAIR` | 0.03 | Min edge vs the de-vigged fair price |
| `MAX_KELLY_FRACTION` | 0.20 | Max % of wallet per bet |
| `MAX_BET_USD` | 5.00 | Hard USD cap per order (arena rule) |

Per-agent aggressiveness (edge bars, Kelly fraction, stake caps, confidence floors,
scout-veto) lives in `harness/profiles.py` and is shared by the harness and live runner.
