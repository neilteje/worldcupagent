# Audit — why the 2026-06-10 friendly harness run underperformed

Scope: full trace of `agent.py` PRE_MATCH cycle, harness path
(`harness/runner.py` → `predictor.py` → `paper_broker.py`), all prompts in
`reasoning/prompts.py`, and the decision/gate stack
(`betting/decision.py` + `reasoning/gates.py`).

## Root causes (ranked)

### RC1 — The council was flying blind in the harness
`harness/predictor.py::_predict_council` passed **`None` for both the Sportmonks
digest and the Supabase digest** — the two strongest grounding inputs. The
council saw only web search, Reddit, and Grok pulse for friendlies, so the
Analyst defaulted near base rates (hence Portugal ≈36% vs Nigeria, "low"
confidence everywhere). `agent.py` fetches both; the harness did not. The
Supabase priors were *always fetchable* (the country resolver works on team
names — Portugal, Nigeria, England are all in `ads_a_h2h_country`); we simply
never called it.

**Fix:** shared context builder `data/fixture_bundle.py` used by the harness
(and available to anything else), fetching Sportmonks (when a fixture id
exists) + Supabase priors (by team name) and running the same digest LLM calls
`agent.py` uses.

### RC2 — Betting on synthetic noise
`paper_broker.synthetic_market` derives demo prices from **our own forecast +
seeded jitter + vig**. Any "edge" against it is the jitter itself — mean-zero
noise minus the vig, i.e. negative EV by construction. The aggressive agent's
one bet (PAK-AFG HT draw @ 0.28) was exactly this. Meanwhile profiles had no
concept of market provenance.

**Fix:** profiles get `trade_synthetic` (default **False**) and
`synthetic_size_multiplier` (×0.25 when explicitly enabled). Trades record
`market_source`; arena path is unaffected (always real Polymarket).

### RC3 — Conservative profile was mathematically unfireable
Gate stack for `conservative`: confidence ≥0.60 AND edge ≥6pp vs fair AND
EV ≥5%/$ AND scout veto. The council's "low" label maps to 0.40 — so on any
low-confidence window (all friendlies, by RC1) the confidence floor alone
guaranteed HOLD before edges were even examined. Four independent AND-gates
multiply into ~zero fire probability.

**Fix:** profiles re-tuned as four distinct, defensible postures
(monk/anchor/hunter/blitz); the default disciplined profile (`anchor`) accepts
low-confidence forecasts and relies on the edge/EV bars instead of a hard
confidence wall. Confidence labels themselves get meaning (RC5).

### RC4 — Double edge gating (two unrelated bars)
`betting/decision.py` requires `edge_vs_fair ≥ MIN_EDGE_VS_FAIR` (3pp,
de-vigged). Then `reasoning/gates.py` Gate 2 *separately* requires
`model_prob − raw_mid ≥ MIN_EDGE` (5pp, vig-inflated). The second bar is
stricter and measured against the worse price, so favorites that legitimately
cleared the fair-price bar died in gates. This re-introduced the exact
"only-check-the-vig-inflated-price" bug the EV engine was built to fix.

**Fix:** single source of edge truth. The profile's `min_edge_vs_fair` drives
tradability in `decision.py`; `evaluate_gates` no longer applies its own edge
bar by default (parameterized — pass `min_edge=` explicitly if a caller wants
it). Gates remain a pure risk overlay: wallet floor, market existence,
cross-market consensus sizing, scout veto, confidence multiplier.

### RC5 — No grounding, no sanity checks, weak calibration language
- Analyst prompt mentioned bookmaker consensus in passing; nothing forced it to
  start from a base rate, list evidence per outcome, or reconcile with an
  explicit anchor.
- Devil was free-form contrarianism, not an attack on the weakest assumption.
- Judge had no rule for when to move vs hold near the bookmaker.
- Council output had no post-hoc checks: nothing renormalized probabilities,
  clamped fabricated extremes, or flagged >15pp divergence from the bookmaker
  without devil support.
- "low" confidence meant "we guessed" rather than "data was thin."

**Fix:** `reasoning/grounding.py` (anchor extraction → sanity checks →
documented low-confidence shrink toward the anchor, logged in the ledger), plus
prompt upgrades for Analyst/Devil/Judge, plus confidence capped at "low"
whenever both structured digests are missing.

## What I am deliberately NOT changing
- **Ledger schema / order placement contract** — untouched.
- **The council role structure** (Pulse→Scout→Analyst→Devil→Judge) and its
  per-role ledger DAG — that's the reasoning-score engine.
- **The deterministic engine** (`agent/`, `models/`, `backtesting/`) — parallel
  path; only borrowed ideas (base rates, anchors, sanity audits).
- **Graceful degradation** — every data fetch still fails soft, but failures
  now log loudly with the role/source name.
- **Kelly math** (`betting/kelly.py`) — correct as is.

## Fix list (what shipped in this pass)
1. `data/fixture_bundle.py` — shared Sportmonks+Supabase context for the
   harness (and any caller); harness `Fixture` gains optional
   `sportmonks_fixture_id`.
2. `reasoning/grounding.py` — anchor (bookmaker consensus → Sportmonks ML →
   none), sanity checks (renormalize, clamp 2–92%, divergence flag), shrink
   `p' = (1−λ)·p + λ·anchor` with λ = 0.5/0.25/0 for low/medium/high
   confidence (only when an anchor exists). Wired into `council.run_council`,
   surfaced as `CouncilResult.grounding`, logged to ledger.
3. Prompt upgrades — Analyst (base-rate start, evidence per outcome, explicit
   unknowns, anchor reconciliation, no fabrication), Devil (single weakest
   assumption), Judge (3-way distribution rules, move-vs-hold rules, extremes
   need cited evidence).
4. `harness/profiles.py` — four profiles (monk/anchor/hunter/blitz) +
   `trade_synthetic` policy + `get_profile()` resolving `AGENT_PROFILE` env;
   same object consumed by `agent.py` (`--profile`) and the harness.
5. `betting/decision.py` — `min_edge_vs_fair` parameterized (profile-driven).
6. `reasoning/gates.py` — edge bar now opt-in (`min_edge=None` default), scout
   veto parameterized; gates = risk overlay only.
7. `agent.py` — profile loading (env/CLI), profile-driven decision + gates +
   sizing caps, profile/market_source/edge-breakdown in ledger payloads.
8. Docs: this file, `docs/HARNESS.md`, `docs/RUN.md`.
