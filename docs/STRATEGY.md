# STRATEGY.md — How we actually win this arena

*Strategy memo, 2026-06-11. Grounded in `docs/GUIDE.md` (arena rules),
`docs/CHALLENGE.md` (prize structure), `agent.py` / `ledger/client.py` (what we
can actually submit), and `harness/` (what the friendlies taught us).*

---

## 0. Executive summary

1. **The biggest prize is not P&L.** Highest Stair AI Score (reasoning +
   prediction skill) pays $2,000; Best P&L pays $1,000; Best Writeup $1,000;
   Community's Choice $500. We should run a **prize-portfolio**, not a betting
   portfolio: one agent ruthlessly optimized for the Score track, one for
   risk-adjusted P&L, one for P&L *upside via skew*, one for event-driven
   opportunism — plus deliberate investment in the writeup (this repo's docs
   are literally a prize entry).
2. **Unspent bankroll is worthless to us.** Whatever the agents hold at the end
   "flows back into the prize pool" (`CHALLENGE.md`). The $100 is ammunition,
   not savings. Capital preservation only matters insofar as a dead wallet
   can't bet. This kills classic Kelly logic (§5).
3. **The $5/order cap + one order per window makes sizing nearly irrelevant
   and selection everything.** Max exposure is ~$10/fixture (PRE + HT). Any
   genuine edge already wants more than $5 under half-Kelly on $100, so the
   cap binds first. Our complexity budget should go to *which* outcomes and
   *when*, not to sizing math.
4. **The P&L leaderboard will be won in the right tail.** A disciplined
   grinder's expected profit is on the order of +$10–30 over the whole
   tournament (math in §5); a lucky variance agent will post +$150–300.
   Therefore our P&L-track agents must buy **positively-skewed +EV outcomes**
   (draws and underdogs at ≤0.40), never grind 0.65-priced favorites for
   $2.70 paydays.
5. **Our durable edges are (a) live-information fusion** (Grok pulse + web
   scout + Reddit at lineup-release time — most competitors will be static
   stats models or single-prompt LLMs), **(b) the HT window** (15 minutes,
   thin attention, overreaction to scorelines — most agents won't even show
   up), and **(c) the grounding layer** (calibrated, anchored, sanity-checked
   probabilities feeding the PSL score, which most LLM agents will fail at
   through overconfidence).

---

## 1. The scoring landscape (facts first)

What the arena actually supports (verified against `GUIDE.md` + our code):

| Mechanic | Reality | Strategic consequence |
|---|---|---|
| Orders | Buy-YES limit orders + explicit close at HT when available | PRE exposure is flattened at HT before fresh halftime risk is opened; remaining exposure settles at FT |
| Windows | One prediction + order workflow per window; PRE_MATCH (→kickoff), HT (ko+45 → ko+60) | HT is a true re-underwrite: close prior fixture exposure, then open only the new +EV side(s) |
| Predictions | `Acting` record, `action_type="prediction"`, **single (outcome, probability)** — not a full distribution (`ledger/client.py`) | The PSL formula's exact shape is unknown to us. This is the single highest-value open question (§6, probe P-1) |
| Scoring | PSL on latest prediction per window + reasoning quality from the full ledger DAG | Predictions are scored **even when we don't bet** → a pure-forecast agent is possible at zero capital risk |
| Identity | 1 API key = 1 agent, multi-agent explicitly sanctioned ("mint a new key for each agent") | Running a coordinated 4-agent portfolio is legal by design |
| Wallet | $100 production; residual returns to prize pool | Rank is the only objective; terminal bankroll has no salvage value to us |
| Tournament | WC2026: 104 matches (72 group + 32 KO), 2 windows each → **208 prediction windows, 208 order slots** | Sample is small for edge to express; tiny for variance to wash out. Leaderboards will be noisy → don't chase them mid-stream |

Scale of the game: 104 matches × $10 max = **$1,040 theoretical max turnover**
per agent. A realistic disciplined agent fires on 30–40% of windows at ~$4
average → $250–350 turnover. At a (good) +4% realized edge that's **+$10–14
expected P&L**. Per-bet σ on a $5 skew bet at price 0.25 is ≈ $8.7; over 40
such bets, portfolio σ ≈ $55. Conclusion: *expected* P&L differences between
competent agents (±$15) are smaller than one standard deviation of luck
(±$55). **"Miles ahead" means +$75+; that comes from skew + luck, not from
grinding.**

---

## 2. Competitor modeling — four archetypes and how to beat each

**A. Notebook clones (most common).** The sample agent ("Deep Field") with a
single Claude call, light edits. Behavior: overconfident point probabilities
(LLMs love 0.70), sporadic $4 bets on the favorite, thin 8-record ledger.
*Beat them by:* calibration (our grounding layer literally clamps and shrinks
what they do raw) and ledger depth (our 14–15 record DAG with per-role
reasoning vs their linear trace). We don't need to out-bet them; they donate
PSL points.

**B. Market mirrors.** Smarter builders who realize the de-vigged Polymarket
mid is a strong forecast and submit it as their prediction. Good PSL floor,
near-zero P&L (they have no edge over the price by construction).
*Beat them by:* being the market **plus** real information at the margin —
lineup news 60–90 min before lock, HT xG. We only need to beat the market on
the 10–20% of fixtures where something is genuinely knowable that the price
hasn't absorbed. Everywhere else, our shrink-to-anchor makes us ≈ them.

**C. Quant stacks.** Elo/xG/Poisson models, decent calibration, systematic
small bets. The real competition on the Score track. Their weakness: they're
**static** — no live news, no social pulse, no HT update beyond scoreline.
*Beat them by:* fusing what they can't (Grok/X pulse, rotation news,
dead-rubber motivation in group game 3) and by out-reasoning them on the
ledger (a Poisson model's trace is one record; ours is a council).

**D. Degens / variance maximizers.** $5 on a longshot every game. One of them
will top the P&L board for a while; most will finish bottom quartile. *Do not
copy. Do not chase.* Their expected finish is poor; their lucky tail is
unbeatable by design. Our answer is a **+EV-skew agent** (§4, SAW) — the same
variance exposure but with positive drift, so over 60+ bets we expect to
finish above almost all of them while retaining a comparable right tail.

---

## 3. Original strategy catalog (ranked)

Each: mechanism → edge thesis → failure mode → codebase fit.

### S1. Score-track specialization (run a pure forecaster) — **rank 1**
**Mechanism:** one agent treats betting as an afterthought (fires only on
enormous edges to stay "active") and pours everything into PSL + ledger
quality: predicts **every** window (208 predictions), submits late in the
window (the arena scores the *latest* prediction — wait for lineups, predict
at T-10), shrinks hard to the bookmaker anchor when uncertain (λ up from 0.5
to ~0.7 for `low`), and keeps the full council DAG.
**Edge thesis:** the $2,000 track rewards exactly what archetypes A and D are
worst at. Volume matters too: an agent predicting all 208 windows with Brier
≈ market beats a sporadic genius.
**Failure mode:** if PSL heavily rewards bold correct calls over calibrated
humility (formula unknown), a pure-shrink agent caps its own ceiling. Probe
the formula first (§6).
**Fit:** trivial — profile knob (`min_edge_vs_fair` ≥ 0.10) + a λ override in
`reasoning/grounding.py`. Prediction-timing needs a scheduler tweak in how we
trigger `agent.py` (run at T-15, not T-60).

### S2. Skew harvesting (draws + underdogs only) — **rank 2**
**Mechanism:** a P&L agent that only buys outcomes priced ≤ 0.40 (draw or
dog) when our blended probability clears fair by ≥ 3pp. Never buys favorites.
**Edge thesis:** two stacked effects. (1) *Payout asymmetry*: at the $5 cap, a
0.25-priced winner returns +$15, a 0.65-priced winner +$2.69 — in a rank
race, only skew moves the needle. (2) *Crowd structure*: retail flow on
Polymarket backs teams it loves; nobody roots for the draw. Group-stage
draws run ~25–30% historically, clustering in cagey openers and dead-rubber
third games, while 3-way markets routinely price draws at 0.22–0.28 with vig
on top. Even a 2–3pp systematic draw underpricing is a real edge with 3–4×
payouts attached.
**Failure mode:** if Polymarket WC markets are professional (market makers
quoting off Pinnacle), the draw discount won't exist — measure before
committing (§6, M-3). Also: draws lose often; this agent will have ugly
losing streaks and needs ~40+ bets for drift to show.
**Fit:** one new profile field (max entry price / outcome-class filter) in
`harness/profiles.py` + a filter in `decide_trades` and the `agent.py`
decision step. Small diff.

### S3. Lineup-window sniping — **rank 3**
**Mechanism:** lineups drop ~T-75 to T-60. Our web scout + Grok pulse already
hunt exactly this (`web_search.gather_research` has a lineups mode;
`SOCIAL_PULSE_SYS` asks for lineup leaks). Run the council *after* lineups
land; bet only when the news is material (star benched, keeper injured) AND
the Polymarket mid hasn't moved versus its pre-lineup level.
**Edge thesis:** prediction markets are slow on team news for non-marquee
fixtures (nobody is market-making MEX–RSA lineups at 2am). This is the
clearest *informational* edge our stack has that static quant agents
structurally cannot have.
**Failure mode:** LLM scout hallucinating materiality (we saw it conflate
AFCON news into a friendly — the scout actually caught and labeled this,
which is encouraging); market already moved and we're the exit liquidity.
Requires logging pre/post-lineup mids to verify the lag exists (§6, M-4).
**Fit:** timing change (trigger `agent.py` at T-45 instead of T-∞) + a scout
flag class for "confirmed lineup surprise". No new architecture.

### S4. HT overreaction reversion — **rank 4**
**Mechanism:** at HT, buy the pre-match favorite when it is level or trailing
by one but dominates xG (our `extract_ht_stats` + `d_checkpoint_snapshot`
already deliver HT xG). Markets over-extrapolate scorelines; xG mean-reverts.
**Edge thesis:** a strong favorite at 0-0 HT often trades 15–20pp below its
conditional win probability; second-half goals from dominant teams are the
most predictable event in soccer. Bonus: almost no hackathon agent will trade
HT at all (15-minute window, requires live data plumbing — we built it).
**Failure mode:** thin HT liquidity → unfilled limits (30s TIF); a red card
or injury behind the scoreline that xG doesn't see (mitigate: scout veto
stays on). Must pre-warm the council at kickoff to fit the 15-min window —
our run takes 2–3 min.
**Fit:** the HT path in `agent.py` exists but still uses the legacy
predict/strategy flow — upgrading it to the council + EV engine is the one
real engineering item this memo implies.

### S5. Cross-market divergence trigger — **rank 5**
**Mechanism:** only bet when Polymarket and Kalshi disagree by >8pp
(`cross_market_signal` already classifies this as `contested`) and our
council sides with one of them; buy the cheaper consistent side. Invert the
current logic: today `contested` *halves* size (`CONTESTED_MULTIPLIER=0.5`) —
for a P&L agent, contested markets are where mispricing provably exists.
**Edge thesis:** when two venues disagree, at least one is wrong; an
independent third estimate (the council) breaks the tie with positive
selection.
**Failure mode:** Kalshi coverage is spotty (`markets_found: 0` in every test
so far) — this strategy may simply never trigger. Keep it as an opportunistic
overlay, not a mandate.
**Fit:** already 90% built (`data/kalshi.py`, gates) — needs one profile flag
to flip the contested multiplier.

### S6. Anti-correlated portfolio construction — **rank 6 (portfolio-level)**
**Mechanism:** all four agents share one forecast, but we *forbid* them from
holding the same outcome on the same fixture beyond two agents. The
shared-forecast harness already gives us this control point
(`harness/runner.py` runs all profiles against one prediction).
**Edge thesis:** prizes are per-agent order statistics. Four copies of one
bet = one lottery ticket at 4× stake; four decorrelated +EV tickets ≈
`1-(1-p)⁴` chance that *someone* lands in the money. We maximize
P(any agent wins any track), not E[sum of P&L].
**Failure mode:** forced decorrelation pushes some agent onto its 2nd-best
bet — a real EV cost. Apply only when bets would otherwise be identical;
roles (§4) make collisions naturally rare.
**Fit:** the arena agents run as independent processes; decorrelation is a
deployment-level convention (different `AGENT_PROFILE` mandates), not code.

### S7. Endgame variance escalation — **rank 7 (conditional)**
**Mechanism:** leaderboard-aware aggression for the knockouts. If a P&L agent
is outside the money entering the round of 16 (~32 windows left), escalate to
max exposure ($5 PRE + $5 HT) on the *most skewed* +EV outcome of every
remaining fixture, accepting ruin risk.
**Edge thesis (the moonshot derivation, prompt 6):** concentrated risk is
rational iff (1) the payoff is a step function of rank — it is; (2) the gap
to the money exceeds remaining steady-state EV — if you're -$40 behind with
32 windows × ~$0.50 expected/window ≈ +$16 of grind left, grinding
*cannot* close the gap, so variance strictly dominates; (3) the variance you
buy is +EV or at worst EV-neutral — skew bets satisfy this; and (4) ruin has
no salvage cost — true here, leftover money isn't ours. All four conditions
are checkable; until (2) triggers, do NOT escalate.
**Failure mode:** escalating off a noisy mid-tournament leaderboard read
(survivorship). Trigger only at fixed checkpoints (end of group stage, end of
R16) with a written rule, not vibes.
**Fit:** no leaderboard API — this is a *manual* profile swap
(`AGENT_PROFILE=blitz` + raised caps) at the checkpoint. Document the rule
now so future-us doesn't improvise.

### S8. Dead-rubber motivation play — **rank 8**
**Mechanism:** group game 3 with qualification already decided → rotation,
draws, and weirdness spike. Scout explicitly hunts motivation context (it's
already in `SCOUT_SYS`); bet draws/dogs harder in flagged dead rubbers.
**Edge thesis:** quant models price teams, not incentives; this is a known
soccer-betting inefficiency that an LLM reading the news is unusually good at
detecting.
**Failure mode:** markets *do* partially price famous dead rubbers; the edge
is in degree. Fold into S2 as a boost rather than a standalone agent.
**Fit:** scout flag class + a profile multiplier. Tiny.

### S9. The writeup as a strategy (meta) — **rank 3 overall, $/effort #1**
$1,000 for Best Writeup + $500 Community's Choice are decided by humans, not
variance. We already have: a real audit (`docs/AUDIT.md`), a paper-trading
harness with plots, a grounding layer with a falsifiable formula, and a
4-agent natural experiment. **The experiment design itself is the writeup**:
"we ran four risk postures on one brain — here's what the World Cup did to
them," with per-agent P&L curves from `harness/performance.py` adapted to
arena results. Budget real hours for this; it's the highest
probability-per-effort prize in the stack.

---

## 4. Recommended roster — keep the four slots, change the mandates

Four agents is the right *count* (4 keys, and prizes are per-agent order
statistics — more tickets in more tracks). But aggressiveness-only
differentiation is the wrong *shape*: it puts all four on the same strategy
with different volume knobs, which correlates them exactly where
decorrelation pays. Re-mandate by **strategy class**, keeping the existing
profile names so nothing breaks:

| Slot | Mandate | Prize track | Policy sketch (delta from today's profile) |
|---|---|---|---|
| `monk` → **ORACLE** | Pure forecast quality. Predicts all 208 windows, late in window; trades only ≥10pp edges (a handful all tournament) | Stair AI Score ($2k) | shrink λ: 0.7/0.35/0.1; prediction at T-15; everything else as tuned |
| `anchor` → **KEEL** | Disciplined all-outcome EV accumulator. The control arm and risk-adjusted P&L play | P&L (steady) + Score (secondary) | exactly today's anchor; it's well-tuned for this |
| `hunter` → **SAW** | Skew harvester: draws + dogs ≤0.40 entry only, max size when it fires, dead-rubber boost | P&L (upside) | add entry-price ceiling; size $5 flat when firing (variance is the product) |
| `blitz` → **SURGE** | Event-driven: lineup snipes + HT reversion + cross-market divergence; endgame escalation vehicle (S7 rule) | P&L (tail) | both windows, scout veto ON for lineup plays / OFF at HT, contested-multiplier inverted |

Notes:
- ORACLE and KEEL will mostly agree (fine — different tracks). SAW and SURGE
  are structurally decorrelated from them by mandate, satisfying S6 without
  any coordination code.
- Every agent keeps the full council + grounding + ledger DAG — reasoning
  quality is scored per agent, and it costs nothing to share the brain.
- The friendlies' lesson is encoded: nobody trades synthetic prices, nobody
  is gated to zero (ORACLE's near-zero betting is deliberate, and its track
  doesn't need bets).

## 5. Sizing philosophy — Kelly is the wrong objective here

Three independent reasons:

1. **The cap binds before Kelly does.** Half-Kelly on $100 wants >$5 for any
   edge ≥ ~3pp at typical prices (e.g. p=0.40 vs price 0.35 → full Kelly
   ≈ 7.7% ≈ $7.7). So Kelly's output is almost always "the cap." Keeping the
   machinery (it handles the rare thin-edge case and we built it) is fine;
   *believing* in it as the strategy is a category error.
2. **Kelly optimizes log terminal wealth we don't keep.** Residual bankroll
   returns to the prize pool. Our true utility is a step function of rank
   across four prize tracks.
3. **Rank races reward variance when +EV.** Kelly shrinks volatile bets;
   the P&L leaderboard *pays* for exactly that volatility (when drift ≥ 0).

What I'd use instead — a **three-rung ladder, selection-gated**:
- **$0** — default. The bar to bet at all stays high (edge vs fair, profile
  policy).
- **$3** — "balanced" rung: KEEL's favorites and low-skew positions, where
  upside is capped anyway.
- **$5 (max)** — *every* skew bet that clears the bar (SAW, SURGE). If the
  thesis is right, the cap is the constraint, not the risk.
- Ruin math sanity: floor is `MIN_WALLET_USD=$2`; even SURGE at $10/fixture
  needs ~10 consecutive full-size losses to halve. P(effective ruin) per
  agent is <10% under any sane fire rate; portfolio-level ruin (all four) is
  negligible. We can afford the ladder.

## 6. Days 1–2 observation protocol (measure before changing anything)

**Probes (day 1, $1 orders, answers we lack):**
- **P-1. PSL formula.** Ask the Stair team in Discord directly (cheapest
  alpha in this memo): is PSL Brier/log-loss on the stated (outcome, p)? Does
  predicting a 0.05 longshot honestly score better than a 0.55 favorite
  honestly? ORACLE's entire policy bends around this answer.
- **P-2. Order semantics.** Does a second order in the same window get
  rejected (GUIDE says one/window) — and can the HT order be on a *different
  outcome* than PRE (the only hedge that exists)? Two $1 probes on a dev
  fixture.
- **P-3. Fill behavior.** We have write-only visibility. Submit at mid+0.02
  and infer fills from `agents/me` wallet deltas. If fill rate <50%, raise
  limit offsets before concluding we "have no edge."

**Measurements (days 1–2, log per fixture window, no behavior change):**
- **M-1. Council vs market calibration:** Brier of our prediction vs Brier of
  the de-vigged Polymarket mid, per window (`harness/performance.py` already
  computes ours; add the market baseline column).
- **M-2. Anchor divergence outcomes:** when we diverge >5pp from the
  bookmaker, who was right? This is the direct test of whether the council
  adds anything over the anchor.
- **M-3. Draw pricing audit:** PM draw mid vs Sportmonks bookmaker draw
  consensus vs realized draws (needs ~15 games to even whisper). SAW's
  mandate lives or dies here.
- **M-4. Lineup lag:** PM mid at T-90 vs T-45 vs lock on every fixture; did
  material lineup news move the price, and how fast? SURGE's pre-match
  mandate lives here.

**Pivot triggers (written now, so we don't improvise later):**
- Council Brier > market Brier by >0.02 after ~16 windows → raise ORACLE's λ
  toward 1.0 (become the market + info tilts only) and tighten everyone's
  divergence tolerance.
- SAW hit rate after 15 skew bets <60% of what our own probabilities implied
  → the market prices draws fine; fold SAW into KEEL.
- Fill rate <50% at mid+0.02 → offsets to +0.03/+0.04 before any strategy
  conclusions.
- Zero bets by KEEL across an entire match day with real markets → its bar is
  miscalibrated for live vig levels; lower `min_edge_vs_fair` by 1pp (this is
  the friendlies' "0 bets" lesson with a numeric trigger attached).
- **Stay-the-course rule:** no strategy-class change before 16 windows of
  data. Everything before that is noise on noise.

## 7. Ideas considered and rejected (including the team's)

| Idea | Verdict | Why |
|---|---|---|
| "YOLO favorites" (team) | **Kill.** | Worst of all worlds: capped upside (+$2.70 on $5 at 0.65), real downside, zero skew. The P&L race is won in the tail; favorites have no tail |
| "Max-size every game" (team) | **Kill as stated, keep the kernel.** | Indiscriminate max-sizing is -EV after vig. The kernel — *when you bet skew, bet the cap* — survives as the §5 ladder |
| "Wait-and-see" (team) | **Kill as a strategy, keep as protocol.** | Passivity forfeits the Score track (predictions cost nothing — predict from day 1, always). The disciplined version is §6: bet normally, measure hard, pivot on written triggers only |
| "HT hedge/double-down" (team) | **Refine.** | "Double down if winning" buys a worse price on a correlated outcome — usually -EV. The defensible versions are S4 (xG reversion against the scoreline) and P-2's cross-outcome lock-in *if* the order model allows it. Test, then decide |
| Martingale / loss-chasing | Kill | $5 cap makes recovery sizing impossible by construction; no exits means no damage control |
| Fade-the-market as identity | Kill | The de-vigged market is the best single forecaster in the room. Fade only with named, evidenced information (S3) — that's a trigger, not an identity |
| Copy the mid-tournament leaderboard leader | Kill | Survivorship on ~30 bets of noise; their variance is not your edge |
| Bloat the ledger for reasoning score | Kill | The rubric rewards structure (DAG, evidence, counterfactuals), not volume. We already submit 14–15 linked records; pad nothing |
| Hedging/exit-based strategies | Re-test | Close now exists for order/position exits; HT close-then-retrade is the only sanctioned exit workflow |
| Trading synthetic/no-liquidity markets | Killed already | Friendlies lesson; `trade_synthetic=false` is default policy |
| All four agents on one bet ("the syndicate") | Kill (mostly) | Correlated copies waste the order statistic (§S6). Exception: a true 5-σ opportunity may justify 2–3 agents converging — cap it there |

## 8. Two experiments worth a backtest (design only)

**E-1. Draw-skew viability (for SAW), WC2022 replay.** Replay 2022 fixtures and
Sportmonks 2022 odds via the proxy through the live policy. Policy
under test: long draw when blended p(draw) − fair(draw) ≥ 3pp, $5 flat.
Output: ROI, max drawdown, terminal P&L distribution vs the KEEL policy on
the same fixtures. ~64 matches, decided in an afternoon once keys work.
Success bar: positive ROI and terminal σ ≥ 2× KEEL's (we're buying variance —
verify we actually get it).

**E-2. HT reversion frequency (for SURGE), data-only study.** From StatsBomb
history (the `ads_a_*` aggregates + worldcup_2022 backtester): conditional on
pre-match favorite level-or-trailing at HT with xG share >55%, what fraction
win / draw? Compare against a naive HT price heuristic (favorite at 0-0
trades ≈ its pre-match price minus 15–20pp). No order placement, no live
data needed; quantifies S4's edge before we risk a dollar on it.

---

*Bottom line: we are not running one betting strategy four times at different
volumes. We are running a prize portfolio — a forecaster (ORACLE), a control
(KEEL), a lottery ticket with positive drift (SAW), and an event sniper with
an endgame clause (SURGE) — on one shared, grounded brain, with a written
observation protocol so the first 48 hours produce decisions instead of
vibes. And we write it all up, because the writeup is a prize too.*
