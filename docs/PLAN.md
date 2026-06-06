# World Cup Agent — Upgrade Game Plan

> Objective: Maximize both **P&L** (by finding market edges) and **Stair AI Score**
> (by producing rich multi-step reasoning traces the ledger rewards).
>
> This document is written for a coding agent to execute sequentially.
> Each section defines the goal, the files to touch, and the exact contract expected.

---

## Current State

```
agent.py          — 9-step pre/HT workflow, places Kelly-sized orders
reasoning/llm.py  — Claude primary + OpenAI fallback + Gemini ensemble
reasoning/prompts.py — 6 digest/predict/strategy prompts
data/sportmonks.py  — fixture detail, schedule
data/supabase_client.py — 5 priors tables + HT checkpoint
data/polymarket.py  — Gamma + CLOB moneyline mids
betting/kelly.py    — Kelly criterion bet sizing
ledger/client.py    — Reasoning Ledger v0.3 batch submission
config.py           — all env vars, model names, betting params
```

The agent already runs end-to-end. These upgrades are **additive layers** on top.

---

## Phase 1 — Intelligence Upgrades (data layer)

### 1A. `data/web_search.py` — Targeted news search (NEW FILE)

**Goal**: Fetch pre-match injury/lineup/weather signals from the public web.
This is NOT open-ended search. Every query is structured and narrow.

**Implementation**:

Use the [Serper API](https://serper.dev) (cheap, fast JSON Google results) or
`httpx` against DuckDuckGo Instant Answer if no key is available.
Add `SERPER_API_KEY` to `config.py` and `.env.example`.

```python
# data/web_search.py

SEARCH_QUERIES = [
    "{home} {away} injury news {date}",
    "{home} starting lineup {date}",
    "{away} starting lineup {date}",
    "{home} {away} match preview {date}",
]

def fetch_injury_news(home: str, away: str, match_date: str) -> list[dict]:
    """
    Run each query template, return list of {title, snippet, url} dicts.
    Deduplicate by URL. Cap at 5 results per query → max 20 results total.
    """

def fetch_lineup_news(home: str, away: str, match_date: str) -> list[dict]:
    """
    Focused on lineup confirmations only.
    Also checks Sportmonks fixture.participants[].meta.position is populated
    (confirmed lineups appear 60 min pre-kickoff) before falling back to search.
    """
```

**Integration point**: `agent.py` step 3.5 — run AFTER Sportmonks fetch, BEFORE predict.
Log result as a `ToolCalling` ledger record.

**Key rules**:
- If Sportmonks already has confirmed lineups (`participants[].meta.position` populated),
  skip the lineup search queries — the API data is more reliable than scraped headlines.
- Extract only: player names mentioned as injured/doubtful/suspended, and starting XI names.
- Pass extracted signals to `reasoning/prompts.py` as a new `news_signals` field in `predict_input`.

---

### 1B. `data/reddit_sentiment.py` — Reddit match thread sentiment (NEW FILE)

**Goal**: Scrape the top Reddit match-preview thread for both teams and
extract a **crowd sentiment score** (−1 to +1 per team) that feeds into
the council's devil's advocate step.

**Why Reddit over Twitter**:
- Reddit API (free tier, no auth needed for read) returns structured JSON
- r/soccer match threads have high-signal fan analysis, not just noise
- Comments are upvoted — top comments surface real tactical takes
- No API key needed: use `https://www.reddit.com/r/soccer/search.json?q={home}+{away}&sort=new`

**Implementation**:

```python
# data/reddit_sentiment.py

REDDIT_HEADERS = {"User-Agent": "worldcup-agent/1.0 (research bot)"}

def search_reddit_threads(home: str, away: str) -> list[dict]:
    """
    Search r/soccer for threads matching the fixture.
    Return top 3 threads by score with title, url, score, num_comments.
    """

def fetch_top_comments(thread_url: str, limit: int = 20) -> list[str]:
    """
    Fetch top 20 comments from a thread (sorted by 'top').
    Append .json to any reddit URL to get the API response.
    Strip markdown formatting. Return list of comment strings.
    """

def get_sentiment_bundle(home: str, away: str) -> dict:
    """
    Returns:
    {
      "threads_found": int,
      "top_comments": [str],          # raw top comments (for LLM)
      "home_mentions": int,           # count of home team name in comments
      "away_mentions": int,           # count of away team name in comments
    }
    Raw comments get passed to the council's scout step for LLM interpretation.
    Do NOT do keyword sentiment here — let the LLM read the comments directly.
    """
```

**Integration point**: `agent.py` step 3.5, parallel with web search.
Log as `ToolCalling` ledger record. Feed into `predict_input` as `reddit_sentiment`.

---

### 1C. `data/kalshi.py` — Kalshi odds fetch (NEW FILE)

**Goal**: Pull Kalshi's moneyline odds for the same fixture and compare
against Polymarket. A spread between the two markets is a deterministic
signal — it means one market is mispriced.

**Implementation**:

Kalshi's public markets API requires no auth for reads:
`https://api.elections.kalshi.com/trade-api/v2/markets?ticker=FIFA`

```python
# data/kalshi.py

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

def search_fixture_markets(home: str, away: str) -> list[dict]:
    """
    Search Kalshi markets for this fixture by team name.
    Returns list of {ticker, title, yes_ask, yes_bid, yes_mid, no_mid}.
    """

def get_moneyline(home: str, away: str) -> dict:
    """
    Returns: {"home": float, "draw": float, "away": float}
    Prices as probabilities (mid price of YES).
    Returns None values for outcomes not found.
    """
```

**Integration point**: `agent.py` step 4.5 — run AFTER Polymarket fetch.
Log as `ToolCalling` ledger record.

**Cross-market spread gate** (deterministic, in `agent.py`):
```python
# If Kalshi and Polymarket agree within 3pp on the winner, confidence +1
# If they disagree by >8pp, flag as "contested market" — reduce bet size by 50%
# If both price a team > 65%, Kelly cap raised to 25% (strong consensus)
pm_home = polymarket_mids["home"]
kalshi_home = kalshi_mids["home"]
spread = abs(pm_home - kalshi_home)
market_consensus = spread < 0.03  # both markets agree
market_contested = spread > 0.08  # markets disagree significantly
```

---

## Phase 2 — Reasoning Council

**Goal**: Replace the current single-LLM predict step with a 4-role council.
Each role is a separate LLM call, logged as a separate ledger record with DAG linkage.
This is the primary driver of the **Reasoning quality** score.

### Architecture

```
[Scout]          fast model, reads raw data, flags anomalies   → Planning record
[Analyst]        deep model, forms base prediction             → Thinking record
[Devil's Advocate] different model, argues opposite side       → Thinking record (parent: Analyst)
[Judge]          synthesizes all, outputs final probability     → Thinking record (parent: Scout + DA)
```

The `parent_ids` field in each ledger record links them into a DAG.
The Judge's Acting/prediction record has `parent_ids = [scout_id, analyst_id, da_id]`.
This is exactly the structure the scoring rubric rewards.

### 2A. New file: `reasoning/council.py`

```python
# reasoning/council.py

def run_council(
    sm_digest: str,
    pm_digest: str,
    sb_digest: str,
    news_signals: str,       # from web_search
    reddit_bundle: str,      # from reddit_sentiment
    home: str,
    away: str,
) -> CouncilResult:
    """
    4-step council. Returns CouncilResult with:
    - final prediction (outcome, probability, confidence)
    - all 4 LLMResult objects (for ledger trace)
    - record IDs for DAG linkage
    """
```

**Step 1 — Scout** (`claude-haiku-4-5-20250929` or `gpt-4o-mini`, thinking_budget=512):
- System: "You are a rapid intelligence scout. Scan all data for red flags: missing key players,
  extreme weather, unusual line movement, lopsided market vs statistical profile. Output a JSON
  list of flags with severity (low/medium/high) and a one-line rationale."
- Input: sm_digest + news_signals (injury/lineup headlines) + reddit top comments
- Output: `{"flags": [{"signal": str, "severity": str, "rationale": str}]}`
- Log as `Planning` record

**Step 2 — Analyst** (`claude-sonnet-4-5-20250929`, thinking_budget=4096):
- System: "You are a football probability analyst. Using historical priors, team form, and
  statistical profiles, produce a market-blind probability estimate. Do NOT look at Polymarket
  prices yet. Use your best statistical judgment."
- Input: sm_digest + sb_digest + scout flags
- Output: `{"outcome": str, "probability": float, "rationale": str, "confidence": "low|medium|high"}`
- Log as `Thinking` record (parent_ids: [scout_record_id])

**Step 3 — Devil's Advocate** (`deepseek-reasoner` OR `gemini-2.5-pro`, thinking_budget=2048):
- System: "You are a contrarian analyst. The lead model has predicted X with probability P.
  Your job is to make the STRONGEST possible case for why this prediction is WRONG.
  What does the lead model miss? What scenarios favor the other outcome? Output a JSON
  counter-argument with an adjusted probability."
- Input: analyst result + sb_digest + sm_digest
- Output: `{"counter_outcome": str, "counter_probability": float, "key_risk_factors": [str]}`
- Use DeepSeek-R1 here — it gives raw chain-of-thought, richest `internal_reasoning` content
- Log as `Thinking` record (parent_ids: [analyst_record_id])

**Step 4 — Judge** (`claude-sonnet-4-5-20250929`, thinking_budget=4096):
- System: "You are the final arbiter. You have a lead analyst prediction and a devil's advocate
  counter. Weigh both arguments. Then look at the Polymarket and Kalshi market prices as a
  calibration reference. Output a final calibrated probability. If the devil's advocate raised
  strong points, adjust accordingly."
- Input: analyst result + DA result + pm_digest + kalshi mids + market_consensus flag
- Output: `{"outcome": str, "probability": float, "confidence": str, "council_summary": str}`
- Log as `Thinking` record (parent_ids: [scout_id, analyst_id, da_id])

### 2B. `CouncilResult` dataclass

```python
@dataclass
class CouncilResult:
    outcome: str
    probability: float
    confidence: str
    council_summary: str
    scout: LLMResult
    analyst: LLMResult
    devil: LLMResult
    judge: LLMResult
    # record IDs assigned during ledger build, for DAG linkage
    scout_rid: str = ""
    analyst_rid: str = ""
    devil_rid: str = ""
    judge_rid: str = ""
```

---

## Phase 3 — Ledger Upgrades

### 3A. DAG linkage in `ledger/client.py`

Current `LedgerSession` doesn't use `parent_ids`. Add support:

```python
# In ledger/client.py:

def thinking(
    self,
    model: str,
    prompt: str,
    inputs: dict,
    output: dict,
    thinking_chain: str,
    upstream_record_id: str | None = None,
    parent_ids: list[str] | None = None,   # NEW
) -> str:
    """Returns the record_id so callers can wire DAG links."""
```

Every Thinking record produced by the council should have `parent_ids` set.
Return the `record_id` string from each ledger method so agent.py can wire them together.

### 3B. New ledger records to add

The current 14-record trace should expand to ~22 records with these additions:

| # | Behavior | Description |
|---|---|---|
| +1 | ToolCalling | Web search — injury/lineup fetch |
| +2 | ToolCalling | Reddit sentiment fetch |
| +3 | ToolCalling | Kalshi odds fetch |
| +4 | Planning | Scout output (council step 1) |
| +5 | Thinking | Analyst output (council step 2) |
| +6 | Thinking | Devil's Advocate output (council step 3) |
| +7 | Thinking | Judge synthesis (council step 4) |
| +8 | Reflecting | Post-trade reflection: "Given the market price was X and we predicted Y, here is what we would do differently" |

The `Reflecting` record at the end is easy to add and is explicitly listed in the
Reasoning Ledger schema as a behavior type. It signals epistemic humility and
is likely rewarded by the quality rubric.

---

## Phase 4 — Deterministic Gate Upgrades (in `agent.py`)

Replace the current simple `should_bet` call with a structured gate:

```python
# reasoning/gates.py  (NEW FILE)

@dataclass
class GateResult:
    should_trade: bool
    bet_multiplier: float   # 0.0 to 1.5 — scales Kelly fraction
    veto_reason: str | None
    boost_reason: str | None

def evaluate_gates(
    model_prob: float,
    pm_mid: float,
    kalshi_mid: float | None,
    scout_flags: list[dict],
    confidence: str,
    wallet_balance: float,
) -> GateResult:
    """
    Deterministic decision gates. Applied BEFORE Kelly sizing.

    Gate 1 — Minimum edge:
      edge = model_prob - pm_mid
      If abs(edge) < 0.05: veto (no trade)

    Gate 2 — Market consensus boost:
      If kalshi_mid is not None:
        spread = abs(pm_mid - kalshi_mid)
        If spread < 0.03 and both favor our outcome: multiplier *= 1.25
        If spread > 0.08: multiplier *= 0.50  (contested, size down)

    Gate 3 — Scout flag veto:
      If any flag has severity="high" and is about our predicted winner: veto

    Gate 4 — Confidence gate:
      If council confidence == "low": multiplier *= 0.5
      If council confidence == "high": multiplier *= 1.2

    Gate 5 — Minimum wallet:
      If wallet < 2.00: veto (preserve capital)
    """
```

---

## Phase 5 — Config & Dependencies

### 5A. `config.py` additions

```python
# Add these:
SERPER_API_KEY: str = os.environ.get("SERPER_API_KEY", "")
KALSHI_API_KEY: str = os.environ.get("KALSHI_API_KEY", "")  # optional, reads are public

# Council model assignments
SCOUT_MODEL: str = "claude-haiku-4-5-20250929"   # fast, cheap
ANALYST_MODEL: str = "claude-sonnet-4-5-20250929" # deep reasoning
DEVIL_MODEL: str = "deepseek-reasoner"            # raw chain-of-thought
JUDGE_MODEL: str = "claude-sonnet-4-5-20250929"  # final synthesis
```

### 5B. `.env.example` additions

```
SERPER_API_KEY=your_serper_key_here     # get from serper.dev (free tier: 2500 queries/month)
KALSHI_API_KEY=                         # optional — public reads work without it
DEEPSEEK_API_KEY=your_deepseek_key_here # already in .env
```

### 5C. `requirements.txt` additions

```
# No new packages needed for Reddit (uses httpx which is already installed)
# Serper uses httpx too
# DeepSeek uses openai SDK (already installed)
```

---

## Execution Order for a Coding Agent

Execute these tasks **in order**. Each task is self-contained and testable.

### Task 1: `data/web_search.py`
- [ ] Create `data/web_search.py` with `fetch_injury_news`, `fetch_lineup_news`
- [ ] Add `SERPER_API_KEY` to `config.py` and `.env.example`
- [ ] Test: `python -c "from data.web_search import fetch_injury_news; print(fetch_injury_news('Mexico', 'South Africa', '2026-06-11'))"`

### Task 2: `data/reddit_sentiment.py`
- [ ] Create `data/reddit_sentiment.py` with `get_sentiment_bundle`
- [ ] Test: `python -c "from data.reddit_sentiment import get_sentiment_bundle; print(get_sentiment_bundle('Mexico', 'South Africa'))"`

### Task 3: `data/kalshi.py`
- [ ] Create `data/kalshi.py` with `get_moneyline`
- [ ] Test: `python -c "from data.kalshi import get_moneyline; print(get_moneyline('Mexico', 'South Africa'))"`

### Task 4: `reasoning/council.py`
- [ ] Create `CouncilResult` dataclass
- [ ] Implement 4-step council using existing `call_claude` from `reasoning/llm.py`
- [ ] Add DeepSeek call for devil's advocate (using OpenAI SDK with DeepSeek base URL — key already in `.env`)
- [ ] Test: `python -c "from reasoning.council import run_council; ..."`

### Task 5: `reasoning/gates.py`
- [ ] Create `GateResult` dataclass + `evaluate_gates` function
- [ ] Unit test all 5 gates with edge cases

### Task 6: Update `ledger/client.py`
- [ ] Add `parent_ids: list[str] | None` param to `thinking()`, `planning()`, `reflecting()`
- [ ] Have all ledger methods return the `record_id` string (currently `thinking` doesn't return it)
- [ ] Add a `reflecting()` method if it doesn't exist yet

### Task 7: Update `agent.py`
- [ ] Import and call `web_search`, `reddit_sentiment`, `kalshi` in step 3.5
- [ ] Replace `llm.predict()` call with `council.run_council()`
- [ ] Replace `should_bet` call with `gates.evaluate_gates()`
- [ ] Wire DAG linkage: pass council record IDs as `parent_ids` in ledger calls
- [ ] Add the new 8 ledger records (see Phase 3B table)
- [ ] Add `Reflecting` record at end of session

### Task 8: End-to-end smoke test
- [ ] `python agent.py --test-connection`
- [ ] `python agent.py --fixture-id 19609127 --window prematch`
- [ ] Verify ledger shows ~22 records, all stored
- [ ] Verify DAG linkage: judge record's `parent_ids` contains analyst + DA + scout record IDs

---

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Ledger records per session | 14 | ~22 |
| Reasoning depth (DAG layers) | 1 | 4 |
| Data sources | 3 (SM, PM, SB) | 6 (+ web, reddit, kalshi) |
| Model providers used | 2-3 | 3-4 (adds DeepSeek) |
| Deterministic gates | 2 | 5 |
| Council roles | 1 | 4 |

The multi-role council with DAG ledger linkage is the single highest-impact change.
It simultaneously improves prediction calibration AND produces the exact trace structure
that the Stair AI reasoning quality rubric rewards.
