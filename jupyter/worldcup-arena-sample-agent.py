#!/usr/bin/env python
# coding: utf-8

# # World Cup Agent Arena — Build-Day Walkthrough
# 
# Welcome! By the end of this notebook you'll have run a complete **trading agent** from start to finish. Use it as the template for the agent you'll build to compete in the arena.
# 
# ## What is this, in plain terms?
# 
# - **The arena** is a competition. Your agent looks at upcoming World Cup 2026 matches, predicts who will win, and (optionally) places play-money bets on those outcomes. You're scored on how good your predictions and trades are.
# - **An agent** is just a program that runs a loop: **gather data → think → act → record why it acted.** This notebook walks through one full pass of that loop for a single match.
# 
# ## The flow (run the cells top to bottom)
# 
# | Step | What it does |
# |------|--------------|
# | Setup | Set your keys and shared settings |
# | 1 | **Find the matches** and pick one to analyze |
# | 2 | **Fetch the match's pre-game data** (model predictions, odds, expected goals) and summarize it |
# | 3 | **Fetch the betting market** and its live prices, and summarize it |
# | 4 | **Fetch historical team stats** and summarize them |
# | 5 | **Predict the result** — the agent forms its own opinion, ignoring the market |
# | 6 | **Decide whether to bet** — compare the agent's opinion to the market |
# | 7 | **Place the bet** (open a position) — or skip if there's no edge |
# | 8 | **Record the agent's reasoning** to the ledger so the arena can audit and score it |
# 
# ## Glossary (skim this first)
# 
# | Term | What it means |
# |------|---------------|
# | **Fixture** | A single scheduled match (e.g. Mexico vs South Africa). "Fixture" is just the sports-data word for "game." |
# | **Sportmonks** | A sports-data provider. We use it for schedules, team info, model predictions, and odds. |
# | **Bookmaker odds** | The prices a betting company offers on each outcome. They imply a probability (e.g. odds that imply "home wins 55% of the time"). |
# | **Expected goals (xG)** | A stat estimating how many goals a team *should* have scored based on the quality of their chances. |
# | **Polymarket** | A prediction market where people trade on real-world outcomes. Prices move like a stock and reflect the crowd's implied probability. |
# | **Moneyline** | The "who wins?" market: three outcomes — home win, draw, away win. |
# | **Event slug** | Polymarket's human-readable id for a market, e.g. `fifwc-mex-rsa-2026-06-11`. We use it to look the market up. |
# | **Mid price** | The midpoint between the best buy and sell price for an outcome, from 0 to 1. A mid of 0.62 ≈ a 62% implied chance. |
# | **Edge** | (your probability − the market's probability). Positive edge = the market is underpricing your pick → maybe worth a bet. |
# | **Supabase** | The database holding the arena's extra historical stats. |
# | **Ledger** | A structured log of every step the agent took and why. The arena reads it to verify and score your agent. |
# | **LLM digest** | We ask Claude to boil a big, noisy API response down to a small, clean JSON summary so later steps stay simple. |
# 
# **Before running:** in the **Setup** cell, replace the two placeholder credentials — `ARENA_KEY` (mint at https://stair-ai.com/api-keys) and `ANTHROPIC_KEY` (get one at https://console.anthropic.com). The Supabase URL and key are shared across all builders and already filled in.

# ## Setup — keys, endpoints, and shared settings
# 
# This cell defines everything the rest of the notebook reuses: your two API keys, the arena's proxy URLs, and a few constants. **You only need to edit the two placeholder keys** — everything else already works on staging.
# 
# | Setting | What it is |
# |---------|------------|
# | `ARENA_KEY` | **← you set this.** Your arena API key; authenticates every arena call. |
# | `ANTHROPIC_KEY` | **← you set this.** Your Anthropic key, used for the Claude "digest" / reasoning calls. |
# | `ARENA` | Base URL of the arena (staging). |
# | `SPORTMONKS_PROXY` / `POLYMARKET_*` | Arena **proxy** URLs. You call the arena; it forwards to Sportmonks / Polymarket with its own upstream keys, so you never need theirs. |
# | `SUPABASE` / `SUPABASE_KEY` | Shared, read-only database access. Already filled in. |
# | `SPORTMONKS_SEASON_ID` | The World Cup 2026 season id — the only tournament this guide uses. |
# | `{ANTHROPIC,OPENAI,DEEPSEEK,GEMINI}_MODEL` | The default model for each provider — each one is the cheapest model in its family that actually exposes an internal-reasoning trace. |
# | `LLM_MAX_TOKENS`, `LLM_THINKING_BUDGET` | Anthropic extended-thinking knobs (consumed by `AnthropicLLM`). `budget_tokens` must be < `max_tokens`. |
# 
# The four provider classes (`AnthropicLLM` / `OpenAILLM` / `DeepSeekLLM` / `GeminiLLM`) each expose the same `complete(system_prompt, user_input) -> LLMResult` method, so every LLM call below looks identical regardless of which provider is active.

# In[42]:


import os, json, time, uuid, requests

ARENA            = "https://stair-ai.com"
SPORTMONKS_PROXY = f"{ARENA}/api/v1/data/proxy/sportmonks/v3/football"
POLYMARKET_CLOB  = f"{ARENA}/api/v1/data/proxy/polymarket-clob"
POLYMARKET_GAMMA = f"{ARENA}/api/v1/data/proxy/polymarket-gamma"
ARENA_KEY        = "FILL IN YOUR ARENA KEY HERE"   # get this from https://stair-ai.com/api-keys
# Staging shares a single publishable Supabase key for every builder — no
# per-account JWT, no extra setup. The arena will publish these two values
    # alongside the API key minted in the portal.
SUPABASE         = "https://ezvbmtvrvzageqixvdak.supabase.co"
SUPABASE_KEY     = "sb_publishable__m8bOkD05ToFwATpaWST5w_2-3fGS7V"
ANTHROPIC_KEY    = "FILL IN YOUR ANTHROPIC KEY HERE"  # get this from https://console.anthropic.com/api-keys

# --- Other LLM providers (OPTIONAL) ------------------------------------------
# This notebook calls Anthropic by default. To use a DIFFERENT provider instead:
#   1. paste its key below,
#   2. pip install its SDK (see the optional section in requirements.txt),
#   3. in the "Pick a provider" cell at the start of Step 2, comment out the
#      AnthropicLLM line and UNCOMMENT the line for your provider.
# The provider-specific reasoning knob + response parsing live inside the four
# classes (AnthropicLLM / OpenAILLM / DeepSeekLLM / GeminiLLM) defined below —
# every step just calls llm_client.complete(system_prompt, user_input), so
# nothing else has to change.
GEMINI_API_KEY   = "FILL IN YOUR GOOGLE GEMINI KEY HERE"    # Google AI Studio: https://aistudio.google.com/apikey
OPENAI_API_KEY   = "FILL IN YOUR OPENAI KEY HERE"           # OpenAI:           https://platform.openai.com/api-keys
DEEPSEEK_API_KEY = "FILL IN YOUR DEEPSEEK KEY HERE"         # DeepSeek:         https://platform.deepseek.com/api_keys
H_ARENA          = {"x-api-key": ARENA_KEY}
H_WCA            = {"apikey": SUPABASE_KEY, "Accept-Profile": "world_cup_arena"}

# Tournament constant — WC2026 is the only season this guide targets.
SPORTMONKS_SEASON_ID = 26618

# Reasoning-Ledger schema constants (per schema/records.schema.json v0.3 in
# StairAI/Reasoning-Ledger). agent_id is NOT set client-side: the arena
# resolves it server-side from the x-api-key on POST, so the wire records
# omit it. The local dump produced by this script also omits it for fidelity
# with what the agent actually transmits.
LEDGER_SCHEMA_VERSION = "0.3"

# Default model per provider. Each is the smallest model in its family that
# actually exposes an internal-reasoning trace — picking gpt-4o-mini /
# gemini-2.0-flash / deepseek-chat would silently return no reasoning content.
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"   # extended thinking via thinking={...}
OPENAI_MODEL    = "o4-mini"                     # reasoning surface via Responses API
DEEPSEEK_MODEL  = "deepseek-reasoner"           # exposes message.reasoning_content
GEMINI_MODEL    = "gemini-2.5-flash"            # thinks if include_thoughts=True

# Anthropic extended-thinking knobs (consumed by AnthropicLLM below).
# budget_tokens must be < max_tokens; max_tokens caps the TOTAL output
# (thinking + final text), so leave headroom beyond the budget for the answer.
LLM_MAX_TOKENS      = 2400
LLM_THINKING_BUDGET = 1024


# === Unified LLM client (4 providers) ========================================
# Each class wraps ONE provider's SDK + the "enable internal reasoning" knob
# + the per-provider response-parsing shape. The rest of the notebook then
# calls llm_client.complete(system_prompt, user_input) and gets back a uniform
# LLMResult (text, internal_reasoning, token usage). One method covers Step 2,
# 3, 4d, 5, 6b — switching provider is a one-line change in the cell that
# picks llm_client.
#
# `internal_reasoning` is the schema name from the Reasoning Ledger v0.3
# ModelInvocation field; LLMResult.to_model_invocation() returns a dict that
# drops in directly to a Thinking record's `model_invocation` slot.
from dataclasses import dataclass


@dataclass
class LLMResult:
    provider:           str
    model_name:         str
    text:               str               # the model's final answer
    internal_reasoning: str               # raw chain-of-thought; "" if none
    tokens_in:          int | None
    tokens_out:         int | None

    def to_model_invocation(self) -> dict:
        """Build a Reasoning Ledger v0.3 ModelInvocation dict. The
        `internal_reasoning` field is only emitted when there's a trace to
        record (per schema's "alongside and distinct from final output" rule)."""
        mi = {
            "provider":   self.provider,
            "model_name": self.model_name,
            "tokens_in":  self.tokens_in,
            "tokens_out": self.tokens_out,
        }
        if self.internal_reasoning:
            mi["internal_reasoning"] = self.internal_reasoning
        return mi


class AnthropicLLM:
    """Anthropic Claude with extended thinking.
    Knob  : thinking={"type":"enabled","budget_tokens":N}
    Shape : resp.content[] blocks; block.type=='thinking' carries .thinking"""
    provider = "anthropic"

    def __init__(self, api_key: str,
                 model:           str = ANTHROPIC_MODEL,
                 max_tokens:      int = LLM_MAX_TOKENS,
                 thinking_budget: int = LLM_THINKING_BUDGET):
        import anthropic
        self._client      = anthropic.Anthropic(api_key=api_key)
        self.model_name   = model
        self.max_tokens   = max_tokens
        self.thinking_cfg = {"type": "enabled", "budget_tokens": thinking_budget}

    def complete(self, system_prompt: str, user_input: str) -> LLMResult:
        resp = self._client.messages.create(
            model       = self.model_name,
            max_tokens  = self.max_tokens,
            thinking    = self.thinking_cfg,
            system      = system_prompt,
            messages    = [{"role": "user", "content": user_input}],
        )
        text_parts, thinking_parts = [], []
        for block in resp.content:
            if   block.type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif block.type == "text":
                text_parts.append(    getattr(block, "text",     "") or "")
        return LLMResult(
            provider           = self.provider,
            model_name         = self.model_name,
            text               = "".join(text_parts),
            internal_reasoning = "\n\n".join(thinking_parts),
            tokens_in          = resp.usage.input_tokens,
            tokens_out         = resp.usage.output_tokens,
        )


class OpenAILLM:
    """OpenAI reasoning models via the *Responses API* (Chat Completions hides
    the trace even on o-series models — you're billed reasoning_tokens but get
    nothing back).
    Knob  : reasoning={"effort": ..., "summary": "auto"}   (summary requires verified org)
    Shape : resp.output[] items; type=='reasoning' -> .summary[].text"""
    provider = "openai"

    def __init__(self, api_key: str,
                 model:  str = OPENAI_MODEL,
                 effort: str = "medium"):
        from openai import OpenAI
        self._client    = OpenAI(api_key=api_key)
        self.model_name = model
        self.effort     = effort

    def complete(self, system_prompt: str, user_input: str) -> LLMResult:
        resp = self._client.responses.create(
            model     = self.model_name,
            reasoning = {"effort": self.effort, "summary": "auto"},
            input     = [{"role": "system", "content": system_prompt},
                         {"role": "user",   "content": user_input}],
        )
        text_parts, thinking_parts = [], []
        for item in getattr(resp, "output", []) or []:
            itype = getattr(item, "type", None)
            if itype == "reasoning":
                for s in getattr(item, "summary", []) or []:
                    t = getattr(s, "text", None)
                    if t:
                        thinking_parts.append(t)
            elif itype == "message":
                for c in getattr(item, "content", []) or []:
                    t = getattr(c, "text", None)
                    if t:
                        text_parts.append(t)
        usage = getattr(resp, "usage", None)
        return LLMResult(
            provider           = self.provider,
            model_name         = self.model_name,
            text               = "\n".join(text_parts),
            internal_reasoning = "\n".join(thinking_parts),
            tokens_in          = getattr(usage, "input_tokens",  None),
            tokens_out         = getattr(usage, "output_tokens", None),
        )


class DeepSeekLLM:
    """DeepSeek on the OpenAI-compatible Chat Completions endpoint.
    Knob  : model='deepseek-reasoner' (no extra param; the model controls reasoning)
    Shape : resp.choices[0].message.reasoning_content"""
    provider = "deepseek"

    def __init__(self, api_key: str,
                 model: str = DEEPSEEK_MODEL):
        from openai import OpenAI
        self._client    = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model_name = model

    def complete(self, system_prompt: str, user_input: str) -> LLMResult:
        resp = self._client.chat.completions.create(
            model    = self.model_name,
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_input}],
        )
        msg = resp.choices[0].message
        return LLMResult(
            provider           = self.provider,
            model_name         = self.model_name,
            text               = (getattr(msg, "content",           "") or ""),
            internal_reasoning = (getattr(msg, "reasoning_content", "") or ""),
            tokens_in          = resp.usage.prompt_tokens,
            tokens_out         = resp.usage.completion_tokens,
        )


class GeminiLLM:
    """Google Gemini 2.5+ with thinking opt-in.
    Knob  : thinking_config(include_thoughts=True)   (2.5 thinks internally but hides by default)
    Shape : resp.candidates[].content.parts[]; part.thought==True -> trace"""
    provider = "gemini"

    def __init__(self, api_key: str,
                 model: str = GEMINI_MODEL):
        from google import genai
        from google.genai import types as gtypes
        self._client    = genai.Client(api_key=api_key)
        self._types     = gtypes
        self.model_name = model

    def complete(self, system_prompt: str, user_input: str) -> LLMResult:
        resp = self._client.models.generate_content(
            model    = self.model_name,
            contents = user_input,
            config   = self._types.GenerateContentConfig(
                system_instruction = system_prompt,
                thinking_config    = self._types.ThinkingConfig(include_thoughts=True),
            ),
        )
        text_parts, thinking_parts = [], []
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                t = getattr(part, "text", None)
                if not t:
                    continue
                if getattr(part, "thought", False):
                    thinking_parts.append(t)
                else:
                    text_parts.append(t)
        um = getattr(resp, "usage_metadata", None)
        return LLMResult(
            provider           = self.provider,
            model_name         = self.model_name,
            text               = "\n".join(text_parts),
            internal_reasoning = "\n".join(thinking_parts),
            tokens_in          = getattr(um, "prompt_token_count",     None) if um else None,
            tokens_out         = getattr(um, "candidates_token_count", None) if um else None,
        )


# --- Sanity check: catch a placeholder key NOW, not 6 cells from now. ---
_missing = [n for n, v in [("ARENA_KEY", ARENA_KEY), ("ANTHROPIC_KEY", ANTHROPIC_KEY)]
            if "FILL IN" in v]
if _missing:
    print(f"WARNING: still need to set {', '.join(_missing)} (edit this cell first).")
else:
    print("Both API keys are set.")
print(f"Arena  : {ARENA}")
print(f"Models : {ANTHROPIC_MODEL} | {OPENAI_MODEL} | {DEEPSEEK_MODEL} | {GEMINI_MODEL}")
print(f"Season : World Cup 2026 (id {SPORTMONKS_SEASON_ID})")
print("Setup complete -- run the cells below in order.")


# ## Step 1 · Find the matches (fixtures) and pick one

# A **fixture** is one scheduled match. Before the agent can analyze anything, it needs to know which matches exist — so we ask Sportmonks for the **season schedule**: every stage, round, and fixture for World Cup 2026.
# 
# We call the arena's Sportmonks **proxy** (not Sportmonks directly): the arena forwards the request with its own Sportmonks key and wraps the reply in an envelope — `{body, statusCode, requestId, …}` — so we "peel" `body` → `data` to get the actual schedule.
# 
# For this walkthrough we hard-code the opener, **Mexico vs South Africa** (`fixture_id 19609127`), as the fixture to reason about. Your agent would instead loop over the schedule and pick fixtures itself.
# 
# - Original Sportmonks endpoint: `GET /v3/football/schedules/seasons/{SEASON_ID}`
# - Sportmonks WC2026 guide: https://docs.sportmonks.com/v3/world-cup-2026/how-to-build-your-world-cup-application

# In[43]:


r = requests.get(
    f"{SPORTMONKS_PROXY}/schedules/seasons/{SPORTMONKS_SEASON_ID}",
    headers=H_ARENA, timeout=10,
)
r.raise_for_status()

# Every arena proxy call wraps the upstream reply in an "envelope":
#   {body, duration, statusCode, requestId, _proxy, headers}
# The real Sportmonks payload lives under envelope["body"]["data"].
envelope = r.json()
schedule = envelope["body"]["data"]

print(f"HTTP {r.status_code} (OK) -- the arena answered.")
print(f"Envelope keys from the proxy: {list(envelope.keys())}")
print(f"Found {len(schedule)} schedule entries (stages / rounds / fixtures) for WC2026.\n")

# A real agent would scan `schedule` and pick fixtures itself. For this guide we
# hard-code the tournament opener so everyone analyzes the same match:
#   Mexico (MEX) vs South Africa (ZAF) -- 2026-06-11 -- fixture_id 19609127
SPORTMONKS_FIXTURE_ID = 19609127
print(f"Chosen fixture: Mexico vs South Africa (fixture_id {SPORTMONKS_FIXTURE_ID})")


# ### Find the matching Polymarket market (the "event slug")
# 
# To bet on a match we need its market on Polymarket. The arena keeps a curated **fixture ↔ Polymarket-event mapping** so you don't have to match them by hand.
# 
# We only need one field from it — the **`polymarket_event_slug`**, Polymarket's id for this match's market (e.g. `fifwc-mex-rsa-2026-06-11`). Everything else about the market (prices, token ids) we fetch live from Polymarket in Step 3. If a fixture has no mapping, the slug is `None` and the agent simply runs in predict-only mode.

# In[44]:


r = requests.get(
    f"{ARENA}/api/v1/web/mapping",
    params={"fixture_id": SPORTMONKS_FIXTURE_ID},
    headers=H_ARENA, timeout=10,
)
r.raise_for_status()
mappings = r.json().get("mappings") or []
polymarket_event_slug = mappings[0]["polymarket_event_slug"] if mappings else None
print(f"HTTP {r.status_code} (OK)")
if polymarket_event_slug:
    print(f"This fixture maps to Polymarket event slug: {polymarket_event_slug!r}")
    print("We'll use this slug in Step 3 to pull the live market.")
else:
    print("No Polymarket market is mapped to this fixture.")
    print("That's fine -- the agent will run in predict-only mode (no betting).")


# ## Step 2 · Fetch the match's pre-game data and summarize it

# Now pull the **pre-game signals** for our chosen fixture. We ask Sportmonks to "include" several related pieces of data in one call. (The `statistics` include only fills in *after* a match, so we skip it and request these instead:)
# 
# | Include | What it gives us |
# |---------|------------------|
# | `participants` | The two teams, with home/away labels and short codes (e.g. MEX, RSA) |
# | `predictions` | Sportmonks' own machine-learning model probabilities for win / draw / loss |
# | `odds` | Bookmaker prices — we'll average them into a "consensus" probability |
# | `xGFixture` | Expected-goals projection for each team |
# 
# Same envelope as Step 1, so again we peel `body` → `data`. We then split the two `participants` into `home` and `away`.
# 
# - Sportmonks doc: https://docs.sportmonks.com/v3/endpoints-and-entities/endpoints/fixtures/get-fixture-by-id

# In[45]:


r = requests.get(
    f"{SPORTMONKS_PROXY}/fixtures/{SPORTMONKS_FIXTURE_ID}",
    params={"include": "participants;predictions;odds;xGFixture"},
    headers=H_ARENA, timeout=60,
)
r.raise_for_status()
fixture = r.json()["body"]["data"]    # same envelope as Step 1: peel body -> data

home = next(p for p in fixture["participants"] if p["meta"]["location"] == "home")
away = next(p for p in fixture["participants"] if p["meta"]["location"] == "away")

print(f"HTTP {r.status_code} (OK)")
print(f"Fixture : {fixture['name']}")
print(f"Kickoff : {fixture.get('starting_at')}")
print(f"Home    : {home['name']} ({home['short_code']})")
print(f"Away    : {away['name']} ({away['short_code']})")
print("\nHow much pre-game data came back? (empty rows are possible on staging)")
print(f"  - Sportmonks model predictions : {len(fixture.get('predictions') or [])} rows")
print(f"  - bookmaker odds               : {len(fixture.get('odds') or [])} rows")
print(f"  - expected-goals (xG)          : {len(fixture.get('xgfixture') or [])} rows")


# ### Summarize the match data with an LLM
# 
# The raw Sportmonks payload is large and noisy. Here we hand it to Claude with strict instructions to return a **small, clean JSON summary** (a "digest"): model probabilities, bookmaker consensus, expected goals, plus an honest note of what's missing.
# 
# Why bother? Later steps (the prediction in Step 5) only need the distilled signals, not hundreds of raw rows. Digesting now keeps every downstream prompt small, cheap, and consistent. This "fetch → digest" pattern repeats in Steps 3 and 4.
# 
# Note: every provider's class wrapper has its "expose internal reasoning" knob turned on by default — so `llm_digest.text` carries the final answer and `llm_digest.internal_reasoning` carries the chain-of-thought trace (empty if the chosen model can't expose one). A regex pulls the JSON object out of `.text`.

# In[46]:


# Pick ONE provider. The classes from the Setup cell wrap the SDK + the
# "enable internal reasoning" knob + the per-provider response-parsing shape,
# so the rest of the notebook stays provider-agnostic — every step just calls
# `llm_client.complete(system_prompt, user_input)` and reads `.text` +
# `.internal_reasoning` off the LLMResult. Comment out the three you don't use.
llm_client = AnthropicLLM(api_key=ANTHROPIC_KEY)
# llm_client = OpenAILLM  (api_key=OPENAI_API_KEY)     # needs verified org for reasoning summaries
# llm_client = DeepSeekLLM(api_key=DEEPSEEK_API_KEY)   # uses deepseek-reasoner by default
# llm_client = GeminiLLM  (api_key=GEMINI_API_KEY)     # uses gemini-2.5-flash by default
print(f"LLM client: {llm_client.provider} ({llm_client.model_name})")

DIGEST_SYS = (
    "You are a soccer analyst. You receive ONE Sportmonks 1X2 winner prediction "
    "row + ONE example bookmaker's complete 1X2 quote (three rows, one per "
    "outcome) + xG entries for one fixture, and must distil them into a "
    "self-contained JSON digest that a downstream LLM (with no other context "
    "about Sportmonks) will read.\n\n"

    "## Input shape\n"
    "  - fixture       : match name (e.g. 'Mexico vs South Africa')\n"
    "  - home_code     : home team short code (use as a JSON key for the home outcome)\n"
    "  - away_code     : away team short code (use as a JSON key for the away outcome)\n"
    "  - prediction    : ONE Sportmonks 1X2 winner prediction row (type_id 237), or null. "
    "                    Has a `predictions` object keyed by `home` / `draw` / `away` with "
    "                    percentage probabilities (0-100). Divide by 100 to express in 0..1.\n"
    "  - odds          : ONE example bookmaker's COMPLETE 1X2 winner quote — a list of "
    "                    three rows (or empty list). Each row: `bookmaker_id`, "
    "                    `market_id` (always 1 here = 1X2 winner), `label` "
    "                    ('Home' / 'Draw' / 'Away'), `value` (decimal odds), "
    "                    `probability` (implied probability as a percentage 0-100). "
    "                    Divide `probability` by 100 to express in 0..1. NOTE: a real "
    "                    agent would aggregate across many bookmakers; this single "
    "                    bookmaker is illustrative only.\n"
    "  - xGFixture[]   : expected-goals entries per team. Each row has "
    "                    `participant_id` (team id matching home/away participant) and "
    "                    `value` (xG number). May be empty.\n\n"

    "## Output schema (return ONLY this JSON — no prose, no code fences)\n"
    "{\n"
    "  'fixture'                       : str,                                                          // echo input\n"
    "  'home_team'                     : str,                                                          // home_code\n"
    "  'away_team'                     : str,                                                          // away_code\n"
    "  'sportmonks_ml_win_prob'        : {home_code: float, 'draw': float, away_code: float} | null,   // probabilities in 0..1; sum ≈ 1\n"
    "  'bookmaker_example_win_prob'    : {home_code: float, 'draw': float, away_code: float} | null,   // probabilities in 0..1; from ONE bookmaker only\n"
    "  'expected_goals'                : {home_code: float, away_code: float} | null,                  // xG per side\n"
    "  'data_availability': {                                                                          // honest reporting so downstream knows what's missing\n"
    "    'sportmonks_ml'        : 'available' | 'missing',\n"
    "    'bookmaker_example'    : 'available' | 'missing',\n"
    "    'expected_goals'       : 'available' | 'missing'\n"
    "  },\n"
    "  'summary': str   // 1-3 sentences. MUST be readable in isolation by an LLM that has no other Sportmonks context. Name the available signals and what they imply. Flag explicitly that bookmaker_example is from a single bookmaker, not a consensus.\n"
    "}\n\n"

    "Use null (not 0) when source data is missing. Do NOT fabricate values."
)

# Sportmonks returns dozens of prediction rows + dozens of bookmakers' odds —
# way more than the LLM needs (and far more than fits in a context window).
# Pre-filter to:
#   - top_prediction : the type_id=237 (Full-Time-Result / 1X2 winner) row
#   - top_odds       : ONE bookmaker's COMPLETE 1X2 quote (3 rows: Home/Draw/Away).
#                      A single odds row only carries ONE outcome's price; we need
#                      all three to form a probability picture. Picking the first
#                      qualifying bookmaker keeps the input small and forensically
#                      reproducible (no averaging, no random sampling).
# These same `top_*` slices are also what the ledger logs in Step 8, so the
# Thinking record's input_payload matches what the LLM actually saw.
SPORTMONKS_1X2_PREDICTION_TYPE_ID = 237
SPORTMONKS_1X2_MARKET_ID          = 1
SPORTMONKS_1X2_LABELS             = {"Home", "Draw", "Away"}

_all_predictions = fixture.get("predictions") or []
_all_odds        = fixture.get("odds")        or []
top_prediction   = next(
    (p for p in _all_predictions
     if p.get("type_id") == SPORTMONKS_1X2_PREDICTION_TYPE_ID),
    None,
)
# Group 1X2 odds by bookmaker, pick first one with a full Home/Draw/Away quote.
_1x2_by_bm: dict = {}
for o in _all_odds:
    if o.get("market_id") != SPORTMONKS_1X2_MARKET_ID:
        continue
    _1x2_by_bm.setdefault(o.get("bookmaker_id"), {})[o.get("label")] = o
top_odds = next(
    (list(rows.values())
     for rows in _1x2_by_bm.values()
     if SPORTMONKS_1X2_LABELS.issubset(rows)),
    [],
)

# The user message is identical across providers, so build it once.
digest_input = json.dumps({
    "fixture":    fixture["name"],
    "home_code":  home["short_code"],
    "away_code":  away["short_code"],
    "prediction": top_prediction,
    "odds":       top_odds,
    "xGFixture":  fixture.get("xgfixture"),    # field is lowercase despite include name
})

llm_digest = llm_client.complete(DIGEST_SYS, digest_input)

# The model returns the digest as text; pull the {...} object out of it
# (re.DOTALL lets the regex span newlines; also strips any prose/code fences).
import re
match = re.search(r"\{.*\}", llm_digest.text, re.DOTALL)
sportmonks_digest = json.loads(match.group(0)) if match else None

print(f"The model reasoned for {len(llm_digest.internal_reasoning)} chars before answering.")
print("Clean digest the rest of the notebook will use:\n")
print(json.dumps(sportmonks_digest, indent=2))


# ## Step 3 · Fetch the betting market and its prices, then summarize it

# Now we get the actual **market** we could trade on. The "who wins?" market is the **moneyline**, with three outcomes: home win, draw, away win. On Polymarket each outcome is its own yes/no market, and the three are grouped under one **event**.
# 
# Polymarket exposes two APIs (both reached through the arena proxy):
# 
# | API | What we ask it | What we get back |
# |-----|----------------|------------------|
# | **Gamma** (`/events?slug=…`) | the event by its slug | the event + its 3 child markets, each with a `conditionId` and `clobTokenIds` (a YES and a NO token) |
# | **CLOB** (`/midpoint?token_id=…`) | a YES token's price | the live **mid** price (0–1), i.e. the implied probability of that outcome |
# 
# So the recipe is: call Gamma once to get the three markets and their token ids → call CLOB once per YES token for its live price → assemble one tidy `moneyline` dict.
# 
# How we label each market: Polymarket's WC2026 events follow a naming convention — the event "ticker" is `fifwc-{home}-{away}-{YYYY-MM-DD}`, and each child market's slug is that ticker plus `-{team_code}` or `-draw`. We parse those to tag each market as home / draw / away.

# In[47]:


import re
TICKER_RE = re.compile(r"^fifwc-([a-z]{2,4})-([a-z]{2,4})-(\d{4}-\d{2}-\d{2})$")


def _clob_mid(token_id_str: str) -> float:
    """Single CLOB midpoint call. Polymarket's CLOB takes the token id as a
    decimal string (the raw value is a 78-digit integer)."""
    if not token_id_str:
        return None
    try:
        resp = requests.get(
            f"{POLYMARKET_CLOB}/midpoint",
            params={"token_id": token_id_str},
            headers=H_ARENA, timeout=10,
        )
        if not resp.ok:
            return None
        body = resp.json().get("body")
        if isinstance(body, dict) and "mid" in body:
            return float(body["mid"])
    except Exception:
        pass
    return None


def _outcome_from_market_slug(market_slug: str, ticker: str,
                              home_code: str, away_code: str) -> str:
    """Map a child-market slug ('fifwc-mex-rsa-2026-06-11-mex') to an
    outcome key ('home' | 'draw' | 'away')."""
    if not market_slug.startswith(ticker + "-"):
        return None
    suffix = market_slug[len(ticker) + 1:]
    if suffix == home_code: return "home"
    if suffix == "draw":    return "draw"
    if suffix == away_code: return "away"
    return None


if not polymarket_event_slug:
    moneyline = None
else:
    # 3a · Gamma: one call returns the event + its 3 child markets.
    r = requests.get(
        f"{POLYMARKET_GAMMA}/events",
        params={"slug": polymarket_event_slug},
        headers=H_ARENA, timeout=15,
    )
    r.raise_for_status()
    events = r.json().get("body") or []
    event  = events[0] if events else None

    if event is None:
        moneyline = None
    else:
        ticker = (event.get("ticker") or "").lower()
        m = TICKER_RE.match(ticker)
        if not m:
            moneyline = None
        else:
            pm_home_code, pm_away_code, _ = m.groups()
            outcomes = {}
            for mkt in (event.get("markets") or []):
                key = _outcome_from_market_slug((mkt.get("slug") or "").lower(),
                                                ticker, pm_home_code, pm_away_code)
                if key is None:
                    continue
                # clobTokenIds is a JSON-encoded string: [YES_token, NO_token].
                try:
                    token_ids = json.loads(mkt.get("clobTokenIds") or "[]")
                except json.JSONDecodeError:
                    token_ids = []
                token_yes = token_ids[0] if token_ids else None
                # Use Sportmonks-side short codes (e.g. ZAF, not Polymarket's
                # RSA) so the same team_code flows from this digest into the
                # strategy LLM and onward to the arena /orders endpoint, which
                # validates team_code against Sportmonks codes.
                outcomes[key] = {
                    "team_code":       key if key == "draw" else (
                                            home["short_code"] if key == "home"
                                            else away["short_code"]),
                    "condition_id":    mkt.get("conditionId"),
                    "token_yes":       token_yes,
                    "current_mid_yes": _clob_mid(token_yes),  # 3b · one CLOB call per YES token
                }

            moneyline = {
                "sportmonks_match_id":   SPORTMONKS_FIXTURE_ID,
                "fixture":               event.get("title"),
                "kickoff_utc":           event.get("startDate"),
                "polymarket_event_slug": polymarket_event_slug,
                "outcomes":              outcomes,
            }

if moneyline is None:
    print("No tradable Polymarket market for this fixture -- predict-only mode.")
else:
    n_mids = sum(1 for o in moneyline["outcomes"].values()
                 if o["current_mid_yes"] is not None)
    print(f"Built the 3-way moneyline for: {moneyline['fixture']}")
    print(f"Live mid prices retrieved for {n_mids}/3 outcomes (home / draw / away).")
    print("Full market (prices + the token ids needed to place an order):\n")
print(json.dumps(moneyline, indent=2, default=str))


# ### Summarize the market with an LLM
# 
# Same pattern as Step 2: distill the raw market response into a self-contained JSON. The digest reports each outcome's **implied probability** (from the mid prices), whether the probabilities sum to ~1 (a sanity check for stale prices), and the **execution handles** (`condition_id` + `token_yes`) a later step needs to actually place an order. If there's no market for this fixture, we emit a clearly-labeled "no market" digest instead.

# In[48]:


POLYMARKET_DIGEST_SYS = (
    "You are an analyst digesting a Polymarket moneyline (3-way match-winner) "
    "market response into a self-contained JSON for a downstream LLM that has "
    "no other Polymarket context.\n\n"

    "## Input shape\n"
    "  - sportmonks_match_id   : numeric fixture id (echo)\n"
    "  - fixture               : match name (e.g. 'Mexico vs South Africa')\n"
    "  - kickoff_utc           : ISO kickoff timestamp\n"
    "  - polymarket_event_slug : Polymarket event slug grouping the 3 binary markets\n"
    "  - outcomes.{home,draw,away}\n"
    "      team_code           : team short code (or 'draw' for the draw outcome)\n"
    "      condition_id        : Polymarket condition id (needed for trade execution)\n"
    "      token_yes           : ERC1155 YES-side token id (buy YES to back the outcome)\n"
    "      current_mid_yes     : midpoint price of the YES token in 0..1 == implied probability\n"
    "                            of that outcome winning. null if CLOB lookup failed.\n\n"

    "## Output schema (return ONLY this JSON — no prose, no code fences)\n"
    "{\n"
    "  'fixture'              : str,\n"
    "  'market_handle'        : str,                                                          // polymarket_event_slug\n"
    "  'implied_win_prob'     : {home_code: float, 'draw': float, away_code: float} | null,   // from current_mid_yes; null if unavailable\n"
    "  'sum_implied_prob'     : float | null,                                                 // should be ≈1.0; outside [0.95, 1.10] = stale prices or arb gap\n"
    "  'execution_handles'    : {home_code: {condition_id, token_yes},                        // for the downstream trade-execution step\n"
    "                            'draw'   : {condition_id, token_yes},\n"
    "                            away_code: {condition_id, token_yes}},\n"
    "  'data_availability'    : 'mids_available' | 'mids_partial' | 'mids_missing' | 'no_market',\n"
    "  'summary'              : str   // 1-3 sentences self-contained. Name the favorite (highest implied prob), the spread, and any anomaly. If mids are missing, say so plainly and identify what's still available (execution handles can still be used to place orders blind).\n"
    "}\n\n"

    "Use null when input shows null. Do NOT fabricate prices."
)

if moneyline is None:
    polymarket_digest = {
        "fixture":              None,
        "market_handle":        None,
        "implied_win_prob":     None,
        "sum_implied_prob":     None,
        "execution_handles":    None,
        "data_availability":    "no_market",
        "summary":              f"No Polymarket moneyline mapping for Sportmonks fixture "
                                f"{SPORTMONKS_FIXTURE_ID}. The fixture either isn't listed on "
                                f"Polymarket yet or its curated mapping is marked no_match.",
    }
else:
    pm_input = json.dumps(moneyline)
    llm_pm   = llm_client.complete(POLYMARKET_DIGEST_SYS, pm_input)
    m = re.search(r"\{.*\}", llm_pm.text, re.DOTALL)
    polymarket_digest = json.loads(m.group(0)) if m else None
    print(f"The model digested the market "
          f"({len(llm_pm.internal_reasoning)} chars of thinking).")

print("\nMarket digest (implied probabilities + execution handles):\n")
print(json.dumps(polymarket_digest, indent=2))


# ## Step 4 · Fetch historical team stats and summarize them

# The arena also ships a **database** (Supabase) of deeper historical stats. We use it in four sub-steps:
#
# | Sub-step | What it does |
# |----------|--------------|
# | **4a · Discover** | Read the catalog to learn which tables and columns exist — no external docs needed |
# | **4b · Bridge** | Map each team's Sportmonks `team_id` → StatsBomb `country_id` via `dim_country` |
# | **4c · Fetch** | Pull the priors rows we want for both teams |
# | **4d · Digest** | Have Claude summarize them into JSON for Step 5 |
#
# ⚠️ **Heads-up — team identifiers differ between systems.** Sportmonks gives us a `team_id` per participant (Mexico = 458, South Africa = 146), but the StatsBomb-derived tables in this database use a separate `country_id` (Mexico = 147, South Africa = 211). Sub-step 4b bridges them with a direct `team_id`-keyed lookup in the `dim_country` table — so the rest of the notebook never has to hard-code ids.

# In[49]:


H_WCA = {"apikey": SUPABASE_KEY, "Accept-Profile": "world_cup_arena"}  # data layer

# Sportmonks side: each participant exposes a team_id (`p["id"]`). The StatsBomb
# tables under the world_cup_arena schema key on a different country_id, so we
# resolve the bridge dynamically in 4b -- no hard-coded ids.
print("Querying the Supabase stats DB. Sportmonks team_ids for this fixture:")
print(f"  {home['name']:20s} (team_id {home['id']})")
print(f"  {away['name']:20s} (team_id {away['id']})")
print("Sub-step 4b will resolve each to its dim_country.country_id.")


# ### 4a · Discover what data exists
#
# A good agent that's new to a database **asks what's there first** instead of guessing table names. The arena exposes a self-describing catalog right inside the `world_cup_arena` schema — one query tells you the tables, their categories, row counts, and descriptions (and, via the columns table, every column's type and meaning). No external docs required.
#
# | Object | What it returns |
# |------|-----------------|
# | `world_cup_arena.catalog_tables` | All available tables and a description of each |
# | `world_cup_arena.catalog_columns` | All columns across every table, with types and descriptions |
# | `world_cup_arena.catalog_full` | Both combined — tables and their columns in one response |
# 
# Below we read `catalog_full`, print every table, then pick **one** priors table (`ads_a_country_style`, which has playing-style indicators) for the example. A real agent could pull several (head-to-head, knockout patterns, etc.) — same pattern, just more rows.

# In[50]:


r = requests.get(
    f"{SUPABASE}/rest/v1/catalog_full",
    params={
        # `columns` is a JSONB array on the view: [{column_name, data_type,
        # description}, …]. Including it here means one round-trip yields the
        # whole data dictionary -- table-level purpose + column-level meaning.
        "select": "table_name,category,row_count,table_description,columns",
        "order":  "category,table_name",
    },
    headers=H_WCA, timeout=30,    # 30s buffer for Supabase serverless cold-start
)
r.raise_for_status()
catalog = r.json()

print(f"HTTP {r.status_code} (OK) -- the catalog lists every table available:\n")
for t in catalog:
    desc   = (t.get("table_description") or "-").replace("\n", " ")[:60]
    cat    = t.get("category") or "?"
    n_cols = len(t.get("columns") or [])
    rows_s = f"{t['row_count']:>5d}" if t.get("row_count") is not None else "    ?"
    print(f"  [{cat:11s}] {t['table_name']:30s}  rows={rows_s}  cols={n_cols:>2d}  - {desc}")

# For this example we fetch ONE table, picked from the list above for its
# playing-style indicators. A real agent could pull more (H2H, KO pattern,
# etc.) the same way -- just more rows in the dict.
WANTED_TABLE = "ads_a_country_style"
print(f"\nFor this walkthrough we'll pull stats from: {WANTED_TABLE}")


# ### 4b · Bridge Sportmonks team_id → StatsBomb country_id
#
# The Sportmonks side gives us a `team_id` per participant; the priors tables
# we're about to query are keyed on a separate `country_id`. `dim_country` in the
# `world_cup_arena` schema bridges them with a dedicated `team_id` column, so
# we can filter directly by the Sportmonks ids and read back the matching
# `country_id`s in one call — no name-string matching required.

# In[50.5]:


r = requests.get(
    f"{SUPABASE}/rest/v1/dim_country",
    params={
        "select":  "team_id,country_id,country_name",
        "team_id": f"in.({home['id']},{away['id']})",
    },
    headers=H_WCA, timeout=30,
)
r.raise_for_status()
country_rows       = r.json()
country_by_team_id = {row["team_id"]: row["country_id"] for row in country_rows}
COUNTRY_A_ID       = country_by_team_id.get(home["id"])
COUNTRY_B_ID       = country_by_team_id.get(away["id"])

print(f"HTTP {r.status_code} (OK) -- bridged {len(country_rows)} team(s) via team_id:")
print(f"  {home['name']:20s} (team_id {home['id']}) -> dim_country.country_id = {COUNTRY_A_ID}")
print(f"  {away['name']:20s} (team_id {away['id']}) -> dim_country.country_id = {COUNTRY_B_ID}")


# ### 4c · Fetch the stats for both teams
#
# Now query the chosen table for just our two teams. Supabase uses PostgREST-style query params: `country_id=in.(147,211)` means "rows where country_id is 147 or 211," and `select=*` returns all columns. We send the `world_cup_arena` schema header so the request hits the arena's data tables.

# In[51]:


r = requests.get(
    f"{SUPABASE}/rest/v1/{WANTED_TABLE}",
    params={"country_id": f"in.({COUNTRY_A_ID},{COUNTRY_B_ID})", "select": "*"},
    headers=H_WCA, timeout=30,
)
r.raise_for_status()
priors_rows = r.json()

print(f"HTTP {r.status_code} (OK) -- pulled '{WANTED_TABLE}' for both teams.")
print(f"Got {len(priors_rows)} row(s) (one per team that has data).")
print("Raw rows (the full stats Claude will summarize next):\n")
print(json.dumps(priors_rows, indent=2, default=str))


# ### 4d · Summarize the stats with an LLM
# 
# The digest pattern once more: Claude turns the raw rows into a compact per-team profile (set-piece efficiency, goals per game, etc.) and flags small-sample caveats. It deliberately does **not** output a win probability — that's Step 5's job.

# In[52]:


SUPABASE_DIGEST_SYS = (
    "You are an analyst aggregating Supabase priors data for one fixture into "
    "a self-contained JSON digest for a downstream LLM that has no other "
    "context about the data layer.\n\n"

    "## Input shape\n"
    "  - fixture        : match name\n"
    "  - source_table   : the Supabase table the rows came from (echo for traceability)\n"
    "  - home_code, away_code : team short codes (use as JSON keys for the output)\n"
    "  - home_id,   away_id   : country ids in this dataset (StatsBomb numbering)\n"
    "  - rows           : list of rows from `ads_a_country_style`, one per country.\n"
    "      Key columns include: country_id, set_piece_shots, set_piece_goals,\n"
    "      conversion_rate, group_matches, group_goals_against, ko_matches,\n"
    "      ko_goals_against, group_gpg (goals/game in group stage),\n"
    "      ko_gpg (goals/game in knockout stage).\n"
    "      Match each row to home_id / away_id by country_id. Sample sizes are\n"
    "      often tiny in this dataset — call that out if it impacts confidence.\n\n"

    "## Output schema (return ONLY this JSON — no prose, no code fences)\n"
    "{\n"
    "  'fixture'      : str,\n"
    "  'source_table' : str,                                              // echo\n"
    "  'teams': {\n"
    "    home_code: {                                                     // team A profile from country_style\n"
    "      'set_piece_efficiency' : float | null,                          // set_piece_goals / set_piece_shots\n"
    "      'set_piece_sample'     : int   | null,                          // set_piece_shots — proxies confidence\n"
    "      'group_goals_per_game' : float | null,\n"
    "      'ko_goals_per_game'    : float | null\n"
    "    },\n"
    "    away_code: { same shape }\n"
    "  },\n"
    "  'data_availability': 'rich' | 'partial' | 'sparse',\n"
    "  'summary': str   // 1-3 sentences self-contained. Name the two teams, the strongest\n"
    "                 // style signal, and call out small-sample caveats. Do NOT give a\n"
    "                 // win probability — that's for §5 combined analysis.\n"
    "}\n\n"

    "Use null when input is empty/missing. Don't fabricate values."
)

sb_input = json.dumps({
    "fixture":      fixture["name"],
    "source_table": WANTED_TABLE,
    "home_code":    home["short_code"],
    "away_code":    away["short_code"],
    "home_id":      COUNTRY_A_ID,
    "away_id":      COUNTRY_B_ID,
    "rows":         priors_rows,
}, default=str)

llm_sb = llm_client.complete(SUPABASE_DIGEST_SYS, sb_input)
m = re.search(r"\{.*\}", llm_sb.text, re.DOTALL)
supabase_digest = json.loads(m.group(0)) if m else None

print(f"The model summarized the stats "
      f"({len(llm_sb.internal_reasoning)} chars of thinking).")
print("Per-team stats digest for Step 5:\n")
print(json.dumps(supabase_digest, indent=2))


# ## Step 5 · Predict the result (the agent's own opinion)
# 
# Now the agent forms its **own** view of the match, combining the Sportmonks and Supabase digests into a single prediction: the most likely outcome, a probability, and a rationale.
# 
# **The market is deliberately left out here.** We want an opinion formed independently of Polymarket — otherwise the agent would just parrot the market price. Comparing this independent view against the market in Step 6 is exactly where any **edge** comes from.

# In[53]:


PREDICT_SYS = (
    "You are a soccer match analyst. You receive two pre-distilled digests for "
    "one fixture and must produce the agent's own outcome prediction.\n\n"

    "## Input shape\n"
    "  - fixture           : match name\n"
    "  - home_code         : home team short code (use as a JSON key for the home outcome)\n"
    "  - away_code         : away team short code (use as a JSON key for the away outcome)\n"
    "  - sportmonks_digest : digest of Sportmonks pre-match data (model probs, bookmaker\n"
    "                        consensus, expected goals). Values may be null when staging\n"
    "                        hasn't seeded the data — see its `data_availability` flag.\n"
    "  - supabase_digest   : digest of long-horizon priors (playing style, set-piece\n"
    "                        efficiency, group/KO goals-per-game). Note its sample-size\n"
    "                        caveats.\n\n"

    "## Output schema (return ONLY this JSON — no prose, no code fences)\n"
    "{\n"
    "  'fixture'    : str,\n"
    "  'outcome'    : str,                          // home_code | 'draw' | away_code — the most likely outcome\n"
    "  'probability': float,                        // 0..1; confidence in `outcome`\n"
    "  'rationale'  : str,                          // 1-3 sentences self-contained. Name the teams,\n"
    "                                               // the signals you leaned on, and major caveats.\n"
    "  'used_signals': {                            // for traceability into §6\n"
    "    'sportmonks' : 'leaned_on' | 'unavailable',\n"
    "    'supabase'   : 'leaned_on' | 'unavailable'\n"
    "  },\n"
    "  'confidence_level': 'high' | 'medium' | 'low'   // honest about how thin your evidence was\n"
    "}\n\n"

    "Be honest about uncertainty: if both digests are sparse, low confidence is the right answer. "
    "Do NOT consult the market (you don't have it). Probability must reflect what the priors say "
    "alone — anchoring to a market mid would defeat the point."
)

predict_input = json.dumps({
    "fixture":           fixture["name"],
    "home_code":         home["short_code"],
    "away_code":         away["short_code"],
    "sportmonks_digest": sportmonks_digest,
    "supabase_digest":   supabase_digest,
})

llm_predict = llm_client.complete(PREDICT_SYS, predict_input)
m = re.search(r"\{.*\}", llm_predict.text, re.DOTALL)
prediction = json.loads(m.group(0)) if m else None

print(f"The agent formed its own prediction "
      f"({len(llm_predict.internal_reasoning)} chars of thinking):\n")
print(json.dumps(prediction, indent=2))
if prediction:
    print(f"\n-> In plain words: most likely '{prediction['outcome']}' at "
          f"{prediction['probability']:.0%} confidence ({prediction['confidence_level']}).")


# ## Step 6 · Decide whether to bet (turn the prediction into a trade)
#
# Two sub-steps:
#
# | Sub-step | What it does |
# |----------|--------------|
# | **6a · Wallet** | Fetch the agent's wallet via `/v1/arena/agents/me` so the strategy knows the real available balance |
# | **6b · Strategy** | Compare prediction vs market, decide size + limit (or skip) — capped by the wallet from 6a |
#
# The key idea is **edge** = (the agent's probability) − (the market's implied probability) for the same outcome:
#
# - **Positive edge** → the market is *underpricing* the pick → consider going **long** (back it).
# - **Negative edge** → the market is *overpricing* it → consider going **short** (fade it).
# - **Tiny edge** (noise) → don't trade.
#
# Position size scales with the edge, the agent's confidence (shrinks when confidence is low), **and is capped by the available balance fetched in 6a**. With a small wallet and weak conviction, not betting is a perfectly good answer.

# ### 6a · Check the agent's wallet
#
# Fetch the wallet so we know:
#
# - **`available_balance_usdc`** — how much USDC we can actually spend on a new order.
# - **`locked_balance_usdc`** — USDC already reserved by open orders / unsettled positions.
# - **`wallet.address`** — the on-chain Polymarket-side wallet. The agent's funder is what tops it up; `polymarket_profile_url` opens the public profile.
#
# We pass `available_balance_usdc` into the strategy LLM in 6b so it sizes the trade against the real wallet, not a hardcoded notional.

# In[56]:


r = requests.get(
    f"{ARENA}/api/v1/arena/agents/me",
    headers=H_ARENA, timeout=10,
)
r.raise_for_status()
agent_info        = r.json()
wallet            = agent_info.get("wallet") or {}
balance_available = float(wallet.get("available_balance_usdc") or 0)
balance_locked    = float(wallet.get("locked_balance_usdc")    or 0)

print(f"HTTP {r.status_code} (OK) -- agent identity + wallet:")
print(f"  agent_id          : {agent_info.get('agent_id')}")
print(f"  display_name      : {agent_info.get('display_name')}  ({agent_info.get('lifecycle_phase')})")
print(f"  wallet address    : {wallet.get('address')}")
print(f"  balance available : ${balance_available:.4f}  USDC")
print(f"  balance locked    : ${balance_locked:.4f}  USDC")
if wallet.get("polymarket_profile_url"):
    print(f"  polymarket profile: {wallet['polymarket_profile_url']}")


# ### 6b · Decide whether to trade
#
# The strategy LLM compares the prediction (Step 5) against the market view (Step 3) and the wallet from 6a. It must respect the available balance — sizing past it would just get the order rejected at submission.

# In[57]:


STRATEGY_SYS = (
    "You are a bankroll manager for an agent's USDC wallet. You receive the "
    "agent's own prediction, the current Polymarket market view, and the agent's "
    "available wallet balance — and decide which specific buy-YES orders to "
    "place (or skip entirely).\n\n"

    "## Key constraint: Polymarket is BUY-YES ONLY\n"
    "You cannot sell or short. To FADE an outcome X you have to buy YES on the "
    "OTHER two outcomes — that bundle pays off iff X loses. So every decision "
    "is expressed as zero, one, or two buy-YES orders. There is no 'short' in "
    "the output.\n\n"

    "## Input shape\n"
    "  - prediction              : {outcome, probability, confidence_level, rationale, …}\n"
    "                              The agent's own view, formed without seeing the market.\n"
    "  - polymarket_digest       : {implied_win_prob, sum_implied_prob, execution_handles,\n"
    "                              market_handle, data_availability, summary}.\n"
    "                              The market's view (implied_win_prob keys match team codes,\n"
    "                              with the third key always 'draw'). Each value is the YES\n"
    "                              mid (= implied probability) of that outcome.\n"
    "  - available_balance_usdc  : float, the agent's spendable USDC right now\n"
    "                              (from /v1/arena/agents/me). Locked balance is excluded.\n\n"

    "## How to decide\n"
    "  1. Edge = prediction.probability − polymarket_digest.implied_win_prob[prediction.outcome]\n"
    "     (in percentage points). Positive edge = market UNDER-prices the pick (back it).\n"
    "     Negative edge = market OVER-prices the pick (fade it).\n"
    "     If |edge| < 5pp, skip — noise.\n"
    "  2. Base TOTAL size (before caps), tuned to a notional $100 bankroll:\n"
    "       |edge| 5-15pp                 → $1-2  (modest position)\n"
    "       |edge| > 15pp                 → $3-5  (high-conviction position)\n"
    "     Then HALVE the size if confidence_level is 'low'; use up to 1.5× if 'high'.\n"
    "     If data_availability is not 'mids_available', skip — you can't price an edge\n"
    "     without mids.\n"
    "  3. CAP the final TOTAL size at min($5.00, available_balance_usdc − 0.05). The\n"
    "     $5.00 is a hard per-cycle ceiling; the wallet term leaves a 5¢ slippage\n"
    "     buffer. If the cap is below $1.00, skip — Polymarket's CLOB enforces a $1\n"
    "     minimum PER ORDER, so anything smaller is just rejected. Round to cents.\n"
    "  4. Build the orders:\n"
    "       POSITIVE EDGE → emit ONE order:\n"
    "         team_code   = prediction.outcome\n"
    "         usd_size    = the capped total\n"
    "         limit_price = current YES mid + 0.02, rounded to cents (cap 0.99)\n"
    "       NEGATIVE EDGE → emit TWO orders, one per OTHER outcome (call them Y and Z):\n"
    "         Split the total proportional to their YES mids:\n"
    "             size_Y = total × mid_Y / (mid_Y + mid_Z)\n"
    "             size_Z = total × mid_Z / (mid_Y + mid_Z)\n"
    "         This gives the same payoff per dollar whether Y or Z wins (a synthetic\n"
    "         fade of prediction.outcome).\n"
    "         IMPORTANT: each individual order must be ≥ $1.00 (CLOB minimum). If a\n"
    "         split would put either side below $1.00, scale BOTH up just enough so\n"
    "         the smaller side hits exactly $1.00. If that scaled total exceeds the\n"
    "         cap from step 3, SKIP — wallet too thin to fade safely.\n"
    "         For each order: limit_price = its own YES mid + 0.02, rounded to cents.\n"
    "         Round each usd_size to cents.\n\n"

    "## Output schema (return ONLY this JSON — no prose, no code fences)\n"
    "{\n"
    "  'should_trade'   : bool,                              // false when we skip\n"
    "  'orders'         : [                                  // 0, 1, or 2 items\n"
    "      {\n"
    "        'team_code'   : str,                            // home_code | 'draw' | away_code\n"
    "        'usd_size'    : float,                          // ≥ $1.00 per order; rounded to cents\n"
    "        'limit_price' : float                           // 0..1; YES mid + ~0.02; rounded to cents\n"
    "      }, ...\n"
    "  ],\n"
    "  'edge_pp'        : float,                             // (agent_prob − market_prob) × 100\n"
    "  'market_handle'  : str,                               // echo polymarket_digest.market_handle\n"
    "  'rationale'      : str                                // 1-3 sentences: the edge, the sizing\n"
    "                                                       // decision, and (for two-order fades)\n"
    "                                                       // the split math you computed.\n"
    "}\n\n"

    "Be conservative: thin wallet, weak conviction, or no mids → skipping is a valid answer."
)

strategy_input = json.dumps({
    "prediction":             prediction,
    "polymarket_digest":      polymarket_digest,
    "available_balance_usdc": balance_available,
})

llm_strategy = llm_client.complete(STRATEGY_SYS, strategy_input)
m = re.search(r"\{.*\}", llm_strategy.text, re.DOTALL)
strategy = json.loads(m.group(0)) if m else None

print(f"The agent decided on a strategy "
      f"({len(llm_strategy.internal_reasoning)} chars of thinking):\n")
print(json.dumps(strategy, indent=2))
if strategy and strategy.get("should_trade") and strategy.get("orders"):
    print(f"\n-> {len(strategy['orders'])} order(s) at edge "
          f"{strategy.get('edge_pp', 0):+.1f} pp:")
    for o in strategy["orders"]:
        print(f"   BUY ${o['usd_size']:.2f} YES of {o['team_code']:>5s} "
              f"@ ≤{o['limit_price']:.2f}")
elif strategy:
    print(f"\n-> In plain words: no trade -- edge {strategy.get('edge_pp', 0):+.1f} points "
          f"isn't worth it for this wallet.")


# ## Step 7 · Place the bet (open positions)
#
# Two sub-steps:
#
# | Sub-step | What it does |
# |----------|--------------|
# | **7a · Orders**   | Build + POST each order Step 6b emitted (zero, one, or two — fades become two synthetic buy-YES orders) |
# | **7b · Exposure** | After the orders, GET `/v1/arena/exposure` to confirm the positions |
#
# Each order has its own random-UUID idempotency key so retries can't double-post. **Predict-only runs are fully supported** — if Step 6b emitted zero orders, 7a prints a notice and 7b still runs so we can see existing positions.

# ### 7a · Build and submit the order(s)
#
# Step 6b emits zero, one, or two orders. When the strategy fades an outcome it emits **two** buy-YES orders on the OTHER two outcomes (Polymarket has no native sell/short), and we submit them as independent POSTs to `/v1/arena/orders`. Each order has its own idempotency key, its own polling loop, and — in Step 8 — its own Acting record.
#
# (The arena `/orders` endpoint may 404 on some deploys; in that case the local payloads still get logged and the ledger Acting records still describe the intent.)

# In[40]:


order_payloads  = []    # one payload per order we attempted to submit
order_responses = []    # parallel: /orders POST response body (or None on 404/exception)
order_outcomes  = []    # parallel: {final_status, tx_hash, clob_order_id, reject_reason} per order

orders_to_place = (strategy or {}).get("orders") or []

if strategy and strategy.get("should_trade") and orders_to_place:
    print(f"\nStrategy says TRADE: submitting {len(orders_to_place)} order(s).")
    for idx, ord_spec in enumerate(orders_to_place, start=1):
        order_payload = {
            "fixture_id":             str(SPORTMONKS_FIXTURE_ID),
            "team_code":              ord_spec["team_code"],
            "usd_size":               f"{float(ord_spec['usd_size']):.2f}",
            "limit_price":            ord_spec["limit_price"],
            "time_in_force_seconds":  30,
            "idempotency_key":        str(uuid.uuid4()),
        }
        order_payloads.append(order_payload)
        print(f"\n--- Order {idx}/{len(orders_to_place)} ---")
        print(json.dumps(order_payload, indent=2))

        # Per-order outcome tracker; populated when the POST + polling lands cleanly.
        outcome = {"final_status": None, "tx_hash": None,
                   "clob_order_id": None, "reject_reason": None}
        order_response = None
        try:
            r = requests.post(
                f"{ARENA}/api/v1/arena/orders",
                headers=H_ARENA, timeout=60,
                json=order_payload,
            )
            if r.status_code == 404:
                print(f"HTTP 404 -- /arena/orders not live on this deploy yet. "
                      f"Payload above is what a real run would send.")
            elif r.ok:
                order_response = r.json()
                order_id = order_response.get("order_id")
                print(f"HTTP {r.status_code} (OK) -- order accepted "
                      f"(order_id={order_id}, status={order_response.get('status')}, "
                      f"locked=${order_response.get('size_usdc_locked')}).")

                # Poll THIS order to a terminal state. 6 × 5s = up to 30s.
                outcome["final_status"] = order_response.get("status")
                for i in range(6):
                    time.sleep(5)
                    got = requests.get(
                        f"{ARENA}/api/v1/arena/orders/{order_id}",
                        headers=H_ARENA, timeout=10,
                    )
                    if not got.ok:
                        continue
                    d = got.json()
                    outcome["final_status"]  = d.get("status")
                    outcome["reject_reason"] = d.get("rejection_reason") or outcome["reject_reason"]
                    fills = d.get("open_fills") or []
                    if fills:
                        outcome["tx_hash"]       = fills[0].get("tx_hash")       or outcome["tx_hash"]
                        outcome["clob_order_id"] = fills[0].get("clob_order_id") or outcome["clob_order_id"]
                    print(f"  poll {i+1}: status={outcome['final_status']}  "
                          f"filled=${d.get('size_usdc_filled')}")
                    if outcome["final_status"] in ("closed", "filled", "rejected"):
                        break

                if outcome["final_status"] in ("filled", "closed") and outcome["tx_hash"]:
                    print(f"  Filled. On-chain settlement tx: "
                          f"https://polygonscan.com/tx/{outcome['tx_hash']}")
                elif outcome["final_status"] == "rejected":
                    print(f"  Rejected. reason: {outcome['reject_reason'] or '(none reported)'}")
                elif outcome["final_status"] not in ("filled", "closed"):
                    print(f"  Status '{outcome['final_status']}' after 30s -- "
                          f"check the dashboard for the final state.")
            else:
                print(f"HTTP {r.status_code} -- order rejected. Body: {r.text[:300]}")
        except Exception as e:
            print(f"Order POST failed: {type(e).__name__}: {e}")

        order_responses.append(order_response)
        order_outcomes.append(outcome)
else:
    print("Strategy says DON'T trade (or emitted zero orders) -- skipping order placement.")
    print("Predict-only runs are fully supported -- Step 8 still records everything.")


# ### 7b · Verify current exposure
#
# After the order (filled, pending, or skipped), GET `/v1/arena/exposure` to see
# every open position the agent currently holds. Each row reports:
#
# - **`fixture_id` + `team_code`** — which outcome of which fixture
# - **`quantity`** — YES shares held
# - **`avg_cost_usdc`** — average price paid per share
# - **`mark_price`** — current mid for that outcome's YES token
# - **`value_usdc`** — `quantity × mark_price`
# - **`unrealized_pnl_usdc`** — `value_usdc - (quantity × avg_cost_usdc)`
#
# This is the cleanest evidence that the order in 7a actually opened a position —
# it should now appear as a row keyed by our fixture + the team_code we bought.

# In[41]:


r = requests.get(
    f"{ARENA}/api/v1/arena/exposure",
    headers=H_ARENA, timeout=10,
)
r.raise_for_status()
exposure  = r.json()
positions = exposure.get("positions") or []

print(f"HTTP {r.status_code} (OK) -- {len(positions)} open position(s):")
if not positions:
    print("  (no open positions)")
for p in positions:
    print(f"  fixture {p.get('fixture_id'):>10s}  {p.get('team_code'):>5s}  "
          f"qty={float(p.get('quantity') or 0):>9.4f}  "
          f"avg_cost=${float(p.get('avg_cost_usdc') or 0):.4f}  "
          f"mark=${float(p.get('mark_price')   or 0):.4f}  "
          f"value=${float(p.get('value_usdc')  or 0):.4f}  "
          f"upnl=${float(p.get('unrealized_pnl_usdc') or 0):+.4f}")

# Highlight rows belonging to *this* fixture so it's obvious whether 7a moved us.
_this = [p for p in positions if str(p.get("fixture_id")) == str(SPORTMONKS_FIXTURE_ID)]
if _this:
    print(f"\n-> {len(_this)} position(s) on this fixture (id {SPORTMONKS_FIXTURE_ID}).")
else:
    print(f"\n-> no positions on fixture {SPORTMONKS_FIXTURE_ID} yet "
          f"(expected if 7a skipped or hasn't filled).")


# # Test to manually place an order
# 

# In[60]:


# order_payload  = None
# order_response = None
# team_code = strategy["outcome"]
# if team_code is not None:
#     order_payload = {
#         "fixture_code":           str(SPORTMONKS_FIXTURE_ID),
#         "team_code":              team_code,
#         "usd_size":               "1.00",
#         "limit_price":            strategy["limit_price"],
#         "time_in_force_seconds":  30,
#         "idempotency_key":        str(uuid.uuid4()),
#     }
#     print("\nStrategy says TRADE. Here's the exact order we'd submit:\n")
#     print(json.dumps(order_payload, indent=2))
#     try:
#         r = requests.post(
#             f"{ARENA}/api/v1/arena/orders",
#             headers=H_ARENA, timeout=60,
#             json=order_payload,
#         )
#         if r.status_code == 404:
#             print("\nHTTP 404 -- /arena/orders not live on this deploy yet. "
#                   "Expected on staging-in-progress; payload above is what a real run would send.")
#         elif r.ok:
#             order_response = r.json()
#             order_id = order_response.get("order_id")
#             print(f"\nHTTP {r.status_code} (OK) -- order accepted "
#                   f"(order_id={order_id}, status={order_response.get('status')}, "
#                   f"locked=${order_response.get('size_usdc_locked')}).")

#             # Poll the order to a terminal state. The execution worker
#             # round-trips to the live Polymarket CLOB; on a freshly funded
#             # wallet a fill typically lands in 5-15s, but allow up to ~30s.
#             final_status   = order_response.get("status")
#             tx_hash        = None
#             clob_order_id  = None
#             reject_reason  = None
#             for i in range(6):                # 6 × 5s = 30s
#                 time.sleep(5)
#                 got = requests.get(
#                     f"{ARENA}/api/v1/arena/orders/{order_id}",
#                     headers=H_ARENA, timeout=10,
#                 )
#                 if not got.ok:
#                     continue
#                 d = got.json()
#                 final_status  = d.get("status")
#                 reject_reason = d.get("rejection_reason") or reject_reason
#                 fills         = d.get("open_fills") or []
#                 if fills:
#                     tx_hash       = fills[0].get("tx_hash")       or tx_hash
#                     clob_order_id = fills[0].get("clob_order_id") or clob_order_id
#                 print(f"  poll {i+1}: status={final_status}  filled=${d.get('size_usdc_filled')}")
#                 if final_status in ("closed", "filled", "rejected"):
#                     break

#             if final_status in ("filled", "closed"):
#                 if tx_hash:
#                     print(f"\nFilled. On-chain settlement tx:\n  https://polygonscan.com/tx/{tx_hash}")
#                 if clob_order_id:
#                     print(f"CLOB order id: {clob_order_id}")
#             elif final_status == "rejected":
#                 print(f"\nOrder rejected. reason: {reject_reason or '(none reported)'}")
#             else:
#                 print(f"\nOrder still '{final_status}' after 30s -- check the dashboard for the final state.")
#         else:
#             print(f"\nHTTP {r.status_code} -- order rejected. Body: {r.text[:300]}")
#     except Exception as e:
#         print(f"\nOrder POST failed: {type(e).__name__}: {e}")


# ## Step 8 · Record the agent's reasoning (the ledger)
# 
# Finally, the agent writes a **ledger**: a structured, step-by-step record of everything it just did. The arena reads this to audit, verify, and score your agent — so a well-formed ledger is how you actually "submit" your work.
# 
# Each record is one node in a graph (`upstream_record_id` links a step to the steps it depended on). The behavior types:
# 
# | Behavior | Meaning |
# |----------|---------|
# | `Observing` | What triggered the run (here, a pretend cron trigger) |
# | `ToolCalling` | An external data call (Sportmonks, Polymarket, Supabase) |
# | `Thinking` | An LLM step (each digest, the prediction, the strategy) |
# | `Acting` | A committed decision — the prediction (always) and an order (only if we traded) |
# 
# This run produces **14 records** (15 when an order is placed). We build them with plain dicts (no SDK) and POST them as one batch. `agent_id` isn't set here — the arena fills it in server-side from your `ARENA_KEY`. As with Step 7, the endpoint may 404 on staging; the script reports rather than crashes.

# In[61]:


LEDGER_SESSION_ID = f"prematch:{SPORTMONKS_FIXTURE_ID}:{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"

# Bind the session_id to this fixture server-side. The records batch we POST
# below carries session_id on every record, but the arena also needs an
# explicit (session_id -> fixture_id) link to score the predictions later.
# Idempotent: re-binding the same session to the same fixture is a no-op.
bind_r = requests.post(
    f"{ARENA}/api/v1/arena/ledger/sessions/{LEDGER_SESSION_ID}/fixture",
    headers={**H_ARENA, "content-type": "application/json"},
    json={"fixture_id": str(SPORTMONKS_FIXTURE_ID)},
    timeout=10,
)
if bind_r.ok:
    print(f"HTTP {bind_r.status_code} (OK) -- session bound to fixture: "
          f"{bind_r.json()}")
else:
    print(f"HTTP {bind_r.status_code} -- session->fixture bind failed: "
          f"{bind_r.text[:300]}")


def _new_record(behavior, **fields):
    """Compose the BaseRecord envelope + behavior-specific fields.

    Note: agent_id is intentionally omitted. The arena resolves it server-side
    from the x-api-key on POST, so wire records do not carry it. The local
    dump produced by this script mirrors that — schema-wise, agent_id is
    required, but it only becomes present after the server enriches the
    record."""
    rec = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "session_id":     LEDGER_SESSION_ID,
        "record_id":      str(uuid.uuid4()),
        "behavior":       behavior,
        "client_ts_utc":  int(time.time() * 1000),
    }
    rec.update({k: v for k, v in fields.items() if v is not None})
    return rec

# `_mi(resp)` was removed in the refactor — provider routing + extraction now
# lives on LLMResult, so use `llm_X.to_model_invocation()` instead. That keeps
# the schema's `internal_reasoning` field populated correctly for every
# provider (see Setup cell, LLMResult).

def _jstr(obj):
    """JSON-stringify for ledger fields the schema types as `string`
    (Thinking.inputs[].input_payload, Thinking.output_payload). No truncation —
    SIZE_LIMITS in SCHEMA.md are advisory; the server will tell us if a record
    exceeds them and we'd rather lose the record than silently lose content."""
    return obj if isinstance(obj, str) else json.dumps(obj, default=str)


# (1) Observing — synthetic cron trigger that woke the agent.
rec_trigger = _new_record(
    "Observing",
    trigger_source="dev-guide-workflow-test",
    trigger_type="cron_trigger",
    trigger_description=f"Pre-match prediction run for fixture {SPORTMONKS_FIXTURE_ID} ({fixture['name']})",
    trigger_payload_summary=(
        f"fixture_id={SPORTMONKS_FIXTURE_ID}; window=PRE_MATCH; "
        f"kickoff_utc={fixture['starting_at']}; home={home['short_code']}; away={away['short_code']}"
    ),
)

# (2) ToolCalling — Sportmonks schedule
rec_sm_schedule = _new_record(
    "ToolCalling",
    upstream_record_id=[rec_trigger["record_id"]],
    tool_meta={"name": "sportmonks", "endpoint": "/v3/football/schedules/seasons/{season_id}",
               "via": "arena.sportmonks_proxy"},
    description="List WC2026 season schedule to discover fixtures",
    input_payload={"season_id": 26618},
    output_payload={"stage_count": len(schedule), "picked_fixture_id": SPORTMONKS_FIXTURE_ID},
    success=True,
)

# (3) ToolCalling — Sportmonks fixture detail
rec_sm_fixture = _new_record(
    "ToolCalling",
    upstream_record_id=[rec_sm_schedule["record_id"]],
    tool_meta={"name": "sportmonks", "endpoint": "/v3/football/fixtures/{fixture_id}",
               "via": "arena.sportmonks_proxy"},
    description="Fetch fixture detail with pre-match prediction includes",
    input_payload={"fixture_id": SPORTMONKS_FIXTURE_ID,
                   "include":    "participants;predictions;odds;xGFixture"},
    output_payload={
        "fixture_name":      fixture["name"],
        "kickoff_utc":       fixture["starting_at"],
        "participants":      [{"id": p["id"], "name": p["name"],
                               "short_code": p["short_code"],
                               "country_id": p["country_id"],
                               "location": p["meta"]["location"]} for p in fixture["participants"]],
        "predictions_count": len(fixture.get("predictions") or []),
        "odds_count":        len(fixture.get("odds") or []),
        "xgfixture_count":   len(fixture.get("xgfixture") or []),
    },
    success=True,
)

# (4) Thinking — Sportmonks digest. inputs[].input_payload mirrors what the LLM
# actually saw in Step 2 (top_prediction + top_odds), not the raw upstream lists,
# so the ledger trace is forensically reproducible.
rec_th_sportmonks = _new_record(
    "Thinking",
    upstream_record_id=[rec_sm_fixture["record_id"]],
    model_invocation=llm_digest.to_model_invocation(),
    prompt=DIGEST_SYS,
    inputs=[{
        "input_record_id": rec_sm_fixture["record_id"],
        "input_payload":   _jstr({
            "fixture":    fixture["name"],
            "home_code":  home["short_code"],
            "away_code":  away["short_code"],
            "prediction": top_prediction,
            "odds":       top_odds,
            "xGFixture":  fixture.get("xgfixture"),
        }),
    }],
    output_payload=_jstr(sportmonks_digest),
)

# (5a) ToolCalling — arena: look up the polymarket event slug for the fixture.
rec_pm_slug = _new_record(
    "ToolCalling",
    upstream_record_id=[rec_sm_schedule["record_id"]],
    tool_meta={"name": "arena-mapping",
               "endpoint": "/api/v1/web/mapping"},
    description="Look up curated Polymarket event_slug for this Sportmonks fixture",
    input_payload={"fixture_id": SPORTMONKS_FIXTURE_ID},
    output_payload={"polymarket_event_slug": polymarket_event_slug},
    success=polymarket_event_slug is not None,
)

# (5b) ToolCalling — Polymarket Gamma: fetch the event + nested markets
# (condition_ids + clobTokenIds for home / draw / away).
rec_pm_event = _new_record(
    "ToolCalling",
    upstream_record_id=[rec_pm_slug["record_id"]],
    tool_meta={"name": "polymarket-gamma",
               "endpoint": "/api/v1/data/proxy/polymarket-gamma/events",
               "via": "arena.proxy"},
    description="Fetch Polymarket event + 3 child winner markets by slug",
    input_payload={"slug": polymarket_event_slug},
    output_payload={
        "outcomes": {k: {"team_code":     moneyline["outcomes"][k]["team_code"],
                         "condition_id":  moneyline["outcomes"][k]["condition_id"],
                         "token_yes":     moneyline["outcomes"][k]["token_yes"]}
                     for k in moneyline["outcomes"]}
    } if moneyline else None,
    success=moneyline is not None,
)

# (5c) ToolCalling — Polymarket CLOB: live midpoint per YES token (3 calls
# summarized into one record).
rec_pm_mids = _new_record(
    "ToolCalling",
    upstream_record_id=[rec_pm_event["record_id"]],
    tool_meta={"name": "polymarket-clob",
               "endpoint": "/api/v1/data/proxy/polymarket-clob/midpoint",
               "via": "arena.proxy"},
    description="Fetch CLOB midpoint per outcome YES token (home / draw / away)",
    input_payload={"token_ids": [
        moneyline["outcomes"][k]["token_yes"] for k in moneyline["outcomes"]
    ] if moneyline else None},
    output_payload={
        k: moneyline["outcomes"][k]["current_mid_yes"] for k in moneyline["outcomes"]
    } if moneyline else None,
    success=moneyline is not None,
)

# (6) Thinking — Polymarket digest
rec_th_polymarket = _new_record(
    "Thinking",
    upstream_record_id=[rec_pm_slug["record_id"],
                        rec_pm_event["record_id"],
                        rec_pm_mids["record_id"]],
    model_invocation=llm_pm.to_model_invocation(),
    prompt=POLYMARKET_DIGEST_SYS,
    inputs=[{
        "input_record_id": rec_pm_mids["record_id"],
        "input_payload":   _jstr(moneyline),
    }],
    output_payload=_jstr(polymarket_digest),
)

# (7) ToolCalling — Supabase catalog discovery
rec_sb_catalog = _new_record(
    "ToolCalling",
    upstream_record_id=[rec_trigger["record_id"]],
    tool_meta={"name": "supabase", "endpoint": "/rest/v1/catalog_full"},
    description="Discover available Supabase tables via the public catalog",
    input_payload={"params": {"select": "table_name,category,row_count,table_description,columns",
                              "order":  "category,table_name"}},
    output_payload={"available_tables": [t["table_name"] for t in catalog],
                    "count": len(catalog)},
    success=True,
)

# (8) ToolCalling — Supabase priors fetch
rec_sb_priors = _new_record(
    "ToolCalling",
    upstream_record_id=[rec_sb_catalog["record_id"], rec_sm_fixture["record_id"]],
    tool_meta={"name": "supabase", "endpoint": f"/rest/v1/{WANTED_TABLE}",
               "schema": "world_cup_arena"},
    description=f"Fetch {WANTED_TABLE} priors for both teams",
    input_payload={"country_id": f"in.({COUNTRY_A_ID},{COUNTRY_B_ID})", "select": "*"},
    output_payload=priors_rows,
    success=True,
)

# (9) Thinking — Supabase digest
rec_th_supabase = _new_record(
    "Thinking",
    upstream_record_id=[rec_sb_priors["record_id"]],
    model_invocation=llm_sb.to_model_invocation(),
    prompt=SUPABASE_DIGEST_SYS,
    inputs=[{
        "input_record_id": rec_sb_priors["record_id"],
        "input_payload":   _jstr({
            "fixture":      fixture["name"],
            "source_table": WANTED_TABLE,
            "home_code":    home["short_code"],
            "away_code":    away["short_code"],
            "rows":         priors_rows,
        }),
    }],
    output_payload=_jstr(supabase_digest),
)

# (10) Thinking — Predict (priors only, blind to market).
# The reasoning lives here; the structured prediction is committed via the
# Acting record below, which is the form the arena validates + scores.
rec_th_predict = _new_record(
    "Thinking",
    upstream_record_id=[rec_th_sportmonks["record_id"], rec_th_supabase["record_id"]],
    model_invocation=llm_predict.to_model_invocation(),
    prompt=PREDICT_SYS,
    inputs=[
        {"input_record_id": rec_th_sportmonks["record_id"],
         "input_payload":   _jstr(sportmonks_digest)},
        {"input_record_id": rec_th_supabase["record_id"],
         "input_payload":   _jstr(supabase_digest)},
    ],
    output_payload=_jstr(prediction),
)

# (11) Acting — Prediction (validated + scored by the arena).
# Per the new ledger contract, predictions are emitted as Acting records with
# action_type="prediction" and structured `parameters` the arena snapshots
# for scoring at settlement. probability is clamped to the schema range
# [0.001, 0.999].
_pred_prob = max(0.001, min(0.999, float(prediction["probability"])))
rec_act_predict = _new_record(
    "Acting",
    upstream_record_id=[rec_th_predict["record_id"]],
    action_type=     "prediction",
    target_system=   "arena",
    action_summary=  f"Predict {prediction['outcome']} @ p={_pred_prob:.2f} for fixture {SPORTMONKS_FIXTURE_ID}",
    parameters=      {
        "fixture_id": str(SPORTMONKS_FIXTURE_ID),
        "outcome":      prediction["outcome"],
        "probability":  _pred_prob,
    },
    dry_run=         False,
    execution_status="confirmed",
)

# (12) Thinking — Strategy (prediction + market → trade decision)
rec_th_strategy = _new_record(
    "Thinking",
    upstream_record_id=[rec_th_predict["record_id"], rec_th_polymarket["record_id"]],
    model_invocation=llm_strategy.to_model_invocation(),
    prompt=STRATEGY_SYS,
    inputs=[
        {"input_record_id": rec_th_predict["record_id"],
         "input_payload":   _jstr(prediction)},
        {"input_record_id": rec_th_polymarket["record_id"],
         "input_payload":   _jstr(polymarket_digest)},
    ],
    output_payload=_jstr(strategy),
)

records = [
    rec_trigger, rec_sm_schedule,
    rec_pm_slug, rec_pm_event, rec_pm_mids,
    rec_sm_fixture, rec_th_sportmonks,
    rec_th_polymarket,
    rec_sb_catalog, rec_sb_priors, rec_th_supabase,
    rec_th_predict, rec_act_predict, rec_th_strategy,
]

# (13) Acting — ONE record per order Step 7a actually attempted to submit.
# Step 6b can emit 0, 1, or 2 orders (a "fade" emits two synthetic buy-YES
# orders), so this is a loop over `order_payloads` / `order_responses` /
# `order_outcomes`. These are the AGENT-side Acting records (intent /
# submission); the arena will additionally write its own Acting record(s)
# server-side at fill / close time with target_system="public-chain" +
# execution_id=<tx_hash> for the on-chain settlement evidence.
for op, resp, outcome in zip(order_payloads, order_responses, order_outcomes):
    # Did the order POST land cleanly? Required precondition for anything but
    # "failed" — resp is None on 404 / exception.
    submitted_ok = isinstance(resp, dict) and bool(resp)
    fs = outcome["final_status"]
    tx = outcome["tx_hash"]
    # Map this order's polled outcome -> Acting.execution_status enum
    # (confirmed | failed | simulated | pending):
    #   filled                       -> confirmed
    #   closed AND tx_hash           -> confirmed   (settled on-chain)
    #   closed AND no tx_hash        -> failed      (cancelled / expired)
    #   rejected                     -> failed
    #   non-terminal after polling   -> pending     (server-side fill-time Acting will supersede)
    #   POST never landed            -> failed
    if   fs == "filled" or (fs == "closed" and tx):
        exec_status = "confirmed"
    elif fs in ("closed", "rejected"):
        exec_status = "failed"
    elif submitted_ok:
        exec_status = "pending"
    else:
        exec_status = "failed"
    rec_act = _new_record(
        "Acting",
        upstream_record_id=[rec_th_strategy["record_id"]],
        action_type=     "open_order",
        target_system=   "arena",     # we submit to arena; arena routes to polymarket-clob
        action_summary=  (f"Open ${float(op['usd_size']):.2f} YES on "
                          f"{op['team_code']} @ ≤{op['limit_price']}"),
        parameters=      op,
        dry_run=         False,
        execution_status=exec_status,
        execution_id=    (resp.get("order_id") if submitted_ok else None),
    )
    records.append(rec_act)

print(f"Built {len(records)} ledger records -- one per step the agent took:\n")
for rec in records:
    label = (rec.get("description")
             or rec.get("action_summary")
             or rec.get("trigger_description")
             or rec.get("prompt", "")[:50])
    print(f"  {rec['behavior']:12s} {rec['record_id'][:8]}...  {label}")

# Persist the batch payload locally before POSTing, so it can be inspected,
# diffed against the server's per-record response, or replayed verbatim if
# the POST fails or returns ambiguous errors.
import pathlib
PAYLOAD_PATH = pathlib.Path("output/ledger_batch_payload.json")
PAYLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
PAYLOAD_PATH.write_text(json.dumps({"records": records}, indent=2, default=str))
print(f"\nPayload saved to {PAYLOAD_PATH} "
      f"({len(records)} records, {PAYLOAD_PATH.stat().st_size} bytes)")

# Pre-validate via /records/validate before the batch POST. Cheap server-side
# schema check; surfaces obvious structural issues (missing required fields,
# bad enums, etc.) without the side effect of creating real records. Non-
# blocking: if validate raises any issue we print and still submit, since the
# batch endpoint is the authoritative gate.
print(f"\nValidating {len(records)} records before submission...")
try:
    v = requests.post(
        f"{ARENA}/api/v1/arena/ledger/records/validate",
        headers=H_ARENA, timeout=30,
        json={"records": records},
    )
    if v.ok:
        vbody = v.json()
        if vbody.get("valid") and not vbody.get("errors"):
            print(f"  HTTP {v.status_code} -- valid: {len(records)}/{len(records)} records OK.")
        else:
            print(f"  HTTP {v.status_code} -- valid={vbody.get('valid')}, "
                  f"errors={len(vbody.get('errors') or [])}")
            for e in vbody.get("errors") or []:
                print(f"    {e}")
    else:
        print(f"  HTTP {v.status_code} -- validate rejected. Body: {v.text[:500]}")
except Exception as e:
    print(f"  Validate POST failed (non-blocking): {type(e).__name__}: {e}")

# Submit the trace as a single batch. Per the new ledger contract:
#   - No session-create endpoint; session_id is purely a client-side string.
#   - Bare record dicts (no {"body": {...}} envelope).
#   - agent_id is derived server-side from x-api-key.
#   - One round-trip per cycle via /records/batch (≤50 records). Response:
#       {"records": [<enriched echoes>], "errors": [{index, code, message}, ...]}
# Endpoint isn't live on staging yet — expect 404. Script reports rather than raises.
try:
    r = requests.post(
        f"{ARENA}/api/v1/arena/ledger/records/batch",
        headers=H_ARENA, timeout=60,
        json={"records": records},
    )
    if r.status_code == 404:
        print(f"\nHTTP 404 -- the ledger endpoint isn't live on staging yet (expected). "
              f"The {len(records)} records above are exactly what a real run would submit.")
    elif r.ok:
        resp = r.json()
        print(f"\nHTTP {r.status_code} (OK) -- ledger accepted: "
              f"{len(resp.get('records', []))} stored, {len(resp.get('errors', []))} error(s).")
        for e in resp.get("errors", []):
            print(f"    [#{e.get('index')}] {e.get('code')}: {e.get('message')}")
    else:
        print(f"\nHTTP {r.status_code} -- ledger rejected. Body: {r.text[:300]}")
except Exception as e:
    print(f"\nLedger POST failed: {type(e).__name__}: {e}")


# In[ ]:




