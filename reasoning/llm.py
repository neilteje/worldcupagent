"""
LLM pipeline — primary: Claude with extended thinking; fallback: OpenAI; ensemble: Gemini.

Priority:
  1. Anthropic Claude (extended thinking, full chain — best for reasoning score)
  2. OpenAI GPT (if no Anthropic key)
  3. Gemini (ensemble on predict step only)

Set ANTHROPIC_API_KEY for best results. OpenAI is used when Claude is unavailable.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any
import config

# ── Clients ────────────────────────────────────────────────────────────────

_anthropic_client = None
_openai_client = None
_openrouter_client = None
_gemini_client = None
_primary: str = (
    "openrouter" if config.OPENROUTER_KEY else
    "anthropic" if config.ANTHROPIC_KEY else
    "openai" if config.OPENAI_KEY else
    "gemini"
)


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None and config.ANTHROPIC_KEY:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None and config.OPENAI_KEY:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=config.OPENAI_KEY)
    return _openai_client


def _get_openrouter():
    global _openrouter_client
    if _openrouter_client is None and config.OPENROUTER_KEY:
        from openai import OpenAI
        headers = {}
        if config.OPENROUTER_SITE_URL:
            headers["HTTP-Referer"] = config.OPENROUTER_SITE_URL
        if config.OPENROUTER_APP_NAME:
            headers["X-OpenRouter-Title"] = config.OPENROUTER_APP_NAME
        kwargs = {
            "api_key": config.OPENROUTER_KEY,
            "base_url": config.OPENROUTER_BASE_URL,
        }
        if headers:
            kwargs["default_headers"] = headers
        _openrouter_client = OpenAI(**kwargs)
    return _openrouter_client


def _get_gemini():
    global _gemini_client
    if _gemini_client is None and config.GEMINI_KEY:
        try:
            from google import genai
            _gemini_client = genai.Client(api_key=config.GEMINI_KEY)
        except ImportError:
            pass
    return _gemini_client


# ── Extraction helpers ─────────────────────────────────────────────────────

def _extract_text_and_thinking(resp) -> tuple[str, str]:
    """
    Extract (final_text, thinking_chain) from any provider response.
    Mirrors the notebook's _extract() function.
    """
    # Anthropic: list of typed blocks
    if hasattr(resp, "content") and isinstance(resp.content, list):
        text_parts, thinking_parts = [], []
        for block in resp.content:
            if block.type == "thinking":
                thinking_parts.append(block.thinking)
            elif block.type == "text":
                text_parts.append(block.text)
        return "".join(text_parts), "\n\n".join(thinking_parts)
    # Gemini
    if hasattr(resp, "candidates"):
        text_parts, thinking_parts = [], []
        for part in resp.candidates[0].content.parts:
            if not getattr(part, "text", None):
                continue
            if getattr(part, "thought", False):
                thinking_parts.append(part.text)
            else:
                text_parts.append(part.text)
        return "\n".join(text_parts), "\n\n".join(thinking_parts)
    return "", ""


def _parse_json(text: str) -> dict:
    """Extract the first {...} JSON object from LLM text output."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {}


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class LLMResult:
    """Generic result for any LLM step."""
    parsed: dict                    # the extracted JSON
    raw_text: str                   # full LLM output text
    thinking: str                   # internal reasoning chain (for ledger)
    model: str
    provider: str
    tokens_in: int = 0
    tokens_out: int = 0
    gemini_parsed: dict = field(default_factory=dict)   # ensemble fallback


# ── Core call (Claude primary, OpenAI fallback) ────────────────────────────

def call_claude(
    system: str,
    user_content: str,
    model: str | None = None,
    thinking_budget: int | None = None,
) -> LLMResult:
    """
    Call the configured primary LLM. OpenRouter is preferred when configured,
    then Claude, then OpenAI.
    """
    openrouter = _get_openrouter()
    if openrouter:
        try:
            return _call_openai_compatible_result(
                openrouter,
                system,
                user_content,
                model or config.OPENROUTER_MODEL,
                "openrouter",
            )
        except Exception as e:
            print(f"  [OpenRouter error: {e}] - falling back to Claude/OpenAI")

    client = _get_anthropic()
    if client:
        try:
            m = model or config.ANTHROPIC_MODEL
            budget = thinking_budget or config.THINKING_BUDGET
            resp = client.messages.create(
                model=m,
                max_tokens=budget + 4096,
                thinking={"type": "enabled", "budget_tokens": budget},
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            text, thinking = _extract_text_and_thinking(resp)
            return LLMResult(
                parsed=_parse_json(text),
                raw_text=text,
                thinking=thinking,
                model=m,
                provider="anthropic",
                tokens_in=resp.usage.input_tokens,
                tokens_out=resp.usage.output_tokens,
            )
        except Exception as e:
            print(f"  [Claude error: {e}] — falling back to OpenAI")

    # OpenAI fallback
    oa = _get_openai()
    if oa:
        return _call_openai_result(oa, system, user_content)

    raise RuntimeError(
        "No LLM provider available. Set OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
    )


def _call_openai_result(client, system: str, user_content: str) -> LLMResult:
    """Call OpenAI with reasoning effort and return an LLMResult."""
    return _call_openai_compatible_result(client, system, user_content, "gpt-4o", "openai")


def _call_openai_compatible_result(
    client,
    system: str,
    user_content: str,
    model: str,
    provider: str,
) -> LLMResult:
    """Call an OpenAI-compatible chat completions endpoint."""
    resp = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
    )
    text = resp.choices[0].message.content or ""
    return LLMResult(
        parsed=_parse_json(text),
        raw_text=text,
        thinking="",   # OpenAI-compatible chat APIs usually do not expose reasoning.
        model=model,
        provider=provider,
        tokens_in=getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
        tokens_out=getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
    )


def _call_gemini(system: str, user_content: str) -> dict | None:
    """
    Call Gemini 2.5 Pro for ensemble calibration on the predict step.
    Returns the parsed JSON dict, or None on failure.
    """
    client = _get_gemini()
    if client is None:
        return None
    try:
        from google.genai import types
        resp = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=2048,
                ),
                max_output_tokens=4096,
            ),
        )
        text, _ = _extract_text_and_thinking(resp)
        return _parse_json(text)
    except Exception:
        return None


# ── Pipeline functions ─────────────────────────────────────────────────────

def digest_sportmonks(user_content: str) -> LLMResult:
    """Step 1: Compress Sportmonks payload → clean digest JSON."""
    from reasoning.prompts import SPORTMONKS_DIGEST_SYS
    return call_claude(SPORTMONKS_DIGEST_SYS, user_content,
                       thinking_budget=1024)


def digest_polymarket(user_content: str) -> LLMResult:
    """Step 2: Compress Polymarket moneyline → digest with execution handles."""
    from reasoning.prompts import POLYMARKET_DIGEST_SYS
    return call_claude(POLYMARKET_DIGEST_SYS, user_content,
                       thinking_budget=1024)


def digest_supabase(user_content: str) -> LLMResult:
    """Step 3: Compress multi-table priors → per-team profile."""
    from reasoning.prompts import SUPABASE_DIGEST_SYS
    return call_claude(SUPABASE_DIGEST_SYS, user_content,
                       thinking_budget=2048)


def predict(user_content: str) -> LLMResult:
    """
    Step 4: Form independent prediction from digests.
    Ensembles Claude + Gemini 2.5 Pro on the probability.
    This is the PSL-scored output.
    """
    from reasoning.prompts import PREDICT_SYS
    result = call_claude(PREDICT_SYS, user_content,
                         thinking_budget=config.THINKING_BUDGET)

    # Ensemble: average Claude's probability with Gemini's
    gemini_parsed = _call_gemini(PREDICT_SYS, user_content)
    result.gemini_parsed = gemini_parsed or {}

    if gemini_parsed:
        c_prob = result.parsed.get("probability", 0)
        g_prob = gemini_parsed.get("probability", c_prob)
        # Only ensemble if same outcome prediction
        if gemini_parsed.get("outcome") == result.parsed.get("outcome"):
            blended = round(0.70 * c_prob + 0.30 * g_prob, 4)
            result.parsed["probability"] = blended
            result.parsed["ensemble_note"] = (
                f"Blended: Claude={c_prob:.3f} Gemini={g_prob:.3f} → {blended:.3f}"
            )

    return result


def strategy(user_content: str) -> LLMResult:
    """Step 5: Compare prediction vs market → trade decision."""
    from reasoning.prompts import STRATEGY_SYS
    return call_claude(STRATEGY_SYS, user_content,
                       thinking_budget=2048)


def ht_predict(user_content: str) -> LLMResult:
    """HT: Bayesian update on live match state."""
    from reasoning.prompts import HT_PREDICT_SYS
    return call_claude(HT_PREDICT_SYS, user_content,
                       thinking_budget=config.THINKING_BUDGET)
