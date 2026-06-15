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
_deepseek_client = None
_grok_client = None
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
        _anthropic_client = anthropic.Anthropic(
            api_key=config.ANTHROPIC_KEY,
            timeout=config.PRIMARY_LLM_TIMEOUT_SECONDS,
        )
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None and config.OPENAI_KEY:
        from openai import OpenAI
        _openai_client = OpenAI(
            api_key=config.OPENAI_KEY,
            timeout=config.PRIMARY_LLM_TIMEOUT_SECONDS,
        )
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
            "timeout": config.PRIMARY_LLM_TIMEOUT_SECONDS,
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


def _get_deepseek():
    global _deepseek_client
    if _deepseek_client is None and config.DEEPSEEK_KEY:
        from openai import OpenAI
        _deepseek_client = OpenAI(
            api_key=config.DEEPSEEK_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
    return _deepseek_client


def _get_grok():
    global _grok_client
    if _grok_client is None and config.XAI_KEY:
        from openai import OpenAI
        _grok_client = OpenAI(
            api_key=config.XAI_KEY,
            base_url=config.XAI_BASE_URL,
        )
    return _grok_client


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


def _chat_message_text_and_thinking(message) -> tuple[str, str]:
    """
    OpenAI-compatible providers differ on where reasoning models put their
    final answer and private reasoning. DeepSeek exposes `reasoning_content`;
    other providers may expose only `content`.
    """
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text_parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                text_parts.append(str(getattr(part, "text", "") or ""))
        content = "\n".join(p for p in text_parts if p)
    thinking = getattr(message, "reasoning_content", "") or ""
    return str(content or ""), str(thinking or "")


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
                timeout=config.PRIMARY_LLM_TIMEOUT_SECONDS,
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
        timeout=config.PRIMARY_LLM_TIMEOUT_SECONDS,
    )
    text, thinking = _chat_message_text_and_thinking(resp.choices[0].message)
    return LLMResult(
        parsed=_parse_json(text),
        raw_text=text,
        thinking=thinking,
        model=model,
        provider=provider,
        tokens_in=getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
        tokens_out=getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
    )


def call_deepseek(
    system: str,
    user_content: str,
    model: str | None = None,
) -> LLMResult:
    """
    Call DeepSeek-R1 (deepseek-reasoner) for the devil's-advocate role.

    DeepSeek exposes the *raw* chain-of-thought in `reasoning_content` — the
    richest possible payload for the ledger's internal_reasoning field. If
    DeepSeek is unavailable we fall back to Claude so the council still runs.
    """
    client = _get_deepseek()
    if client:
        try:
            m = model or config.DEVIL_MODEL
            resp = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                timeout=config.DEVIL_TIMEOUT_SECONDS,
            )
            msg = resp.choices[0].message
            text, thinking = _chat_message_text_and_thinking(msg)
            usage = getattr(resp, "usage", None)
            return LLMResult(
                parsed=_parse_json(text),
                raw_text=text,
                thinking=thinking,
                model=m,
                provider="deepseek",
                tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
                tokens_out=getattr(usage, "completion_tokens", 0) or 0,
            )
        except Exception as e:
            print(f"  [DeepSeek error: {e}] — falling back to Claude for devil's advocate")
    return call_claude(system, user_content, thinking_budget=2048)


def call_grok(
    system: str,
    user_content: str,
    model: str | None = None,
) -> LLMResult:
    """
    Call Grok (xAI). Grok is trained with live access to X/Twitter, so it is the
    best model for a real-time social-pulse read (breaking team news, fan mood,
    injury chatter). OpenAI-SDK compatible. Returns an empty-provider result if
    no XAI key is configured so callers degrade gracefully.
    """
    client = _get_grok()
    if not client:
        return LLMResult(parsed={}, raw_text="", thinking="",
                         model="", provider="unavailable")
    try:
        m = model or config.GROK_MODEL
        resp = client.chat.completions.create(
            model=m,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            timeout=config.GROK_TIMEOUT_SECONDS,
        )
        msg = resp.choices[0].message
        text, thinking = _chat_message_text_and_thinking(msg)
        usage = getattr(resp, "usage", None)
        return LLMResult(
            parsed=_parse_json(text),
            raw_text=text,
            thinking=thinking,
            model=m,
            provider="xai",
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
        )
    except Exception as e:
        print(f"  [Grok error: {e}]")
        return LLMResult(parsed={}, raw_text="", thinking="",
                         model=model or config.GROK_MODEL, provider="error")


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
