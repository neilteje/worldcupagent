from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import time
from typing import Any

import httpx


ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODELS = (
    os.getenv("ANTHROPIC_MODEL")
    or os.getenv("CLAUDE_MODEL")
    or "claude-haiku-4-5-20251001",
    "claude-3-5-haiku-latest",
    "claude-3-haiku-20240307",
)
SONNET_MODELS = (
    os.getenv("ANTHROPIC_ANALYST_MODEL")
    or os.getenv("ANTHROPIC_SONNET_MODEL")
    or (os.getenv("ANTHROPIC_MODEL") if "sonnet" in (os.getenv("ANTHROPIC_MODEL") or "").lower() else "")
    or (os.getenv("CLAUDE_MODEL") if "sonnet" in (os.getenv("CLAUDE_MODEL") or "").lower() else "")
    or "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-latest",
)


def anthropic_key_status(settings) -> dict:
    key = getattr(settings, "anthropic_key", "") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY") or ""
    return {"present": bool(key), "length": len(key)}


def anthropic_health_check(settings, *, timeout_seconds: float = 30.0) -> dict:
    """Make a real low-token Anthropic API call without exposing the key."""
    key = getattr(settings, "anthropic_key", "") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY") or ""
    if not key:
        return {"called": False, "ok": False, "provider": "anthropic", "reason": "anthropic_key_missing"}

    prompt = (
        "Return only JSON: {\"ok\": true, \"role\": \"health_check\", "
        "\"note\": \"anthropic_api_reachable\"}."
    )
    started = time.perf_counter()
    errors: list[str] = []
    for model in _model_candidates():
        try:
            payload = _messages_payload(model, prompt, max_tokens=80)
            response = httpx.post(
                ANTHROPIC_MESSAGES_URL,
                headers=_headers(key),
                json=payload,
                timeout=timeout_seconds,
            )
            if not response.is_success:
                errors.append(f"{model}: HTTP {response.status_code} {response.text[:180]}")
                continue
            body = response.json()
            text = _extract_text(body)
            return {
                "called": True,
                "ok": True,
                "provider": "anthropic",
                "model": body.get("model") or model,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "usage": body.get("usage") or {},
                "parsed": _parse_json(text),
                "text_preview": text[:240],
            }
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {exc}")
    return {
        "called": True,
        "ok": False,
        "provider": "anthropic",
        "latency_seconds": round(time.perf_counter() - started, 3),
        "errors": errors[-3:],
    }


def critique_decisions_with_anthropic(settings, decisions: list[dict], *, storage_dir: Path | None = None) -> dict:
    """Ask Claude for critique only; deterministic checks still decide orders."""
    key = getattr(settings, "anthropic_key", "") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY") or ""
    if not key:
        return {"called": False, "ok": False, "provider": "anthropic", "reason": "anthropic_key_missing"}
    compact = [_compact_decision(d) for d in decisions[:10]]
    prompt = (
        "You are a strict reviewer for a dry-run soccer prediction-market agent. "
        "You may critique probabilities, missing data, reporting, and risk flags. "
        "You must not authorize orders and must not suggest bypassing deterministic gates. "
        "Return only minified JSON, no markdown fences. Use at most 3 short strings per array. "
        "Keys: probability_concerns, risk_flag_suggestions, reporting_improvements, "
        "next_engineering_steps, order_authorization.\n\n"
        f"Decisions JSON:\n{json.dumps(compact, sort_keys=True)[:9000]}"
    )
    started = time.perf_counter()
    errors: list[str] = []
    for model in _model_candidates():
        try:
            response = _post_with_optional_thinking(key, model, prompt, max_tokens=2200, thinking_budget=1024, timeout=60)
            if not response.is_success:
                errors.append(f"{model}: HTTP {response.status_code} {response.text[:180]}")
                continue
            body = response.json()
            text = _extract_text(body)
            thinking = _extract_thinking(body)
            result = {
                "called": True,
                "ok": True,
                "provider": "anthropic",
                "model": body.get("model") or model,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "usage": body.get("usage") or {},
                "parsed": _parse_json(text),
                "internal_reasoning": thinking[:4000],
                "text_preview": text[:2000],
                "order_authorization_allowed": False,
            }
            if storage_dir is not None:
                result["artifact"] = str(_write_artifact(storage_dir, result))
            return result
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {exc}")
    return {
        "called": True,
        "ok": False,
        "provider": "anthropic",
        "latency_seconds": round(time.perf_counter() - started, 3),
        "errors": errors[-3:],
    }


def analyze_decision_signals_with_anthropic(settings, decision_context: dict, *, storage_dir: Path | None = None) -> dict:
    """Ask a Sonnet model to analyze one decision's signals for combined gating."""
    key = getattr(settings, "anthropic_key", "") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY") or ""
    if not key:
        return {"called": False, "ok": False, "provider": "anthropic", "role": "signal_analyst", "reason": "anthropic_key_missing"}
    compact = _compact_signal_context(decision_context)
    prompt = (
        "You are the LLM analyst for a soccer prediction-market agent. "
        "Analyze the deterministic signals, source agreement, edge, confidence, uncertainty, and risk flags. "
        "Your output is used as a required second decision input, but deterministic gates remain authoritative. "
        "Do not authorize real orders, do not suggest bypassing risk gates, and be conservative when data is missing. "
        "Return only minified JSON, no markdown fences. "
        "Schema: {\"recommendation\":\"BET|SKIP|WATCH\",\"risk_posture\":\"approve|caution|veto\","
        "\"confidence_adjustment\": number between -0.10 and 0.10,"
        "\"supporting_signals\":[up to 3 short strings],\"contradicting_signals\":[up to 3 short strings],"
        "\"additional_risk_flags\":[up to 4 snake_case strings],\"rationale\":\"short string\","
        "\"order_authorization\":\"NO_AUTHORITY\"}.\n\n"
        f"Decision context JSON:\n{json.dumps(compact, sort_keys=True)[:7000]}"
    )
    started = time.perf_counter()
    errors: list[str] = []
    for model in _model_candidates(SONNET_MODELS):
        try:
            response = _post_with_optional_thinking(key, model, prompt, max_tokens=1800, thinking_budget=1024, timeout=60)
            if not response.is_success:
                errors.append(f"{model}: HTTP {response.status_code} {response.text[:180]}")
                continue
            body = response.json()
            text = _extract_text(body)
            thinking = _extract_thinking(body)
            result = {
                "called": True,
                "ok": True,
                "provider": "anthropic",
                "role": "signal_analyst",
                "model": body.get("model") or model,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "usage": body.get("usage") or {},
                "parsed": _parse_json(text),
                "internal_reasoning": thinking[:4000],
                "text_preview": text[:1600],
                "order_authorization_allowed": False,
            }
            if storage_dir is not None:
                result["artifact"] = str(_write_artifact(storage_dir, result, prefix="anthropic_signal_analysis"))
            return result
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {exc}")
    return {
        "called": True,
        "ok": False,
        "provider": "anthropic",
        "role": "signal_analyst",
        "latency_seconds": round(time.perf_counter() - started, 3),
        "errors": errors[-4:],
    }


def _headers(key: str) -> dict[str, str]:
    return {
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _messages_payload(model: str, prompt: str, *, max_tokens: int) -> dict:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }


def _post_with_optional_thinking(key: str, model: str, prompt: str, *, max_tokens: int, thinking_budget: int, timeout: float) -> httpx.Response:
    payload = _messages_payload(model, prompt, max_tokens=max_tokens)
    payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    response = httpx.post(
        ANTHROPIC_MESSAGES_URL,
        headers=_headers(key),
        json=payload,
        timeout=timeout,
    )
    if response.status_code == 400 and "thinking" in response.text.lower():
        payload.pop("thinking", None)
        response = httpx.post(
            ANTHROPIC_MESSAGES_URL,
            headers=_headers(key),
            json=payload,
            timeout=timeout,
        )
    return response


def _model_candidates(candidates: tuple[str, ...] = DEFAULT_MODELS) -> tuple[str, ...]:
    seen: set[str] = set()
    models: list[str] = []
    for model in candidates:
        if model and model not in seen:
            models.append(model)
            seen.add(model)
    return tuple(models)


def _extract_text(body: dict[str, Any]) -> str:
    parts = []
    for block in body.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "".join(parts)


def _extract_thinking(body: dict[str, Any]) -> str:
    parts = []
    for block in body.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "thinking":
            parts.append(str(block.get("thinking") or ""))
    return "\n\n".join(p for p in parts if p)


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {}


def _compact_decision(decision: dict) -> dict:
    return {
        "fixture_code": decision.get("fixture_code"),
        "window": decision.get("window"),
        "final_probs": decision.get("final_probs"),
        "market_probs": decision.get("market_probs"),
        "bookmaker_probs": decision.get("bookmaker_probs"),
        "sportmonks_probs": decision.get("sportmonks_probs"),
        "consensus_case": decision.get("consensus_case"),
        "best_outcome": decision.get("best_outcome"),
        "best_edge": decision.get("best_edge"),
        "edge_tier": decision.get("edge_tier"),
        "confidence": decision.get("confidence"),
        "uncertainty": decision.get("uncertainty"),
        "risk_flags": decision.get("risk_flags"),
        "blocking_risk_flags": decision.get("blocking_risk_flags"),
        "data_completeness": decision.get("data_completeness"),
        "market_stale": decision.get("market_stale"),
        "top_signals": decision.get("top_signals"),
        "action": decision.get("action"),
        "dry_run": decision.get("dry_run"),
    }


def _compact_signal_context(context: dict) -> dict:
    keys = (
        "fixture_code",
        "window",
        "final_probs",
        "market_probs",
        "bookmaker_probs",
        "sportmonks_probs",
        "source_reconciliation",
        "consensus_case",
        "edge",
        "confidence",
        "uncertainty",
        "top_signals",
        "signal_conflict",
        "data_completeness",
        "market_stale",
        "risk",
        "lineup",
        "halftime",
        "dry_run",
    )
    return {k: context.get(k) for k in keys if k in context}


def _write_artifact(storage_dir: Path, result: dict, *, prefix: str = "anthropic_review") -> Path:
    reviews = Path(storage_dir) / "reviews"
    reviews.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = reviews / f"{prefix}_{stamp}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path
