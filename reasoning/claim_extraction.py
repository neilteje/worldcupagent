from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import re
import time
from typing import Any

import httpx

from agent.config import Settings
from models.calibration import OUTCOMES
from models.signal_scoring import score_signal
from reasoning.anthropic_review import (
    ANTHROPIC_MESSAGES_URL,
    ANTHROPIC_VERSION,
    SONNET_MODELS,
    _extract_text,
    _model_candidates,
    _parse_json,
)


CLAIM_TYPES = {
    "injury",
    "suspension",
    "lineup",
    "weather",
    "tactical_change",
    "motivation",
    "market_stale",
    "contradiction",
}
WEAK_SOURCE_KINDS = {"web", "news", "reddit", "text", "rumor"}
DEFAULT_LLM_DELTA_CAP = 0.02


@dataclass(frozen=True)
class Claim:
    claim_type: str
    source_kind: str
    source_name: str
    subject: str
    team: str | None
    outcome: str | None
    probability_delta: dict[str, float]
    confidence: float
    freshness: float
    evidence: str
    timestamp_utc: str

    def to_dict(self) -> dict:
        return asdict(self)


def extract_claims_with_anthropic(settings: Settings, sources: list[dict], *, delta_cap: float = DEFAULT_LLM_DELTA_CAP) -> dict:
    key = getattr(settings, "anthropic_key", "") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY") or ""
    if not key:
        return {"called": False, "ok": False, "provider": "anthropic", "role": "claim_extractor", "reason": "anthropic_key_missing", "claims": []}

    source_payload = [
        {
            "source_kind": str(s.get("source_kind") or s.get("kind") or "text")[:32],
            "source_name": str(s.get("source_name") or s.get("name") or "unknown")[:80],
            "text": str(s.get("text") or "")[:4000],
        }
        for s in sources
        if str(s.get("text") or "").strip()
    ]
    prompt = (
        "Extract only typed soccer betting-relevant claims from the provided source text. "
        "Do not forecast match probabilities. Do not infer beyond the text. "
        "Allowed claim_type values: injury, suspension, lineup, weather, tactical_change, motivation, market_stale, contradiction. "
        "Return strict JSON matching the schema. Keep probability_delta small and evidence directly grounded in the text.\n\n"
        f"Sources JSON:\n{json.dumps(source_payload, sort_keys=True)}"
    )
    started = time.perf_counter()
    errors: list[str] = []
    for model in _model_candidates(SONNET_MODELS):
        try:
            response = httpx.post(
                ANTHROPIC_MESSAGES_URL,
                headers={"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"},
                json=_claim_payload(model, prompt),
                timeout=60,
            )
            if response.status_code == 400 and "output_config" in response.text:
                fallback = _claim_payload(model, prompt)
                fallback.pop("output_config", None)
                response = httpx.post(
                    ANTHROPIC_MESSAGES_URL,
                    headers={"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"},
                    json=fallback,
                    timeout=60,
                )
            if not response.is_success:
                errors.append(f"{model}: HTTP {response.status_code} {response.text[:180]}")
                continue
            body = response.json()
            parsed = _parse_json(_extract_text(body))
            validation = validate_claim_json(parsed, delta_cap=delta_cap)
            return {
                "called": True,
                "ok": validation["ok"],
                "provider": "anthropic",
                "role": "claim_extractor",
                "model": body.get("model") or model,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "usage": body.get("usage") or {},
                "claims": validation["claims"],
                "errors": validation["errors"],
            }
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {exc}")
    return {"called": True, "ok": False, "provider": "anthropic", "role": "claim_extractor", "latency_seconds": round(time.perf_counter() - started, 3), "errors": errors[-4:], "claims": []}


def validate_claim_json(payload: dict | None, *, delta_cap: float = DEFAULT_LLM_DELTA_CAP) -> dict:
    if not isinstance(payload, dict):
        return {"ok": False, "claims": [], "errors": ["payload_not_object"]}
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        return {"ok": False, "claims": [], "errors": ["claims_not_list"]}

    claims: list[dict] = []
    errors: list[str] = []
    now = datetime.now(timezone.utc).isoformat()
    for idx, item in enumerate(raw_claims):
        if not isinstance(item, dict):
            errors.append(f"claim_{idx}_not_object")
            continue
        claim_type = str(item.get("claim_type") or "").strip().lower()
        if claim_type not in CLAIM_TYPES:
            errors.append(f"claim_{idx}_invalid_type")
            continue
        source_kind = str(item.get("source_kind") or "text").strip().lower()
        source_name = str(item.get("source_name") or "unknown").strip()[:120]
        evidence = str(item.get("evidence") or item.get("quote") or item.get("source_text") or "")[:300]
        subject = str(item.get("subject") or item.get("claim") or item.get("description") or item.get("summary") or evidence or "").strip()[:160]
        if not subject:
            errors.append(f"claim_{idx}_missing_subject")
            continue
        confidence = _clamp01(item.get("confidence", 0.4))
        freshness = _clamp01(item.get("freshness", 0.4))
        delta = cap_probability_delta(item.get("probability_delta") or {}, delta_cap=delta_cap)
        claims.append(
            Claim(
                claim_type=claim_type,
                source_kind=source_kind,
                source_name=source_name,
                subject=subject,
                team=_optional_text(item.get("team"), 80),
                outcome=_optional_outcome(item.get("outcome")),
                probability_delta=delta,
                confidence=confidence,
                freshness=freshness,
                evidence=evidence,
                timestamp_utc=str(item.get("timestamp_utc") or now)[:80],
            ).to_dict()
        )
    deduped = dedupe_claims(claims)
    return {"ok": len(errors) == 0, "claims": deduped, "errors": errors}


def dedupe_claims(claims: list[dict]) -> list[dict]:
    by_key: dict[tuple, dict] = {}
    for claim in claims:
        key = (
            str(claim.get("claim_type", "")).lower(),
            str(claim.get("source_kind", "")).lower(),
            _fingerprint(str(claim.get("subject", ""))),
            str(claim.get("team") or "").lower(),
        )
        existing = by_key.get(key)
        if existing is None or float(claim.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
            by_key[key] = claim
    return list(by_key.values())


def apply_official_overrides(claims: list[dict], *, lineup_result: dict | None = None, official_claims: list[dict] | None = None) -> dict:
    official_claims = official_claims or []
    official_keys = {
        (str(c.get("claim_type", "")).lower(), _fingerprint(str(c.get("subject", ""))), str(c.get("team") or "").lower())
        for c in official_claims
        if str(c.get("source_kind", "")).lower() in {"official", "sportmonks", "lineup"}
    }
    lineup_complete_no_shock = bool(lineup_result) and not lineup_result.get("lineup_shock") and "lineup_unconfirmed" not in lineup_result.get("risk_flags", [])
    kept: list[dict] = []
    dropped: list[dict] = []
    for claim in claims:
        claim_type = str(claim.get("claim_type", "")).lower()
        source_kind = str(claim.get("source_kind", "")).lower()
        key = (claim_type, _fingerprint(str(claim.get("subject", ""))), str(claim.get("team") or "").lower())
        overridden = key in official_keys and source_kind in WEAK_SOURCE_KINDS
        if lineup_complete_no_shock and source_kind in WEAK_SOURCE_KINDS and claim_type in {"injury", "suspension", "lineup"}:
            overridden = True
        if overridden:
            dropped.append({**claim, "override_reason": "official_data_overrides_web_claim"})
        else:
            kept.append(claim)
    return {"claims": kept, "dropped": dropped}


def claims_to_signals(claims: list[dict], *, delta_cap: float = DEFAULT_LLM_DELTA_CAP) -> list[dict]:
    signals: list[dict] = []
    for claim in claims:
        source_kind = str(claim.get("source_kind") or "text").lower()
        source_quality = _source_quality(source_kind)
        delta = cap_probability_delta(claim.get("probability_delta") or {}, delta_cap=delta_cap if source_kind in WEAK_SOURCE_KINDS else 0.05)
        signals.append(
            score_signal(
                name=f"claim_{claim.get('claim_type')}",
                source=source_kind,
                category=str(claim.get("claim_type") or "claim"),
                probability_delta=delta,
                source_quality=source_quality,
                freshness=float(claim.get("freshness", 0.4) or 0.4),
                corroboration=float(claim.get("confidence", 0.4) or 0.4),
                reason=str(claim.get("evidence") or claim.get("subject") or ""),
            )
        )
    return signals


def cap_probability_delta(delta: dict[str, Any], *, delta_cap: float = DEFAULT_LLM_DELTA_CAP) -> dict[str, float]:
    vals = {k: float((delta or {}).get(k, 0.0) or 0.0) for k in OUTCOMES}
    vals = {k: (v if -1.0 < v < 1.0 else 0.0) for k, v in vals.items()}
    cap = max(0.0, min(0.15, float(delta_cap)))
    max_abs = max(abs(v) for v in vals.values()) if vals else 0.0
    if max_abs > cap and max_abs > 0:
        scale = cap / max_abs
        vals = {k: v * scale for k, v in vals.items()}
    return vals


def _claim_payload(model: str, prompt: str) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_type": {"type": "string"},
                        "source_kind": {"type": "string"},
                        "source_name": {"type": "string"},
                        "subject": {"type": "string"},
                        "team": {"type": ["string", "null"]},
                        "outcome": {"type": ["string", "null"]},
                        "probability_delta": {"type": "object"},
                        "confidence": {"type": "number"},
                        "freshness": {"type": "number"},
                        "evidence": {"type": "string"},
                        "timestamp_utc": {"type": "string"},
                    },
                    "required": ["claim_type", "source_kind", "source_name", "subject", "probability_delta", "confidence", "freshness", "evidence"],
                },
            }
        },
        "required": ["claims"],
    }
    return {
        "model": model,
        "max_tokens": 1200,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }


def _optional_text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] if text else None


def _optional_outcome(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text if text in OUTCOMES else None


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _fingerprint(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _source_quality(source_kind: str) -> float:
    return {
        "official": 0.95,
        "sportmonks": 0.9,
        "lineup": 0.9,
        "news": 0.55,
        "web": 0.35,
        "reddit": 0.25,
        "text": 0.35,
    }.get(source_kind, 0.35)
