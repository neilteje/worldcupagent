from __future__ import annotations

from pathlib import Path
import json

from reasoning.anthropic_review import _extract_text, _extract_thinking, _model_candidates, _parse_json, _post_with_optional_thinking, _write_artifact

CENTRAL_MODELS = (
    "claude-haiku-4-5-20251001",
    "claude-3-5-haiku-latest",
    "claude-sonnet-4-5-20250929",
    "claude-3-5-sonnet-latest",
)


def central_match_forecast_with_anthropic(settings, decision_context: dict, *, storage_dir: Path | None = None) -> dict:
    key = getattr(settings, "anthropic_key", "") or ""
    if not key:
        return {"called": False, "ok": False, "provider": "anthropic", "role": "central_forecaster", "reason": "anthropic_key_missing"}

    compact = _compact_context(decision_context)
    prompt = (
        "You are the primary forecaster for a soccer prediction-market agent. "
        "You must synthesize all provided features into one final 3-way probability forecast. "
        "Use every relevant feature: Sportmonks, bookmaker, Polymarket, Supabase priors, lineups, halftime state, "
        "source reconciliation, claim-extracted signals, stale-market detection, confidence, and uncertainty clues. "
        "Do not default to SKIP because of missing secondary features when bookmaker, market, and model signals are aligned. "
        "Use missing data as a confidence haircut, not an automatic veto, unless there is severe contradictory evidence. "
        "Return only minified JSON. No markdown fences. "
        "Schema: {"
        "\"probabilities\":{\"home\":number,\"draw\":number,\"away\":number},"
        "\"confidence\":number,"
        "\"uncertainty\":number,"
        "\"recommendation\":\"BET|SKIP|WATCH\","
        "\"risk_posture\":\"approve|caution|veto\","
        "\"supporting_signals\":[up to 4 short strings],"
        "\"contradicting_signals\":[up to 4 short strings],"
        "\"additional_risk_flags\":[up to 4 snake_case strings],"
        "\"rationale\":\"short string\","
        "\"order_authorization\":\"NO_AUTHORITY\""
        "}.\n\n"
        f"Decision context JSON:\n{json.dumps(compact, sort_keys=True)[:14000]}"
    )

    errors: list[str] = []
    for model in _model_candidates(CENTRAL_MODELS):
        try:
            response = _post_with_optional_thinking(key, model, prompt, max_tokens=2200, thinking_budget=2048, timeout=60)
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
                "role": "central_forecaster",
                "model": body.get("model") or model,
                "usage": body.get("usage") or {},
                "parsed": _parse_json(text),
                "internal_reasoning": thinking[:4000],
                "text_preview": text[:2000],
                "order_authorization_allowed": False,
            }
            if storage_dir is not None:
                result["artifact"] = str(_write_artifact(storage_dir, result, prefix="anthropic_central_forecast"))
            return result
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {exc}")
    return {
        "called": True,
        "ok": False,
        "provider": "anthropic",
        "role": "central_forecaster",
        "errors": errors[-4:],
    }


def _compact_context(context: dict) -> dict:
    keys = (
        "fixture_code",
        "window",
        "market_probs",
        "bookmaker_probs",
        "sportmonks_probs",
        "supabase_priors",
        "lineup",
        "halftime",
        "structured_claims",
        "claim_signals",
        "source_reconciliation",
        "market_stale",
        "data_completeness",
        "signal_conflict",
        "deterministic_prematch",
        "deterministic_model",
        "consensus_case_hint",
        "top_signals",
        "dry_run",
    )
    return {k: context.get(k) for k in keys if k in context}
