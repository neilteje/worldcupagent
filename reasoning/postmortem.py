from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json


def classify_postmortem(decision: dict, result: dict) -> dict:
    final_outcome = str(result.get("outcome") or result.get("winner") or "").lower()
    predicted = str(decision.get("best_outcome") or "").lower()
    action = str(decision.get("action") or "").upper()
    risk_flags = list(decision.get("risk_flags") or [])
    top_signals = list(decision.get("top_signals") or [])
    correct = bool(final_outcome and predicted == final_outcome)
    error_type = "correct" if correct else "wrong_side"
    if action != "BET" and correct and float(decision.get("best_edge") or 0.0) >= 0.03:
        error_type = "missed_edge"
    if action == "BET" and not correct and "source_divergence_high" in risk_flags:
        error_type = "ignored_source_divergence"
    if action == "BET" and not correct and "lineup_unconfirmed" in risk_flags:
        error_type = "lineup_uncertainty"
    bad_signal = None
    if not correct and top_signals:
        bad_signal = max(top_signals, key=lambda s: float(s.get("impact", 0.0) or 0.0)).get("name")
    missed_signal = None
    if error_type == "missed_edge":
        missed_signal = "edge_threshold_or_order_gate"
    suggestions = []
    if error_type in {"ignored_source_divergence", "lineup_uncertainty"}:
        suggestions.append("Consider lowering max_order_usd or increasing required confidence for this flag class; add regression tests first.")
    if error_type == "missed_edge":
        suggestions.append("Review edge thresholds against backtest distribution before changing MIN_EDGE_TO_BET.")
    return {
        "fixture_code": decision.get("fixture_code"),
        "window": decision.get("window"),
        "predicted": predicted,
        "actual": final_outcome,
        "action": action,
        "correct": correct,
        "error_type": error_type,
        "bad_signal": bad_signal,
        "missed_signal": missed_signal,
        "suggested_config_changes": suggestions,
        "auto_apply": False,
    }


def record_postmortem(storage_dir: Path, decision: dict, result: dict) -> dict:
    postmortem = classify_postmortem(decision, result)
    path = Path(storage_dir) / "postmortems"
    path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_path = path / f"{decision.get('fixture_code', 'fixture')}_{decision.get('window', 'window')}_{stamp}.json"
    file_path.write_text(json.dumps(postmortem, indent=2), encoding="utf-8")
    return {**postmortem, "artifact": str(file_path)}
