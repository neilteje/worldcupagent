from __future__ import annotations
from models.calibration import valid_probs

BLOCKING = {"probabilities_invalid", "market_data_missing_for_order", "lineup_unconfirmed_and_edge_lineup_driven", "bookmaker_market_agree_against_model", "confidence_too_low", "uncertainty_too_high", "edge_below_threshold", "dry_run_enabled"}

def audit_decision(probabilities: dict[str, float], edge: dict, confidence: float, uncertainty: float, dry_run: bool, market_complete: bool, lineup_result: dict | None = None, consensus_case: str | None = None) -> dict:
    flags: list[str] = []
    if not valid_probs(probabilities): flags.append("probabilities_invalid")
    if not market_complete: flags.append("market_data_missing_for_order")
    if confidence < .55: flags.append("confidence_too_low")
    if uncertainty > .45: flags.append("uncertainty_too_high")
    if edge.get("edge_tier") == "none": flags.append("edge_below_threshold")
    if dry_run: flags.append("dry_run_enabled")
    if lineup_result and "lineup_unconfirmed" in lineup_result.get("risk_flags", []) and edge.get("edge_type") == "lineup_not_priced_in": flags.append("lineup_unconfirmed_and_edge_lineup_driven")
    if consensus_case == "bookmaker_polymarket_vs_model": flags.append("bookmaker_market_agree_against_model")
    blocking = [f for f in flags if f in BLOCKING]
    return {"risk_flags": flags, "blocking_risk_flags": blocking, "order_allowed": len(blocking) == 0}
