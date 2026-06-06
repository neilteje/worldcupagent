from __future__ import annotations
from models.calibration import valid_probs

BLOCKING = {"probabilities_invalid", "market_data_missing_for_order", "lineup_unconfirmed_and_edge_lineup_driven", "bookmaker_market_agree_against_model", "confidence_too_low", "uncertainty_too_high", "edge_below_threshold", "dry_run_enabled", "duplicate_order", "data_completeness_too_low", "edge_tier_confidence_mismatch", "source_divergence_high", "llm_analysis_missing", "llm_analysis_failed", "llm_recommends_skip", "llm_recommends_watch", "llm_veto", "llm_not_approving_order"}

def audit_decision(probabilities: dict[str, float], edge: dict, confidence: float, uncertainty: float, dry_run: bool, market_complete: bool, lineup_result: dict | None = None, consensus_case: str | None = None, *, duplicate_order: bool = False, data_completeness: float | None = None, extra_flags: list[str] | None = None) -> dict:
    flags: list[str] = list(extra_flags or [])
    if not valid_probs(probabilities): flags.append("probabilities_invalid")
    if not market_complete: flags.append("market_data_missing_for_order")
    if confidence < .55 - 1e-9: flags.append("confidence_too_low")
    if uncertainty > .45: flags.append("uncertainty_too_high")
    if data_completeness is not None and data_completeness < .45: flags.append("data_completeness_too_low")
    if edge.get("edge_tier") == "none": flags.append("edge_below_threshold")
    if edge.get("edge_tier") == "strong" and confidence < .65:
        flags.append("edge_tier_confidence_mismatch")
    if edge.get("edge_tier") == "medium" and confidence < .60:
        flags.append("edge_tier_confidence_mismatch")
    if edge.get("edge_tier") == "soft" and confidence < .75:
        flags.append("confidence_insufficient_for_soft_edge")
    if edge.get("edge_tier") == "none" and abs(float(edge.get("best_edge") or 0.0)) < .03:
        flags.append("edge_margin_within_noise")
    if dry_run: flags.append("dry_run_enabled")
    if duplicate_order: flags.append("duplicate_order")
    if lineup_result and "lineup_unconfirmed" in lineup_result.get("risk_flags", []) and edge.get("edge_type") == "lineup_not_priced_in": flags.append("lineup_unconfirmed_and_edge_lineup_driven")
    if consensus_case == "bookmaker_polymarket_vs_model": flags.append("bookmaker_market_agree_against_model")
    if consensus_case == "all_disagree": flags.append("source_disagreement_unresolved")
    if consensus_case in {"bookmaker_polymarket_vs_model", "all_disagree"}:
        flags.append("multi_source_conflict")
    deduped = list(dict.fromkeys(flags))
    blocking = [f for f in deduped if f in BLOCKING]
    return {"risk_flags": deduped, "blocking_risk_flags": blocking, "order_allowed": len(blocking) == 0}
