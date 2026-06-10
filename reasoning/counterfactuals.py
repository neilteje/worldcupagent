from __future__ import annotations

from models.calibration import OUTCOMES, normalize_probs


def decision_counterfactuals(
    *,
    model_probs: dict[str, float],
    market_probs: dict[str, float] | None,
    edge: dict,
    risk: dict,
    confidence: float,
    uncertainty: float,
    max_items: int = 6,
) -> list[dict]:
    """
    Explain what would flip the action.

    These are not recommendations to chase prices. They are auditable
    thresholds that make the decision policy legible.
    """
    facts: list[dict] = []
    model = normalize_probs(model_probs)
    market = normalize_probs(market_probs) if market_probs else None
    best = edge.get("best_outcome") or max(OUTCOMES, key=lambda k: model[k])
    best_edge = float(edge.get("best_edge") or 0.0)
    blocking = set(risk.get("blocking_risk_flags") or [])

    if market and best in OUTCOMES:
        needed_market = max(0.01, model[best] - 0.06)
        facts.append(
            {
                "condition": f"{best}_market_price <= {needed_market:.3f}",
                "effect": "edge_tier_reaches_medium_candidate",
                "reason": "A medium edge requires roughly six probability points between model and market.",
            }
        )
        needed_model = min(0.99, market[best] + 0.06)
        facts.append(
            {
                "condition": f"{best}_model_probability >= {needed_model:.3f}",
                "effect": "edge_tier_reaches_medium_candidate",
                "reason": "Model probability would need to rise enough to clear the medium-edge threshold.",
            }
        )
    else:
        facts.append(
            {
                "condition": "complete_market_snapshot_available",
                "effect": "order_gate_can_evaluate_price_edge",
                "reason": "Predictions can be scored without market data, but orders need complete market prices.",
            }
        )

    if "confidence_too_low" in blocking or confidence < 0.58:
        facts.append(
            {
                "condition": "confidence >= 0.58",
                "effect": "medium_edge_confidence_gate_can_pass",
                "reason": f"Current confidence is {confidence:.3f}.",
            }
        )
    if "uncertainty_too_high" in blocking or uncertainty > 0.52:
        facts.append(
            {
                "condition": "uncertainty <= 0.52",
                "effect": "uncertainty_gate_can_pass",
                "reason": f"Current uncertainty is {uncertainty:.3f}.",
            }
        )
    if "lineup_unconfirmed_and_edge_lineup_driven" in blocking:
        facts.append(
            {
                "condition": "both_lineups_confirmed_by_official_source",
                "effect": "lineup_driven_edge_can_be_considered",
                "reason": "Lineup edges are blocked until official lineup data confirms the shock.",
            }
        )
    if "bookmaker_market_agree_against_model" in blocking:
        facts.append(
            {
                "condition": "bookmaker_no_longer_agrees_with_market_against_model",
                "effect": "source_conflict_gate_can_relax",
                "reason": "The policy blocks most model-only edges when both external price references disagree.",
            }
        )
    if "dry_run_enabled" in blocking:
        facts.append(
            {
                "condition": "DRY_RUN=false",
                "effect": "orders_can_be_submitted_after_all_other_gates_pass",
                "reason": "Dry-run mode intentionally records predictions and skips live order submission.",
            }
        )

    if edge.get("edge_tier") != "none" and not blocking:
        facts.append(
            {
                "condition": f"best_edge falls below 0.03 from current {best_edge:.3f}",
                "effect": "action_flips_to_skip",
                "reason": "The no-edge threshold is three probability points.",
            }
        )

    return facts[:max_items]
