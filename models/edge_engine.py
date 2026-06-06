from __future__ import annotations
from models.calibration import OUTCOMES, normalize_probs


def _tier(edge: float) -> str:
    if edge < 0.03: return "none"
    if edge < 0.06: return "soft"
    if edge < 0.10: return "medium"
    return "strong"


def _pick(probs: dict[str, float] | None) -> str | None:
    if not probs: return None
    p = normalize_probs(probs)
    return max(OUTCOMES, key=lambda k: p[k])


def evaluate_edge(fixture_code: str, window: str, final_model_probs: dict[str, float], market_probs: dict[str, float] | None, bookmaker_probs: dict[str, float] | None, confidence_score: float, uncertainty_score: float, consensus_case: str, signals: list[str] | None = None) -> dict:
    model = normalize_probs(final_model_probs)
    if not market_probs:
        return {"edges": {k: 0.0 for k in OUTCOMES}, "best_outcome": None, "best_edge": 0.0, "edge_tier": "none", "edge_type": "no_clear_edge", "should_bet": False, "reason": "Market data missing; prediction allowed but order blocked."}
    market = normalize_probs(market_probs)
    edges = {k: model[k] - market[k] for k in OUTCOMES}
    best_outcome = max(OUTCOMES, key=lambda k: edges[k])
    best_edge = edges[best_outcome]
    tier = _tier(best_edge)
    signal_text = " ".join(signals or []).lower()
    model_pick, market_pick, book_pick = _pick(model), _pick(market), _pick(bookmaker_probs)
    supported = book_pick == model_pick and market_pick != model_pick
    against = book_pick is not None and book_pick == market_pick and model_pick != market_pick
    edge_type = "no_clear_edge"
    if tier == "none":
        edge_type = "no_clear_edge"
    elif consensus_case == "model_bookmaker_vs_polymarket" or supported:
        edge_type = "model_bookmaker_vs_polymarket"
    elif best_outcome == "draw" and best_edge >= 0.03:
        edge_type = "draw_underpriced"
    elif ("lineup shock" in signal_text or "missing impact" in signal_text or "goalkeeper" in signal_text or "formation" in signal_text) and "unconfirmed" not in signal_text:
        edge_type = "lineup_not_priced_in"
    elif window.upper() == "HT" and ("luck" in signal_text or "overreaction" in signal_text or "comeback" in signal_text):
        edge_type = "ht_scoreline_overreaction"
    elif "stale" in signal_text:
        edge_type = "market_stale"
    elif tier != "none":
        edge_type = "model_only_edge"
    reasons = [f"best {best_outcome} edge {best_edge:+.3f} is {tier}."]
    should_bet = tier in {"medium", "strong"} or (tier == "soft" and consensus_case == "model_bookmaker_vs_polymarket" and confidence_score >= 0.75)
    if tier == "none": should_bet = False; reasons.append("Edge below 3% threshold.")
    if confidence_score < 0.55: should_bet = False; reasons.append("Confidence below 0.55.")
    if uncertainty_score > 0.45: should_bet = False; reasons.append("Uncertainty above 0.45.")
    if signal_text and all(w in signal_text for w in ["sentiment"]): should_bet = False; reasons.append("Sentiment-only edge is not tradable.")
    if against and edge_type not in {"lineup_not_priced_in", "ht_scoreline_overreaction"}:
        should_bet = False; reasons.append("Bookmaker and Polymarket agree against the model.")
    if supported: reasons.append("Bookmaker supports model against market.")
    return {"edges": edges, "best_outcome": best_outcome, "best_edge": best_edge, "edge_tier": tier, "edge_type": edge_type, "should_bet": should_bet, "reason": " ".join(reasons)}
