from __future__ import annotations


def dynamic_source_weights(
    base_weights: dict[str, float],
    *,
    archetype: dict | None = None,
    data_completeness: dict | None = None,
    market_stale: dict | None = None,
    source_reconciliation: dict | None = None,
) -> dict:
    """
    Return context-adjusted source weights with an audit trail.

    This is intentionally deterministic and conservative. It does not try to
    learn online yet; it creates the contract where future postmortem-derived
    reliability tables can plug in.
    """
    weights = {k: max(0.0, float(v or 0.0)) for k, v in base_weights.items()}
    reasons: list[str] = []
    tags = set((archetype or {}).get("tags") or [])
    market_regime = (archetype or {}).get("market_regime")
    match_archetype = (archetype or {}).get("match_archetype")
    draw_regime = (archetype or {}).get("draw_regime")

    if "thin_data" in tags:
        _mul(weights, {"bookmaker": 1.15, "polymarket": 1.10, "supabase": 0.80, "sportmonks": 0.90})
        reasons.append("Thin data: lean slightly more on liquid reference prices.")
    elif "rich_data" in tags:
        _mul(weights, {"bookmaker": 1.02, "sportmonks": 1.03, "supabase": 1.03})
        reasons.append("Rich data: allow football-specific sources to contribute normally.")

    if market_regime == "market_against_model_bookmaker":
        _mul(weights, {"bookmaker": 1.12, "sportmonks": 1.06, "polymarket": 0.82})
        reasons.append("Bookmaker/model disagreement with market: reduce market anchoring.")
    elif market_regime == "model_against_market_bookmaker":
        _mul(weights, {"bookmaker": 1.10, "polymarket": 1.08, "sportmonks": 0.88, "supabase": 0.92})
        reasons.append("Market/bookmaker agree against model: shrink toward external consensus.")
    elif market_regime == "market_stale":
        _mul(weights, {"polymarket": 0.74, "bookmaker": 1.14, "sportmonks": 1.08})
        reasons.append("Stale-market evidence: reduce current Polymarket weight.")

    if match_archetype == "ht_low_xg_level":
        _mul(weights, {"halftime": 1.20, "prematch": 0.86, "bookmaker": 0.96, "polymarket": 0.92})
        reasons.append("HT low-xG level state: increase live-state/draw persistence weight.")
    elif match_archetype == "ht_scoreline_luck":
        _mul(weights, {"halftime": 1.16, "prematch": 1.05, "polymarket": 0.88})
        reasons.append("HT scoreline luck: trust performance update over raw market reaction.")
    elif match_archetype == "strong_favorite" and market_regime == "market_consensus":
        _mul(weights, {"bookmaker": 1.16, "polymarket": 1.12, "sportmonks": 0.86, "supabase": 0.82, "draw_model": 0.90})
        reasons.append("Strong favorite with market/bookmaker consensus: reduce neutral-prior longshot inflation.")
    elif match_archetype == "balanced_match":
        _mul(weights, {"draw_model": 1.15, "supabase": 1.04})
        reasons.append("Balanced match: draw model and priors matter more.")

    if draw_regime == "draw_elevated":
        _mul(weights, {"draw_model": 1.18})
        reasons.append("Elevated draw regime: preserve draw-model signal.")

    completeness_score = float((data_completeness or {}).get("score", 1.0) or 0.0)
    if completeness_score < 0.55:
        _mul(weights, {"llm_claims": 0.65, "lineup": 0.75})
        reasons.append("Low completeness: haircut fragile side-channel signals.")

    if source_reconciliation and "source_divergence_high" in (source_reconciliation.get("flags") or []):
        _mul(weights, {"sportmonks": 0.92, "supabase": 0.92})
        reasons.append("High source divergence: damp model-only football sources.")

    normalized = _normalize(weights)
    return {
        "weights": normalized,
        "raw_weights": weights,
        "base_weights": dict(base_weights),
        "reasons": reasons or ["Default source weights retained."],
        "archetype_tags": sorted(tags),
    }


def _mul(weights: dict[str, float], multipliers: dict[str, float]) -> None:
    for key, multiplier in multipliers.items():
        if key in weights:
            weights[key] *= max(0.0, float(multiplier))


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v or 0.0)) for v in weights.values())
    if total <= 0:
        return dict(weights)
    return {k: max(0.0, float(v or 0.0)) / total for k, v in weights.items()}
