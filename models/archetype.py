from __future__ import annotations

from models.calibration import OUTCOMES, normalize_probs


def _usable(probs: dict[str, float] | None) -> bool:
    return bool(probs) and sum(float((probs or {}).get(k, 0.0) or 0.0) for k in OUTCOMES) > 0


def _pick(probs: dict[str, float] | None) -> str | None:
    if not _usable(probs):
        return None
    p = normalize_probs(probs)
    return max(OUTCOMES, key=lambda k: p[k])


def _gap(probs: dict[str, float] | None) -> float:
    if not _usable(probs):
        return 0.0
    p = sorted(normalize_probs(probs).values(), reverse=True)
    return p[0] - p[1]


def classify_match_archetype(
    *,
    window: str,
    model_probs: dict[str, float] | None = None,
    sportmonks_probs: dict[str, float] | None = None,
    bookmaker_probs: dict[str, float] | None = None,
    market_probs: dict[str, float] | None = None,
    lineup: dict | None = None,
    halftime: dict | None = None,
    live: dict | None = None,
    market_stale: dict | None = None,
    data_completeness: dict | None = None,
) -> dict:
    """
    Classify the decision context into reusable regimes.

    The labels are deliberately coarse: they are stable enough to drive source
    reliability, reporting, and future council reconciliation without making
    brittle tactical claims from thin data.
    """
    window_n = "HT" if str(window).upper() in {"HT", "HALFTIME"} else "PRE_MATCH"
    model_pick = _pick(model_probs)
    market_pick = _pick(market_probs)
    book_pick = _pick(bookmaker_probs)
    sm_pick = _pick(sportmonks_probs)
    tags: list[str] = [f"window_{window_n.lower()}"]
    risks: list[str] = []

    if data_completeness:
        score = float(data_completeness.get("score", 0.0) or 0.0)
        if score < 0.45:
            tags.append("thin_data")
            risks.append("archetype_low_data_completeness")
        elif score < 0.70:
            tags.append("partial_data")
        else:
            tags.append("rich_data")

    favorite_gap = max(_gap(bookmaker_probs), _gap(market_probs), _gap(model_probs))
    if favorite_gap >= 0.24:
        match_archetype = "strong_favorite"
        tags.append("large_favorite_gap")
    elif favorite_gap <= 0.08:
        match_archetype = "balanced_match"
        tags.append("small_strength_gap")
    else:
        match_archetype = "moderate_favorite"

    if window_n == "HT":
        live = live or {}
        home_goals = int(float(live.get("home_goals", 0) or 0))
        away_goals = int(float(live.get("away_goals", 0) or 0))
        home_xg = float(live.get("home_xg", 0.0) or 0.0)
        away_xg = float(live.get("away_xg", 0.0) or 0.0)
        total_xg = home_xg + away_xg
        if home_goals == away_goals and total_xg < 0.75:
            match_archetype = "ht_low_xg_level"
            tags.extend(["level_score", "low_ht_xg"])
        elif home_goals == away_goals:
            match_archetype = "ht_level_state"
            tags.append("level_score")
        elif abs(home_xg - away_xg) >= 0.75 and ((home_goals > away_goals and home_xg < away_xg) or (away_goals > home_goals and away_xg < home_xg)):
            match_archetype = "ht_scoreline_luck"
            tags.append("scoreline_xg_divergence")
        if int(float(live.get("home_red", 0) or 0)) or int(float(live.get("away_red", 0) or 0)):
            tags.append("red_card_state")

    if lineup and "lineup_unconfirmed" in (lineup.get("risk_flags") or []):
        lineup_regime = "lineup_unconfirmed"
    elif lineup and lineup.get("lineup_shock"):
        lineup_regime = "lineup_shock"
        tags.append("lineup_shock")
    else:
        lineup_regime = "lineup_normal_or_unknown"

    market_regime = "market_missing"
    if _usable(market_probs):
        if model_pick and market_pick and model_pick != market_pick:
            if book_pick == model_pick:
                market_regime = "market_against_model_bookmaker"
                tags.append("model_bookmaker_vs_market")
            elif book_pick == market_pick:
                market_regime = "model_against_market_bookmaker"
                tags.append("bookmaker_market_vs_model")
            else:
                market_regime = "three_way_source_disagreement"
                tags.append("all_sources_disagree")
        else:
            market_regime = "market_consensus"
        if market_stale and market_stale.get("is_stale"):
            market_regime = "market_stale"
            tags.append("stale_market")

    draw_regime = "draw_normal"
    draw_refs = [p.get("draw") for p in [model_probs or {}, market_probs or {}, bookmaker_probs or {}] if p and p.get("draw") is not None]
    if draw_refs:
        avg_draw = sum(float(x) for x in draw_refs) / len(draw_refs)
        if avg_draw >= 0.34 or "low_ht_xg" in tags:
            draw_regime = "draw_elevated"
        elif avg_draw <= 0.18 and "large_favorite_gap" in tags:
            draw_regime = "draw_suppressed_by_favorite"

    return {
        "match_archetype": match_archetype,
        "market_regime": market_regime,
        "draw_regime": draw_regime,
        "lineup_regime": lineup_regime,
        "model_pick": model_pick,
        "market_pick": market_pick,
        "bookmaker_pick": book_pick,
        "sportmonks_pick": sm_pick,
        "tags": list(dict.fromkeys(tags)),
        "risk_flags": risks,
    }
