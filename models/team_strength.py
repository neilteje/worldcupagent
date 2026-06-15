"""
Historical team-strength layer for the deterministic engine.

Turns a per-team pre-match state (Elo-like ``live_rating`` + ``base_rating``
+ accumulated xG / goals for & against) into a pair of expected goals
(lambda_home, lambda_away) for the Poisson model.

Design goals:
- Use ONLY pre-match information (the caller passes the pre-match snapshot).
- Degrade gracefully: a team with 0 matches played falls back to its rating
  prior (and ultimately the league average), never to garbage.
- Combine two independent supremacy signals — Elo/rating diff and xG form —
  with sample-size shrinkage so early-tournament noise is damped.
- Be fully deterministic and inspectable; every intermediate is returned.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrengthConfig:
    league_avg_goals: float = 1.30   # generic international goals/team/match prior (NOT fit to 2022)
    shrink_matches: float = 2.5      # pseudo-matches pulling xG form toward league average
    rating_to_goals: float = 0.90    # goal supremacy per 1.0 unit of rating difference
    rating_weight: float = 0.50      # weight on rating-supremacy vs xG-supremacy
    xg_weight_vs_goals: float = 0.65  # within form, trust xG over realized goals
    min_lambda: float = 0.20
    max_lambda: float = 4.0
    home_field_goals: float = 0.0    # 0 for a neutral-site tournament
    elo_blend: float = 0.30          # weight on the long-horizon Elo prior vs in-tournament form rating
    xg_regression_penalty: float = 0.20 # penalize overperformance in xG
    ko_experience_weight: float = 0.40  # additional weight to base_rating in knockouts
    fatigue_penalty_xg: float = 0.15    # penalty to attack/defense if rest < 72h
    continent_advantage_goals: float = 0.10 # goal supremacy edge if matching host continent
    talent_rating_bonus: float = 0.05   # weight given to squad talent depth score (0-5)
    h2h_xg_tilt: float = 0.10           # how much to tilt xG supremacy based on H2H win rate diff
    bz_xg_weight: float = 0.30          # weight to blend BZZOIRO xG into the form attack
    bz_momentum_weight: float = 0.15    # weight of BZZOIRO momentum added to effective rating


def effective_rating(state: dict, cfg: "StrengthConfig", is_knockout: bool = False) -> float:
    """Blend the proper chronological Elo (scaled) with the in-tournament form
    rating. Diversifying two rating estimators is more robust than either alone.
    Falls back to whichever signal is present."""
    live = float((state or {}).get("live_rating", 0.0) or 0.0)
    talent_bonus = float((state or {}).get("talent_score", 0.0) or 0.0) * cfg.talent_rating_bonus
    bz_momentum = float((state or {}).get("bzzoiro_momentum", 0.0) or 0.0) / 100.0 * cfg.bz_momentum_weight
    
    if "elo_scaled" not in (state or {}):
        return live + talent_bonus + bz_momentum
    elo = float(state.get("elo_scaled", 0.0) or 0.0)
    b = max(0.0, min(1.0, cfg.elo_blend))
    if is_knockout:
        b = max(0.0, min(1.0, b + cfg.ko_experience_weight))
    return b * elo + (1 - b) * live + talent_bonus + bz_momentum


def _team_form(state: dict, cfg: StrengthConfig) -> tuple[float, float]:
    """Return (attack, concede) expected goals/match, shrunk toward league avg."""
    avg = cfg.league_avg_goals
    matches = float((state or {}).get("matches", 0) or 0)
    if matches <= 0:
        return avg, avg
    xg_for = float(state.get("xg_for", 0.0) or 0.0) / matches
    xg_ag = float(state.get("xg_against", 0.0) or 0.0) / matches
    g_for = float(state.get("goals_for", 0.0) or 0.0) / matches
    g_ag = float(state.get("goals_against", 0.0) or 0.0) / matches
    w_xg = cfg.xg_weight_vs_goals
    attack = w_xg * xg_for + (1 - w_xg) * g_for
    concede = w_xg * xg_ag + (1 - w_xg) * g_ag
    
    # xG Regression to the mean
    if g_for > xg_for + 0.2:
        attack -= (g_for - xg_for) * cfg.xg_regression_penalty
    if g_ag < xg_ag - 0.2:
        concede += (xg_ag - g_ag) * cfg.xg_regression_penalty
        
    # Mix BZZOIRO xG
    bz_xg = float(state.get("bzzoiro_xg", 0.0))
    if bz_xg > 0.0:
        attack = (1 - cfg.bz_xg_weight) * attack + cfg.bz_xg_weight * bz_xg
        
    # Sample-size shrinkage toward the league average.
    shrink = matches / (matches + cfg.shrink_matches)
    attack = shrink * attack + (1 - shrink) * avg
    concede = shrink * concede + (1 - shrink) * avg

    # Fatigue penalty
    rest_hours = float((state or {}).get("rest_hours", 72.0) or 72.0)
    if rest_hours < 72.0:
        attack = max(0.1, attack - cfg.fatigue_penalty_xg)
        concede = concede + cfg.fatigue_penalty_xg

    return attack, concede


def expected_goals(home_state: dict, away_state: dict, *, cfg: StrengthConfig | None = None,
                   neutral: bool = True, is_knockout: bool = False, host_continent: str | None = None) -> dict:
    """Compute (lambda_home, lambda_away) plus all intermediates for inspection."""
    cfg = cfg or StrengthConfig()
    avg = cfg.league_avg_goals

    att_h, def_h = _team_form(home_state, cfg)
    att_a, def_a = _team_form(away_state, cfg)

    # xG-based expected goals: own attack scaled by opponent leakiness.
    lam_h_xg = att_h * (def_a / avg)
    lam_a_xg = att_a * (def_h / avg)
    sup_xg = lam_h_xg - lam_a_xg

    # Head-to-Head Edge tilt
    if "h2h_home_win_rate" in home_state and "h2h_home_win_rate" in away_state:
        h2h_edge = float(home_state["h2h_home_win_rate"]) - float(away_state["h2h_home_win_rate"])
        sup_xg += h2h_edge * cfg.h2h_xg_tilt

    total = lam_h_xg + lam_a_xg

    # Rating-based supremacy (independent prior; carries the early rounds).
    rating_h = effective_rating(home_state, cfg, is_knockout=is_knockout)
    rating_a = effective_rating(away_state, cfg, is_knockout=is_knockout)
    sup_rating = cfg.rating_to_goals * (rating_h - rating_a)

    rw = cfg.rating_weight
    supremacy = rw * sup_rating + (1 - rw) * sup_xg
    
    # Continent advantage
    if host_continent:
        home_cont = home_state.get("continent")
        away_cont = away_state.get("continent")
        if home_cont == host_continent and away_cont != host_continent:
            supremacy += cfg.continent_advantage_goals
        elif away_cont == host_continent and home_cont != host_continent:
            supremacy -= cfg.continent_advantage_goals

    if neutral:
        supremacy += cfg.home_field_goals
    else:
        supremacy += cfg.home_field_goals  # explicit home edge already folded above when 0

    lam_h = (total + supremacy) / 2.0
    lam_a = (total - supremacy) / 2.0
    lam_h = max(cfg.min_lambda, min(cfg.max_lambda, lam_h))
    lam_a = max(cfg.min_lambda, min(cfg.max_lambda, lam_a))

    return {
        "lambda_home": round(lam_h, 4),
        "lambda_away": round(lam_a, 4),
        "attack_home": round(att_h, 4),
        "attack_away": round(att_a, 4),
        "defense_home": round(def_h, 4),
        "defense_away": round(def_a, 4),
        "supremacy_rating": round(sup_rating, 4),
        "supremacy_xg": round(sup_xg, 4),
        "supremacy": round(supremacy, 4),
        "expected_total": round(total, 4),
    }


def elo_1x2(home_state: dict, away_state: dict, *, scale: float = 1.10,
            draw_base: float = 0.28, draw_decay: float = 0.05,
            cfg: "StrengthConfig | None" = None, is_knockout: bool = False) -> dict:
    """Direct Elo/rating outcome model (independent of the Poisson layer).

    Logistic on the rating difference for the win split, with a draw share that
    shrinks as the rating gap grows. Mirrors the old ``_rating_probs`` shape but
    is exposed as a standalone, configurable ensemble component.
    """
    import math
    from models.calibration import normalize_probs

    cfg = cfg or StrengthConfig()
    rating_h = effective_rating(home_state, cfg, is_knockout=is_knockout)
    rating_a = effective_rating(away_state, cfg, is_knockout=is_knockout)
    diff = max(-2.5, min(2.5, rating_h - rating_a))
    draw = max(0.16, min(0.34, draw_base - draw_decay * abs(diff)))
    home_no_draw = 1.0 / (1.0 + math.exp(-scale * diff))
    rem = 1.0 - draw
    return normalize_probs({"home": rem * home_no_draw, "draw": draw, "away": rem * (1.0 - home_no_draw)})


__all__ = ["StrengthConfig", "expected_goals", "elo_1x2", "effective_rating"]
