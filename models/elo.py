"""
Proper chronological Elo ratings for international teams.

Replaces the old crude additive 2018 win/loss tally with a standard Elo system
(logistic expectation, goal-difference-weighted K-factor) updated match-by-match
in date order. Used to produce a *pre-match* rating for every fixture — the
rating is read before the match and updated only afterwards, so there is no
future leakage.

Pure, deterministic, no I/O: the caller passes an already-sorted list of match
result dicts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EloConfig:
    base: float = 1500.0
    k: float = 26.0            # base K-factor
    gd_weight: bool = True     # scale K by margin of victory (World Football Elo style)
    scale: float = 400.0       # logistic scale
    rating_unit: float = 280.0  # Elo points per 1.0 of the model's rating-diff scale


def expected_home(rating_home: float, rating_away: float, *, hfa: float = 0.0, scale: float = 400.0) -> float:
    """Expected score (win=1, draw=0.5) for the home/first side."""
    return 1.0 / (1.0 + 10 ** (-((rating_home + hfa) - rating_away) / scale))


def _gd_multiplier(goal_diff: int) -> float:
    g = abs(int(goal_diff))
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11 + g) / 8.0  # 3 -> 1.75, 4 -> 1.875, ...


def update_pair(rating_home: float, rating_away: float, home_score: int, away_score: int,
                *, cfg: EloConfig, hfa: float = 0.0) -> tuple[float, float]:
    exp_h = expected_home(rating_home, rating_away, hfa=hfa, scale=cfg.scale)
    if home_score > away_score:
        score_h = 1.0
    elif home_score < away_score:
        score_h = 0.0
    else:
        score_h = 0.5
    k = cfg.k * (_gd_multiplier(home_score - away_score) if cfg.gd_weight else 1.0)
    delta = k * (score_h - exp_h)
    return rating_home + delta, rating_away - delta


def build_timeline(matches: list[dict], *, cfg: EloConfig | None = None) -> dict:
    """Compute pre-match Elo ratings for a date-sorted match list.

    Each match dict needs: ``match_id``, ``home_key``, ``away_key``,
    ``home_score``, ``away_score``, optional ``neutral``/``hfa``.

    Returns ``{match_id: {"home": pre_elo, "away": pre_elo,
    "home_diff_scaled": x, "away_diff_scaled": x}}`` where the scaled values are
    in the model's rating-diff units (Elo / rating_unit, centred on base).
    """
    cfg = cfg or EloConfig()
    ratings: dict[str, float] = {}
    out: dict = {}
    for m in matches:
        hk, ak = m["home_key"], m["away_key"]
        rh = ratings.setdefault(hk, cfg.base)
        ra = ratings.setdefault(ak, cfg.base)
        hfa = float(m.get("hfa", 0.0) or 0.0)
        out[m["match_id"]] = {
            "home": round(rh, 2),
            "away": round(ra, 2),
            "home_scaled": round((rh - cfg.base) / cfg.rating_unit, 4),
            "away_scaled": round((ra - cfg.base) / cfg.rating_unit, 4),
        }
        nh, na = update_pair(rh, ra, int(m["home_score"]), int(m["away_score"]), cfg=cfg, hfa=hfa)
        ratings[hk], ratings[ak] = nh, na
    return out


__all__ = ["EloConfig", "expected_home", "update_pair", "build_timeline"]
