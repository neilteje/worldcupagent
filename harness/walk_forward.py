"""Chronological walk-forward backtest + parameter sweep + ablations (spec §30).

Leakage-free by construction: matches are processed in kickoff order; each match
is PREDICTED from team states built only from STRICTLY EARLIER matches, then used
to update the Elo / rolling-form state. Labels come from real final scores.

Data: StatsBomb World Cup 2018 + 2022 match dumps cached in
``storage/backtests/cache/wc2022/``. xG is not in the match list, so realized
goals serve as the form proxy (flagged in coverage).

Public API:
    load_matches() -> list[Match]
    walk_forward(matches, params, eval_season) -> dict   # metrics + per-match
    sweep(matches, grid) -> list[dict]                    # ranked configs
    ablations(matches, best_params) -> dict
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from models.calibration import OUTCOMES
from models.chronological_elo import ChronologicalEloBuilder
from models.deterministic_v2 import EnsembleConfig, predict_v2
from models.rolling_form import RollingFormBuilder
from models.team_state_builder import build_team_state
from models.team_strength import StrengthConfig

_CACHE = Path("storage/backtests/cache/wc2022")
_FILES = {2018: _CACHE / "statsbomb_matches_43_3.json",
          2022: _CACHE / "statsbomb_matches_43_106.json"}


@dataclass(frozen=True)
class Match:
    match_id: str
    date: datetime
    season: int
    home: str
    away: str
    home_goals: int
    away_goals: int
    is_knockout: bool

    @property
    def result(self) -> str:
        if self.home_goals > self.away_goals:
            return "home"
        if self.home_goals < self.away_goals:
            return "away"
        return "draw"


def _team_name(node) -> str:
    if isinstance(node, dict):
        return node.get("home_team_name") or node.get("away_team_name") or node.get("country", {}).get("name") or str(node.get("home_team_id") or node.get("away_team_id"))
    return str(node)


def load_matches() -> list[Match]:
    out: list[Match] = []
    for season, path in _FILES.items():
        if not path.exists():
            continue
        for m in json.loads(path.read_text()):
            try:
                stage = (m.get("competition_stage") or {}).get("name", "")
                date = datetime.fromisoformat(m["match_date"])
                out.append(Match(
                    match_id=str(m["match_id"]), date=date, season=season,
                    home=_team_name(m["home_team"]), away=_team_name(m["away_team"]),
                    home_goals=int(m["home_score"]), away_goals=int(m["away_score"]),
                    is_knockout="group" not in stage.lower(),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    out.sort(key=lambda x: x.date)
    return out


def _cfg_from_params(p: dict) -> EnsembleConfig:
    return EnsembleConfig(
        w_elo=p["w_elo"], w_poisson=p["w_poisson"], w_market=0.0,
        use_market=False,
        temperature=p["temperature"], base_rate_shrink=p["base_rate_shrink"],
        strength=StrengthConfig(rating_weight=p["rating_weight"]),
    )


def walk_forward(matches: list[Match], params: dict, *, eval_season: int = 2022) -> dict:
    elo = ChronologicalEloBuilder(k_factor=params["k_factor"])
    form = RollingFormBuilder(short_half_life_days=params.get("short_hl", 75.0),
                              long_half_life_days=params.get("long_hl", 240.0))
    last_played: dict[str, datetime] = {}
    cfg = _cfg_from_params(params)

    rows = []
    for m in matches:
        if m.season == eval_season:
            # Predict BEFORE updating state — strictly prior info only.
            hs_rating, as_rating = elo.scaled(m.home), elo.scaled(m.away)
            rest_h = _rest(last_played, m.home, m.date)
            rest_a = _rest(last_played, m.away, m.date)
            home_state = build_team_state(m.home, m.home, elo_scaled=hs_rating,
                                          form=form.form(m.home, m.date), rest_hours=rest_h,
                                          opponent_scaled=as_rating).model_state()
            away_state = build_team_state(m.away, m.away, elo_scaled=as_rating,
                                          form=form.form(m.away, m.date), rest_hours=rest_a,
                                          opponent_scaled=hs_rating).model_state()
            out = predict_v2(home_state, away_state, market_probs=None, cfg=cfg,
                             neutral=True, is_knockout=m.is_knockout)
            rows.append((out["probabilities"], m.result))

        # Update state with the actual result (now it is "past").
        elo.process_match(m.home, m.away, m.home_goals, m.away_goals,
                          is_competitive=True, is_neutral=True)
        form.add_match(m.home, m.date, m.home_goals, m.away_goals)
        form.add_match(m.away, m.date, m.away_goals, m.home_goals)
        last_played[m.home] = m.date
        last_played[m.away] = m.date

    return {**_score(rows), "n": len(rows), "params": params}


def _rest(last: dict, team: str, date: datetime) -> float:
    if team not in last:
        return 96.0
    return max(24.0, (date - last[team]).total_seconds() / 3600.0)


# ── metrics ────────────────────────────────────────────────────────────────

def _score(rows: list[tuple]) -> dict:
    if not rows:
        return {"brier": None, "logloss": None, "rps": None, "accuracy": None, "ece": None}
    brier = logloss = rps = hits = 0.0
    buckets: dict[int, list] = {}
    for probs, result in rows:
        p = [probs["home"], probs["draw"], probs["away"]]
        y = [1.0 if o == result else 0.0 for o in OUTCOMES]
        brier += sum((pi - yi) ** 2 for pi, yi in zip(p, y))
        logloss += -math.log(max(1e-9, probs[result]))
        # Ranked probability score (ordered home<draw<away cumulative).
        cum_p = cum_y = 0.0
        for pi, yi in zip(p, y):
            cum_p += pi; cum_y += yi
            rps += (cum_p - cum_y) ** 2
        pick = OUTCOMES[max(range(3), key=lambda i: p[i])]
        hits += 1.0 if pick == result else 0.0
        conf = max(p)
        b = min(9, int(conf * 10))
        buckets.setdefault(b, []).append((conf, 1.0 if pick == result else 0.0))
    n = len(rows)
    ece = 0.0
    for b, items in buckets.items():
        avg_conf = sum(c for c, _ in items) / len(items)
        acc = sum(h for _, h in items) / len(items)
        ece += (len(items) / n) * abs(avg_conf - acc)
    return {
        "brier": round(brier / n, 4),
        "logloss": round(logloss / n, 4),
        "rps": round(rps / (2.0 * n), 4),   # /2 so a "1 step off" miss scores ~0.5
        "accuracy": round(hits / n, 4),
        "ece": round(ece, 4),
    }


# ── sweep + ablations ────────────────────────────────────────────────────────

DEFAULT_GRID = {
    "split": [(1.0, 0.0), (0.85, 0.15), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7)],
    "temperature": [0.9, 1.0, 1.1, 1.25, 1.4],
    "base_rate_shrink": [0.0, 0.1, 0.2, 0.3],
    "rating_weight": [0.4, 0.6, 0.8],
    "k_factor": [20.0, 40.0, 60.0],
}


def sweep(matches: list[Match], grid: dict | None = None, *, eval_season: int = 2022) -> list[dict]:
    grid = grid or DEFAULT_GRID
    results = []
    for (we, wp) in grid["split"]:
        for temp in grid["temperature"]:
            for shrink in grid["base_rate_shrink"]:
                for rw in grid["rating_weight"]:
                    for k in grid["k_factor"]:
                        params = {"w_elo": we, "w_poisson": wp, "temperature": temp,
                                  "base_rate_shrink": shrink, "rating_weight": rw, "k_factor": k}
                        results.append(walk_forward(matches, params, eval_season=eval_season))
    # Rank by log loss (proper score), then Brier.
    results.sort(key=lambda r: (r["logloss"], r["brier"]))
    return results


def ablations(matches: list[Match], best_params: dict, *, eval_season: int = 2022) -> dict:
    """Component ablations under the best calibration params."""
    out = {}
    variants = {
        "full": dict(best_params),
        "elo_only": {**best_params, "w_elo": 1.0, "w_poisson": 0.0},
        "poisson_only": {**best_params, "w_elo": 0.0, "w_poisson": 1.0},
        "no_calibration": {**best_params, "temperature": 1.0, "base_rate_shrink": 0.0},
    }
    for name, p in variants.items():
        out[name] = walk_forward(matches, p, eval_season=eval_season)
    return out


def market_baseline(matches: list[Match], *, eval_season: int = 2022) -> dict:
    """Uniform-prior baseline (no market odds available offline) for reference."""
    rows = [({"home": 0.40, "draw": 0.26, "away": 0.34}, m.result)
            for m in matches if m.season == eval_season]
    return {**_score(rows), "n": len(rows), "params": {"model": "base_rate_prior"}}
