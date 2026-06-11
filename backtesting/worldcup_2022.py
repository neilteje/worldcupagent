from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
import json
import math
import re
import urllib.request

from models.calibration import OUTCOMES, normalize_probs
from models.elo import EloConfig, build_timeline
from models.lineup_delta import evaluate_lineup_delta


STATSBOMB_RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CHECKBESTODDS_2022_URL = "https://checkbestodds.com/football-odds/archive-world-cup-2022/"
WC2022_COMPETITION_ID = 43
WC2022_SEASON_ID = 106
WC2018_SEASON_ID = 3

TEAM_ALIASES = {
    "ir iran": "iran",
    "iran": "iran",
    "usa": "united states",
    "u.s.a.": "united states",
    "united states": "united states",
    "south korea": "korea republic",
    "korea republic": "korea republic",
    "korea": "korea republic",
    "saudi arabia": "saudi arabia",
    "saudi": "saudi arabia",
}

CONTINENT_MAP = {
    "qatar": "AFC", "ecuador": "CONMEBOL", "senegal": "CAF", "netherlands": "UEFA",
    "england": "UEFA", "iran": "AFC", "united states": "CONCACAF", "wales": "UEFA",
    "argentina": "CONMEBOL", "saudi arabia": "AFC", "mexico": "CONCACAF", "poland": "UEFA",
    "france": "UEFA", "australia": "AFC", "denmark": "UEFA", "tunisia": "CAF",
    "spain": "UEFA", "costa rica": "CONCACAF", "germany": "UEFA", "japan": "AFC",
    "belgium": "UEFA", "canada": "CONCACAF", "morocco": "CAF", "croatia": "UEFA",
    "brazil": "CONMEBOL", "serbia": "UEFA", "switzerland": "UEFA", "cameroon": "CAF",
    "portugal": "UEFA", "ghana": "CAF", "uruguay": "CONMEBOL", "korea republic": "AFC"
}

TALENT_SCORE_MAP = {
    # 5: elite depth, 4: very good, 3: solid, 2: weak depth, 1: domestic only
    "brazil": 5, "france": 5, "england": 5, "portugal": 5, "spain": 5, "germany": 5, "argentina": 5,
    "netherlands": 4, "belgium": 4, "croatia": 4, "uruguay": 4, "senegal": 3, "denmark": 3, "switzerland": 3,
    "serbia": 3, "morocco": 3, "united states": 3, "mexico": 3, "poland": 3, "wales": 2, "japan": 3,
    "korea republic": 3, "ecuador": 2, "cameroon": 2, "ghana": 2, "canada": 2, "costa rica": 1,
    "tunisia": 2, "saudi arabia": 1, "iran": 2, "australia": 2, "qatar": 1
}


@dataclass
class HistoricalFixture:
    fixture_code: str
    match_id: int
    kickoff_utc: str
    home_team: str
    away_team: str
    stage: str
    match_week: int
    home_score: int
    away_score: int
    result: str
    market: dict[str, float]
    odds: dict[str, float]
    odds_quality: dict
    priors: dict[str, float]
    sportmonks: dict[str, float]
    lineup_payload: dict
    lineup: dict
    pre_state: dict
    post_state: dict = field(default_factory=dict)


@dataclass
class TeamState:
    team: str
    base_rating: float = 0.0
    matches: int = 0
    goals_for: float = 0.0
    goals_against: float = 0.0
    xg_for: float = 0.0
    xg_against: float = 0.0
    shots_for: float = 0.0
    shots_against: float = 0.0
    last_starters: list[dict] = field(default_factory=list)
    last_match_time: str | None = None

    @property
    def live_rating(self) -> float:
        if self.matches <= 0:
            return self.base_rating
        goal_diff = (self.goals_for - self.goals_against) / max(self.matches, 1)
        xg_diff = (self.xg_for - self.xg_against) / max(self.matches, 1)
        shot_diff = (self.shots_for - self.shots_against) / max(self.matches, 1)
        form = 0.26 * goal_diff + 0.34 * xg_diff + 0.012 * shot_diff
        return self.base_rating + max(-0.85, min(0.85, form))


def build_worldcup_2022_history(cache_dir: Path, *, limit: int | None = None) -> list[HistoricalFixture]:
    """
    Build a chronological no-future-leakage 2022 World Cup backtest set.

    Sources:
    - StatsBomb open data for real matches, lineups, and event-derived xG/shots.
    - CheckBestOdds archive for historical 1X2 odds.
    - StatsBomb 2018 match results for pre-tournament priors.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    matches = sorted(_load_statsbomb_matches(cache_dir, WC2022_SEASON_ID), key=_match_sort_key)
    odds_by_key = _load_checkbestodds(cache_dir)
    base_ratings = _load_prior_ratings_2018(cache_dir)
    elo_timeline = _build_elo_timeline(cache_dir, matches)
    states: dict[str, TeamState] = {}
    fixtures: list[HistoricalFixture] = []

    for match in matches:
        home = match["home_team"]["home_team_name"]
        away = match["away_team"]["away_team_name"]
        home_key, away_key = team_key(home), team_key(away)
        home_state = states.setdefault(home_key, TeamState(home, base_ratings.get(home_key, 0.0)))
        away_state = states.setdefault(away_key, TeamState(away, base_ratings.get(away_key, 0.0)))

        odds = _match_odds(odds_by_key, home, away)
        odds_quality = odds_reference_quality(odds) if odds else {"tradable": False, "flags": ["odds_missing"], "implied_sum": 0.0}
        market = implied_probs_from_odds(odds) if odds else _rating_probs(home_state.live_rating, away_state.live_rating)
        priors = _rating_probs(home_state.base_rating, away_state.base_rating)
        sportmonks = _rating_probs(home_state.live_rating, away_state.live_rating)
        lineups = _load_lineups(cache_dir, int(match["match_id"]))
        lineup_payload = _lineup_payload(lineups, home, away, home_state, away_state)
        lineup = evaluate_lineup_delta(**lineup_payload)
        events = _load_events(cache_dir, int(match["match_id"]))
        post = _event_summary(events, home, away)
        result = _result_from_scores(int(match["home_score"]), int(match["away_score"]))

        match_time = _kickoff_iso(match)
        match_dt = datetime.fromisoformat(match_time)
        home_rest = 72.0
        if home_state.last_match_time:
            home_rest = (match_dt - datetime.fromisoformat(home_state.last_match_time)).total_seconds() / 3600.0
        away_rest = 72.0
        if away_state.last_match_time:
            away_rest = (match_dt - datetime.fromisoformat(away_state.last_match_time)).total_seconds() / 3600.0

        home_pre = _with_elo(_state_payload(home_state), elo_timeline, int(match["match_id"]), "home")
        home_pre.update({
            "continent": CONTINENT_MAP.get(home_key),
            "talent_score": TALENT_SCORE_MAP.get(home_key, 2.5),
            "rest_hours": home_rest,
        })
        away_pre = _with_elo(_state_payload(away_state), elo_timeline, int(match["match_id"]), "away")
        away_pre.update({
            "continent": CONTINENT_MAP.get(away_key),
            "talent_score": TALENT_SCORE_MAP.get(away_key, 2.5),
            "rest_hours": away_rest,
        })

        fixtures.append(
            HistoricalFixture(
                fixture_code=f"WC2022-{match['match_id']}",
                match_id=int(match["match_id"]),
                kickoff_utc=match_time,
                home_team=home,
                away_team=away,
                stage=match.get("competition_stage", {}).get("name", ""),
                match_week=int(match.get("match_week") or 0),
                home_score=int(match["home_score"]),
                away_score=int(match["away_score"]),
                result=result,
                market=market,
                odds=odds or {},
                odds_quality=odds_quality,
                priors=priors,
                sportmonks=sportmonks,
                lineup_payload=lineup_payload,
                lineup=lineup,
                pre_state={
                    "home": home_pre,
                    "away": away_pre,
                    "odds_source": "checkbestodds_archive" if odds else "rating_fallback_no_odds",
                    "odds_quality": odds_quality,
                },
                post_state=post,
            )
        )

        _update_state(home_state, post["home"])
        _update_state(away_state, post["away"])
        home_state.last_starters = lineup_payload["confirmed_home"]
        away_state.last_starters = lineup_payload["confirmed_away"]
        home_state.last_match_time = match_time
        away_state.last_match_time = match_time

        if limit and len(fixtures) >= limit:
            break
    return fixtures


def parse_checkbestodds_html(text: str) -> list[dict]:
    pattern = re.compile(
        r'<tr>\s*<td class="l2 match">\s*'
        r'<span ts="(?P<ts>\d+)" class="time hM">(?P<time>[^<]+)</span>\s*'
        r'<a href="(?P<href>[^"]+)">\s*(?P<home>[^<]+?)\s*-\s*(?P<away>[^<]+?)</a></td>\s*'
        r'<td class="r">\s*<b class="[^"]*">(?P<h>[0-9.]+)</b></td>\s*'
        r'<td class="r">\s*<b class="[^"]*">(?P<d>[0-9.]+)</b></td>\s*'
        r'<td class="r">\s*<b class="[^"]*">(?P<a>[0-9.]+)</b></td></tr>',
        re.S,
    )
    rows = []
    for match in pattern.finditer(text):
        gd = {k: unescape(v).strip() for k, v in match.groupdict().items()}
        rows.append(
            {
                "timestamp": int(gd["ts"]),
                "home_team": " ".join(gd["home"].split()),
                "away_team": " ".join(gd["away"].split()),
                "home_odds": float(gd["h"]),
                "draw_odds": float(gd["d"]),
                "away_odds": float(gd["a"]),
                "href": gd["href"],
            }
        )
    return rows


def implied_probs_from_odds(odds: dict[str, float]) -> dict[str, float]:
    raw = {
        "home": 1.0 / max(float(odds["home"]), 1.01),
        "draw": 1.0 / max(float(odds["draw"]), 1.01),
        "away": 1.0 / max(float(odds["away"]), 1.01),
    }
    return normalize_probs(raw)


def odds_reference_quality(odds: dict[str, float]) -> dict:
    """Flag odds rows that are useful as context but too distorted for trade simulation."""
    values = [float(odds[k]) for k in OUTCOMES]
    implied_sum = sum(1.0 / max(v, 1.01) for v in values)
    flags: list[str] = []
    if implied_sum < 0.9:
        flags.append("archive_underround_too_large")
    if implied_sum > 1.18:
        flags.append("archive_overround_too_large")
    if max(values) >= 80.0:
        flags.append("extreme_best_price_outlier")
    return {
        "tradable": not flags,
        "flags": flags,
        "implied_sum": round(implied_sum, 4),
    }


def team_key(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", name.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return TEAM_ALIASES.get(cleaned, cleaned)


def _load_statsbomb_matches(cache_dir: Path, season_id: int) -> list[dict]:
    path = cache_dir / f"statsbomb_matches_43_{season_id}.json"
    return _fetch_json(path, f"{STATSBOMB_RAW}/matches/{WC2022_COMPETITION_ID}/{season_id}.json")


def _load_lineups(cache_dir: Path, match_id: int) -> list[dict]:
    return _fetch_json(cache_dir / "lineups" / f"{match_id}.json", f"{STATSBOMB_RAW}/lineups/{match_id}.json")


def _load_events(cache_dir: Path, match_id: int) -> list[dict]:
    return _fetch_json(cache_dir / "events" / f"{match_id}.json", f"{STATSBOMB_RAW}/events/{match_id}.json")


def _load_checkbestodds(cache_dir: Path) -> dict[tuple[str, str], dict[str, float]]:
    path = cache_dir / "checkbestodds_worldcup_2022.html"
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        req = urllib.request.Request(CHECKBESTODDS_2022_URL, headers={"User-Agent": "Mozilla/5.0"})
        text = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
        path.write_text(text, encoding="utf-8")
    rows = parse_checkbestodds_html(text)
    return {
        (team_key(row["home_team"]), team_key(row["away_team"])): {
            "home": row["home_odds"],
            "draw": row["draw_odds"],
            "away": row["away_odds"],
        }
        for row in rows
    }


def _elo_rows(matches: list[dict]) -> list[dict]:
    """Minimal date-sorted match rows for the Elo timeline."""
    rows = []
    for m in sorted(matches, key=_match_sort_key):
        rows.append({
            "match_id": int(m["match_id"]),
            "home_key": team_key(m["home_team"]["home_team_name"]),
            "away_key": team_key(m["away_team"]["away_team_name"]),
            "home_score": int(m["home_score"]),
            "away_score": int(m["away_score"]),
        })
    return rows


def _build_elo_timeline(cache_dir: Path, matches_2022: list[dict]) -> dict:
    """Pre-match Elo (scaled to model rating units) for each 2022 fixture.

    Seeded by the full 2018 World Cup played in date order, then continued
    through 2022. Ratings are read before each match and updated only after, so
    there is no future leakage.
    """
    rows = _elo_rows(_load_statsbomb_matches(cache_dir, WC2018_SEASON_ID)) + _elo_rows(matches_2022)
    return build_timeline(rows, cfg=EloConfig())


def _with_elo(payload: dict, timeline: dict, match_id: int, side: str) -> dict:
    entry = timeline.get(match_id) or {}
    payload["elo"] = entry.get(side, 1500.0)
    payload["elo_scaled"] = entry.get(f"{side}_scaled", 0.0)
    return payload


def _load_prior_ratings_2018(cache_dir: Path) -> dict[str, float]:
    ratings: dict[str, float] = {}
    matches = _load_statsbomb_matches(cache_dir, WC2018_SEASON_ID)
    for match in matches:
        home, away = team_key(match["home_team"]["home_team_name"]), team_key(match["away_team"]["away_team_name"])
        ratings.setdefault(home, 0.0)
        ratings.setdefault(away, 0.0)
        hg, ag = int(match["home_score"]), int(match["away_score"])
        gd = max(-3, min(3, hg - ag))
        if hg > ag:
            ratings[home] += 0.65
            ratings[away] -= 0.65
        elif ag > hg:
            ratings[away] += 0.65
            ratings[home] -= 0.65
        ratings[home] += 0.12 * gd
        ratings[away] -= 0.12 * gd
    return {team: max(-1.2, min(1.2, rating / 5.0)) for team, rating in ratings.items()}


def _fetch_json(path: Path, url: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=45).read().decode("utf-8"))
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _match_sort_key(match: dict) -> tuple[str, str, int]:
    return (match.get("match_date", ""), match.get("kick_off", ""), int(match.get("match_id") or 0))


def _kickoff_iso(match: dict) -> str:
    raw = f"{match.get('match_date')}T{str(match.get('kick_off') or '00:00:00').split('.')[0]}"
    dt = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _match_odds(odds_by_key: dict[tuple[str, str], dict[str, float]], home: str, away: str) -> dict[str, float] | None:
    direct = odds_by_key.get((team_key(home), team_key(away)))
    if direct:
        return direct
    flipped = odds_by_key.get((team_key(away), team_key(home)))
    if not flipped:
        return None
    return {"home": flipped["away"], "draw": flipped["draw"], "away": flipped["home"]}


def _rating_probs(home_rating: float, away_rating: float) -> dict[str, float]:
    diff = max(-2.5, min(2.5, home_rating - away_rating))
    draw = max(0.18, min(0.34, 0.285 - 0.045 * abs(diff)))
    home_no_draw = 1.0 / (1.0 + math.exp(-1.25 * diff))
    rem = 1.0 - draw
    return normalize_probs({"home": rem * home_no_draw, "draw": draw, "away": rem * (1.0 - home_no_draw)})


def _result_from_scores(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home"
    if away_score > home_score:
        return "away"
    return "draw"


def _lineup_payload(lineups: list[dict], home: str, away: str, home_state: TeamState, away_state: TeamState) -> dict:
    by_team = {team_key(row.get("team_name", "")): row.get("lineup") or [] for row in lineups}
    confirmed_home = _starters(by_team.get(team_key(home), []))
    confirmed_away = _starters(by_team.get(team_key(away), []))
    return {
        "expected_home": home_state.last_starters,
        "confirmed_home": confirmed_home,
        "expected_away": away_state.last_starters,
        "confirmed_away": confirmed_away,
        "expected_formations": {},
        "confirmed_formations": {},
    }


def _starters(players: list[dict]) -> list[dict]:
    starters = []
    for player in players:
        start_positions = [
            p for p in (player.get("positions") or [])
            if p.get("start_reason") == "Starting XI" or p.get("from") == "00:00"
        ]
        if not start_positions:
            continue
        pos = start_positions[0]
        starters.append(
            {
                "id": player.get("player_id"),
                "name": player.get("player_name"),
                "position": pos.get("position"),
                "starter": True,
                "expected_starter": True,
            }
        )
    return starters


def _event_summary(events: list[dict], home: str, away: str) -> dict:
    summary = {
        "home": {"goals_for": 0, "goals_against": 0, "xg_for": 0.0, "xg_against": 0.0, "shots_for": 0, "shots_against": 0},
        "away": {"goals_for": 0, "goals_against": 0, "xg_for": 0.0, "xg_against": 0.0, "shots_for": 0, "shots_against": 0},
    }
    home_key, away_key = team_key(home), team_key(away)
    for event in events:
        if (event.get("type") or {}).get("name") != "Shot":
            continue
        side = "home" if team_key((event.get("team") or {}).get("name", "")) == home_key else "away"
        other = "away" if side == "home" else "home"
        shot = event.get("shot") or {}
        xg = float(shot.get("statsbomb_xg") or 0.0)
        summary[side]["shots_for"] += 1
        summary[other]["shots_against"] += 1
        summary[side]["xg_for"] += xg
        summary[other]["xg_against"] += xg
        if (shot.get("outcome") or {}).get("name") == "Goal":
            summary[side]["goals_for"] += 1
            summary[other]["goals_against"] += 1
    return summary


def _update_state(state: TeamState, observed: dict) -> None:
    state.matches += 1
    state.goals_for += float(observed.get("goals_for", 0.0))
    state.goals_against += float(observed.get("goals_against", 0.0))
    state.xg_for += float(observed.get("xg_for", 0.0))
    state.xg_against += float(observed.get("xg_against", 0.0))
    state.shots_for += float(observed.get("shots_for", 0.0))
    state.shots_against += float(observed.get("shots_against", 0.0))


def _state_payload(state: TeamState) -> dict:
    return {
        "team": state.team,
        "base_rating": round(state.base_rating, 4),
        "live_rating": round(state.live_rating, 4),
        "matches": state.matches,
        "xg_for": round(state.xg_for, 3),
        "xg_against": round(state.xg_against, 3),
        "goals_for": state.goals_for,
        "goals_against": state.goals_against,
        "known_previous_starters": len(state.last_starters),
    }
