"""
Sportmonks proxy wrapper.

All Sportmonks Football v3 paths go through the arena proxy:
  api.sportmonks.com/v3/football/<path>
  → staging.stair-ai.com/api/v1/data/proxy/sportmonks/v3/football/<path>

Auth: x-api-key: ARENA_KEY  (arena covers the Sportmonks subscription)
"""
from __future__ import annotations
import httpx
from typing import Any
import config

_BASE = f"{config.ARENA_API}/v1/data/proxy/sportmonks/v3/football"
_HEADERS = {"x-api-key": config.ARENA_KEY}


def _get(path: str, params: dict | None = None) -> Any:
    """
    All Sportmonks proxy responses are wrapped in an arena envelope:
      {body, duration, headers, requestId, statusCode, _proxy}
    Real payload lives at envelope["body"]["data"].
    """
    url = f"{_BASE}/{path.lstrip('/')}"
    resp = httpx.get(url, headers=_HEADERS, params=params or {}, timeout=60)
    resp.raise_for_status()
    envelope = resp.json()
    if isinstance(envelope, dict) and "body" in envelope:
        body = envelope["body"]
        if isinstance(body, dict) and "data" in body:
            return body["data"]
        return body
    return envelope


# ── Fixtures ──────────────────────────────────────────────────────────────

def get_season_schedule(season_id: int = config.SEASON_ID) -> list[dict]:
    """All schedule entries (stages/rounds/fixtures) for WC 2026."""
    data = _get(f"schedules/seasons/{season_id}")
    return data if isinstance(data, list) else []


def get_fixtures_by_season(season_id: int = config.SEASON_ID) -> list[dict]:
    """All fixtures for the WC 2026 season."""
    data = _get("fixtures", params={"filter": f"seasonId:{season_id}", "per_page": 200})
    return data if isinstance(data, list) else []


def get_fixture(fixture_id: int) -> dict:
    """
    Full fixture record with: predictions, odds, xGFixture, participants.
    Uses the same includes as the sample notebook.
    The envelope is already peeled by _get(); returns the fixture dict directly.
    """
    data = _get(
        f"fixtures/{fixture_id}",
        params={
            "include": "participants;predictions;odds;xGFixture"
        },
    )
    return data if isinstance(data, dict) else {}


def get_live_fixtures(season_id: int = config.SEASON_ID) -> list[dict]:
    """Fixtures currently in-play (for HT window detection)."""
    data = _get("livescores/inplay", params={"filter": f"seasonId:{season_id}"})
    return data if isinstance(data, list) else []


def get_fixture_predictions(fixture_id: int) -> dict:
    """Sportmonks ML pre-match probability predictions for a fixture."""
    data = _get(f"predictions/probabilities/fixtures/{fixture_id}")
    return data if isinstance(data, dict) else {}


def extract_ml_probabilities(fixture: dict) -> dict[str, float] | None:
    """
    Pull the Sportmonks ML win/draw/win probabilities out of a fixture record.
    Returns {home_win, draw, away_win} or None if unavailable.
    """
    predictions = fixture.get("predictions") or []
    if not predictions:
        return None
    pred = predictions[0] if isinstance(predictions, list) else predictions
    predictions_data = pred.get("predictions", {})
    if not predictions_data:
        return None
    return {
        "home_win": float(predictions_data.get("home_win", 0)) / 100,
        "draw":     float(predictions_data.get("draw", 0)) / 100,
        "away_win": float(predictions_data.get("away_win", 0)) / 100,
    }


def extract_bookmaker_odds(fixture: dict) -> dict[str, float] | None:
    """
    Convert best available 1X2 bookmaker odds to implied probabilities.
    Returns {home_win, draw, away_win} or None if unavailable.
    """
    odds_list = fixture.get("odds") or []
    if not odds_list:
        return None
    # Look for first market that has home/draw/away labels
    for market in odds_list:
        values = market.get("values") or []
        result: dict[str, float] = {}
        for v in values:
            label = str(v.get("label", "")).lower()
            odd = v.get("value")
            if odd and label in ("1", "home", "home win"):
                result["home_win"] = 1.0 / float(odd)
            elif odd and label in ("x", "draw"):
                result["draw"] = 1.0 / float(odd)
            elif odd and label in ("2", "away", "away win"):
                result["away_win"] = 1.0 / float(odd)
        if len(result) == 3:
            # Normalise to remove the bookmaker margin
            total = sum(result.values())
            return {k: v / total for k, v in result.items()}
    return None


def extract_ht_stats(fixture: dict) -> dict:
    """
    Pull half-time match stats: score, xG, shots, possession, cards.
    Returns an empty dict pre-match or if stats unavailable.
    """
    stats = {}
    scores = fixture.get("scores") or []
    for score in scores:
        if score.get("score", {}).get("period") == "HT":
            stats["ht_score"] = score["score"].get("goals", {})
            break

    statistics = fixture.get("statistics") or []
    for stat in statistics:
        type_name = (stat.get("type", {}) or {}).get("name", "")
        value = stat.get("data", {})
        if type_name in (
            "Expected Goals",
            "Ball Possession",
            "Shots on Goal",
            "Shots off Goal",
            "Yellow Cards",
            "Red Cards",
        ):
            stats[type_name] = value
    return stats
