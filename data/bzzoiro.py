"""
BZZOIRO Football API client.

Fetches data from sports.bzzoiro.com endpoints, specifically:
- /api/v2/events/
- /api/v2/events/{id}/stats/
- /api/v2/events/{id}/lineups/
- /api/v2/predictions/

Fails soft and degrades gracefully.
"""
from __future__ import annotations
import httpx
from typing import Any
import config

_BASE = config.BZZOIRO_API
_HEADERS = {"Authorization": f"Token {config.BZZOIRO_KEY}"} if config.BZZOIRO_KEY else {}

def _get(path: str, params: dict | None = None) -> Any:
    if not config.BZZOIRO_KEY:
        # Fallback if no key is configured, just pretend it's empty
        return {}
    url = f"{_BASE}/{path.lstrip('/')}"
    try:
        resp = httpx.get(url, headers=_HEADERS, params=params or {}, timeout=15.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"  [bzzoiro] GET {path} failed: {exc!r}")
        return {}

def search_teams(name: str) -> list[dict]:
    """Search for a team by name to get its BZZOIRO ID."""
    data = _get("v2/teams/", params={"name": name, "limit": 10})
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data if isinstance(data, list) else []

def get_event(event_id: int) -> dict:
    """Get full event detail."""
    data = _get(f"v2/events/{event_id}/")
    return data if isinstance(data, dict) else {}

def search_events(home_team_name: str, away_team_name: str = "", date_from: str = "", date_to: str = "") -> list[dict]:
    """Find events matching team names."""
    params = {"team_name": home_team_name}
    if date_from: params["date_from"] = date_from
    if date_to: params["date_to"] = date_to
    
    data = _get("v2/events/", params=params)
    results = data.get("results", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    
    if away_team_name:
        away_lower = away_team_name.lower()
        results = [r for r in results if away_lower in (r.get("away_team") or {}).get("name", "").lower()]
    return results

def get_event_stats(event_id: int) -> dict:
    """Get event stats, including xG, momentum, and shots."""
    data = _get(f"v2/events/{event_id}/stats/")
    return data if isinstance(data, dict) else {}

def get_event_lineups(event_id: int) -> dict:
    """Get confirmed or predicted lineups for an event."""
    data = _get(f"v2/events/{event_id}/lineups/")
    return data if isinstance(data, dict) else {}

def get_event_prediction(event_id: int) -> dict:
    """Get CatBoost ML prediction for an event."""
    data = _get(f"v2/events/{event_id}/prediction/")
    return data if isinstance(data, dict) else {}

def extract_ml_probabilities(prediction: dict) -> dict[str, float] | None:
    """
    Extract ML win/draw/win probabilities from a BZZOIRO PredictionV2Schema response.
    Returns {home_win, draw, away_win} or None.
    """
    if not prediction:
        return None
    match_result = prediction.get("match_result", {})
    if not match_result:
        return None
        
    home = match_result.get("home_win_probability")
    draw = match_result.get("draw_probability")
    away = match_result.get("away_win_probability")
    
    if home is not None and draw is not None and away is not None:
        # BZZOIRO probabilities are usually 0-1
        try:
            return {
                "home_win": float(home),
                "draw": float(draw),
                "away_win": float(away)
            }
        except (ValueError, TypeError):
            pass
    return None

def extract_event_stats_summary(stats: dict) -> dict:
    """
    Extract a simplified summary of BZZOIRO stats for the council/deterministic engine.
    """
    if not stats:
        return {}
    
    # We mainly care about xG, momentum, possession, shots
    summary = {}
    teams_stats = stats.get("teams", {})
    home_stats = teams_stats.get("home", {})
    away_stats = teams_stats.get("away", {})
    
    if home_stats and away_stats:
        summary["home_xg"] = home_stats.get("expected_goals", 0.0)
        summary["away_xg"] = away_stats.get("expected_goals", 0.0)
        summary["home_possession"] = home_stats.get("possession_time", 0)
        summary["away_possession"] = away_stats.get("possession_time", 0)
        summary["home_momentum"] = home_stats.get("momentum_score", 0)
        summary["away_momentum"] = away_stats.get("momentum_score", 0)
        
    # Maybe add per-minute xG summary if available
    return summary
