from __future__ import annotations
import httpx
import time
import json
import hashlib
from typing import Any, Iterator
from datetime import datetime, timezone
import config

class BzzoiroError(Exception):
    pass

class BzzoiroRateLimitError(BzzoiroError):
    pass

class BzzoiroAPIClient:
    def __init__(self):
        self.base_url = config.BZZOIRO_API
        self.headers = {"Authorization": f"Token {config.BZZOIRO_KEY}"} if config.BZZOIRO_KEY else {}
        self.timeout = config.BZZOIRO_TIMEOUT_SECONDS
        self.max_retries = config.BZZOIRO_MAX_RETRIES
        self._client = httpx.Client(base_url=self.base_url, headers=self.headers, timeout=self.timeout)

    def _request(self, method: str, path: str, params: dict | None = None) -> dict:
        if not config.BZZOIRO_ENABLED:
            return {"error": "BZZOIRO_ENABLED is False"}
        if not config.BZZOIRO_KEY:
            return {"error": "No BZZOIRO_KEY configured"}

        url_path = path.lstrip('/')
        retries = 0
        backoff = 1.0

        while retries <= self.max_retries:
            try:
                response = self._client.request(method, url_path, params=params)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", backoff))
                    time.sleep(retry_after)
                    retries += 1
                    backoff *= 2
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and retries < self.max_retries:
                    time.sleep(backoff)
                    retries += 1
                    backoff *= 2
                    continue
                return {"error": f"HTTP {e.response.status_code}", "message": str(e)}
            except httpx.RequestError as e:
                if retries < self.max_retries:
                    time.sleep(backoff)
                    retries += 1
                    backoff *= 2
                    continue
                return {"error": "RequestError", "message": str(e)}
            except Exception as e:
                return {"error": "Exception", "message": str(e)}

        return {"error": "MaxRetriesExceeded"}

    def _paginate(self, path: str, params: dict | None = None) -> Iterator[dict]:
        params = params or {}
        params["limit"] = params.get("limit", 100)
        params["offset"] = params.get("offset", 0)

        while True:
            data = self._request("GET", path, params)
            if "error" in data:
                yield data
                break

            results = data.get("results", [])
            if not isinstance(results, list):
                break
            
            for item in results:
                yield item
            
            if not data.get("next"):
                break
            
            params["offset"] += params["limit"]

    def search_teams(self, name: str) -> list[dict]:
        return list(self._paginate("v2/teams/", {"name": name}))

    def get_event(self, event_id: int) -> dict:
        return self._request("GET", f"v2/events/{event_id}/")

    def search_events(self, home_team_name: str, away_team_name: str = "", date_from: str = "", date_to: str = "") -> list[dict]:
        params = {"team_name": home_team_name}
        if date_from: params["date_from"] = date_from
        if date_to: params["date_to"] = date_to
        
        results = list(self._paginate("v2/events/", params))
        if away_team_name and results and "error" not in results[0]:
            away_lower = away_team_name.lower()
            results = [r for r in results if away_lower in (r.get("away_team") or {}).get("name", "").lower()]
        return results

    def get_event_stats(self, event_id: int) -> dict:
        return self._request("GET", f"v2/events/{event_id}/stats/")

    def get_event_lineups(self, event_id: int) -> dict:
        return self._request("GET", f"v2/events/{event_id}/lineups/")

    def get_event_prediction(self, event_id: int) -> dict:
        return self._request("GET", f"v2/events/{event_id}/prediction/")

client = BzzoiroAPIClient()

# Export functions for backwards compatibility, but use the new client
def search_teams(name: str) -> list[dict]: return client.search_teams(name)
def get_event(event_id: int) -> dict: return client.get_event(event_id)
def search_events(home_team_name: str, away_team_name: str = "", date_from: str = "", date_to: str = "") -> list[dict]: return client.search_events(home_team_name, away_team_name, date_from, date_to)
def get_event_stats(event_id: int) -> dict: return client.get_event_stats(event_id)
def get_event_lineups(event_id: int) -> dict: return client.get_event_lineups(event_id)
def get_event_prediction(event_id: int) -> dict: return client.get_event_prediction(event_id)

def extract_ml_probabilities(prediction: dict) -> dict[str, float] | None:
    try:
        mr = prediction.get("match_result", {})
        if not mr:
            return None
        return {
            "home_win": float(mr["home_win_probability"]),
            "draw": float(mr["draw_probability"]),
            "away_win": float(mr["away_win_probability"])
        }
    except (KeyError, TypeError, ValueError):
        return None

def extract_event_stats_summary(stats: dict) -> dict:
    try:
        teams = stats.get("teams", {})
        if not teams:
            return {}
        home = teams.get("home", {})
        away = teams.get("away", {})
        return {
            "home_xg": float(home.get("expected_goals", 0.0)),
            "away_xg": float(away.get("expected_goals", 0.0)),
            "home_possession": int(home.get("possession_time", 0)),
            "away_possession": int(away.get("possession_time", 0)),
            "home_momentum": float(home.get("momentum_score", 0.0)),
            "away_momentum": float(away.get("momentum_score", 0.0))
        }
    except (KeyError, TypeError, ValueError):
        return {}
