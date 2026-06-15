from __future__ import annotations
import httpx
import time
import json
import hashlib
from typing import Any, Iterator
from datetime import datetime, timezone
import config

def _team_name(value) -> str:
    """`home_team`/`away_team` come back as plain strings in the v2 events feed,
    but some embeds nest `{name: ...}`. Accept either shape."""
    if isinstance(value, dict):
        return value.get("name", "") or value.get("short_name", "") or ""
    return value or ""


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
        if away_team_name and results and isinstance(results[0], dict) and "error" not in results[0]:
            from data.bzzoiro_mapper import canonicalize_team_name

            away_norm = canonicalize_team_name(away_team_name)
            results = [
                r for r in results
                if away_norm in canonicalize_team_name(_team_name(r.get("away_team")))
                or canonicalize_team_name(_team_name(r.get("away_team"))) in away_norm
            ]
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

def _as_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_ml_probabilities(prediction: dict) -> dict[str, float] | None:
    """Pull the 1X2 distribution from a v2 prediction payload.

    Real schema: ``markets.match_result.{prob_home, prob_draw, prob_away}`` —
    values are PERCENTAGES (e.g. 43.1) so we renormalise to sum 1.0. Falls back
    to the legacy flat ``match_result.*_probability`` shape if present.
    """
    if not isinstance(prediction, dict):
        return None
    mr = (prediction.get("markets") or {}).get("match_result") or {}
    home = _as_float(mr.get("prob_home"))
    draw = _as_float(mr.get("prob_draw"))
    away = _as_float(mr.get("prob_away"))
    if home is None or draw is None or away is None:
        legacy = prediction.get("match_result") or {}
        home = _as_float(legacy.get("home_win_probability"))
        draw = _as_float(legacy.get("draw_probability"))
        away = _as_float(legacy.get("away_win_probability"))
    if home is None or draw is None or away is None:
        return None
    total = home + draw + away
    if total <= 0:
        return None
    return {
        "home_win": round(home / total, 4),
        "draw": round(draw / total, 4),
        "away_win": round(away / total, 4),
    }


def extract_prediction_summary(prediction: dict) -> dict:
    """The genuinely useful structured fields beyond 1X2: expected goals,
    over/under, BTTS, the model's own recommendation and confidence/version."""
    if not isinstance(prediction, dict):
        return {}
    markets = prediction.get("markets") or {}
    xg = markets.get("expected_goals") or {}
    ou = markets.get("over_under") or {}
    btts = markets.get("btts") or {}
    score = markets.get("score") or {}
    recs = prediction.get("recommendations") or {}
    model = prediction.get("model") or {}
    out = {
        "expected_goals": {"home": _as_float(xg.get("home")), "away": _as_float(xg.get("away"))},
        "over_under": {k: _as_float(v) for k, v in ou.items()},
        "btts_yes": _as_float(btts.get("prob_yes")),
        "most_likely_score": score.get("most_likely"),
        "model_favorite": recs.get("favorite"),
        "model_favorite_prob": _as_float(recs.get("favorite_prob")),
        "model_confidence": _as_float(model.get("confidence")),
        "model_version": model.get("version"),
    }
    return {k: v for k, v in out.items() if v not in (None, {}, [])}


def extract_event_stats_summary(stats: dict) -> dict:
    """Real schema: ``stats.stats.{home,away}`` carries per-side metrics (xG and,
    for in-play/finished matches, possession/shots). ``stats.momentum`` is a
    time series; we collapse its final reading into a per-side momentum score.
    Upcoming matches legitimately return only xG (often null)."""
    if not isinstance(stats, dict):
        return {}
    inner = stats.get("stats") or {}
    if not inner:
        return {}
    home = inner.get("home") or {}
    away = inner.get("away") or {}

    def side(d: dict, *keys) -> float:
        for k in keys:
            v = _as_float(d.get(k))
            if v is not None:
                return v
        return 0.0

    out = {
        "home_xg": side(home, "xg", "expected_goals"),
        "away_xg": side(away, "xg", "expected_goals"),
        "home_possession": side(home, "possession", "possession_time", "ball_possession"),
        "away_possession": side(away, "possession", "possession_time", "ball_possession"),
        "home_shots": side(home, "shots", "shots_total"),
        "away_shots": side(away, "shots", "shots_total"),
    }

    momentum = stats.get("momentum") or []
    home_m = away_m = 0.0
    if isinstance(momentum, list) and momentum and isinstance(momentum[-1], dict):
        last = momentum[-1]
        home_m = _as_float(last.get("home")) or _as_float(last.get("home_momentum")) or 0.0
        away_m = _as_float(last.get("away")) or _as_float(last.get("away_momentum")) or 0.0
    out["home_momentum"] = home_m
    out["away_momentum"] = away_m
    return out
