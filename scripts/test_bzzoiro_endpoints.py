"""Live read-only probe of the BZZOIRO football API (spec §13/§14 investigation).

Hits every high-value football v2 endpoint with the configured key, captures the
REAL response shape (many sub-resources have inline, undocumented schemas), and
checks compatibility with the rest of the system: the corrected ML-prediction
extractor, the ``bzzoiro_mapper.map_event`` bridge, the ``ProviderSnapshot``
contract, and the deterministic ``home_state``/``away_state`` keys.

Read-only: GET requests only, NO orders, the only write is a JSON report under
``storage/backtests/``. Run:  python3 scripts/test_bzzoiro_endpoints.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

# Force-enable for this probe run (does not persist).
config.BZZOIRO_ENABLED = True

import httpx
from data import bzzoiro, bzzoiro_mapper
from data.bzzoiro_snapshots import create_bzzoiro_snapshot

API = config.BZZOIRO_API  # https://sports.bzzoiro.com/api
HEADERS = {"Authorization": f"Token {config.BZZOIRO_KEY}"} if config.BZZOIRO_KEY else {}
TIMEOUT = config.BZZOIRO_TIMEOUT_SECONDS

_client = httpx.Client(base_url=API, headers=HEADERS, timeout=TIMEOUT)
REPORT: dict = {"probed_at": datetime.now(timezone.utc).isoformat(), "base": API,
                "key_present": bool(config.BZZOIRO_KEY), "endpoints": {}}


def _trim(obj, depth=0):
    """Shallow preview: top-level keys + first list item, truncated."""
    if isinstance(obj, dict):
        return {k: _trim(v, depth + 1) for k, v in list(obj.items())[:25]} if depth < 2 else f"<dict {list(obj)[:8]}>"
    if isinstance(obj, list):
        return [_trim(obj[0], depth + 1)] + ([f"... +{len(obj) - 1}"] if len(obj) > 1 else []) if obj else []
    if isinstance(obj, str) and len(obj) > 80:
        return obj[:77] + "..."
    return obj


def probe(label: str, path: str, params: dict | None = None) -> dict:
    """One GET. Records HTTP status, top-level keys, trimmed sample."""
    row = {"path": path, "params": params or {}}
    try:
        r = _client.get(path.lstrip("/"), params=params or {})
        row["status"] = r.status_code
        try:
            body = r.json()
            row["keys"] = list(body)[:30] if isinstance(body, dict) else f"list[{len(body)}]"
            row["sample"] = _trim(body)
        except Exception:
            row["keys"] = None
            row["sample"] = r.text[:200]
    except Exception as exc:
        row["status"] = "EXC"
        row["error"] = repr(exc)[:200]
    REPORT["endpoints"][label] = row
    status = row.get("status")
    mark = "OK " if status == 200 else "!! "
    print(f"  {mark}{label:34} {status}  keys={row.get('keys')}")
    return row


def _first_id(label: str):
    row = REPORT["endpoints"].get(label, {})
    body = None
    try:
        r = _client.get(row["path"].lstrip("/"), params=row.get("params") or {})
        body = r.json()
    except Exception:
        return None
    results = body.get("results") if isinstance(body, dict) else body
    if isinstance(results, list) and results and isinstance(results[0], dict):
        return results[0].get("id") or results[0].get("event_id") or results[0].get("team_id")
    return None


def main() -> int:
    if not config.BZZOIRO_KEY:
        print("[probe] no BZZOIRO_KEY in env — aborting (set BZZOIRO_KEY / BZZOIRO_API_KEY)")
        return 1
    print(f"[probe] base={API}  key={'set' if config.BZZOIRO_KEY else 'MISSING'}\n")

    # ── Collection endpoints ─────────────────────────────────────────────
    print("[probe] collections:")
    probe("events.list", "/v2/events/", {"limit": 3})
    probe("leagues.list", "/v2/leagues/", {"limit": 3})
    probe("teams.list", "/v2/teams/", {"limit": 3})
    probe("predictions.list", "/v2/predictions/", {"limit": 3, "upcoming": "true"})
    probe("odds.bookmakers", "/v2/bookmakers/", None)
    probe("worldcup.squads", "/v2/worldcup/squads/", {"limit": 3})
    probe("events.live", "/v2/events/live/", None)

    # ── Resolve concrete ids ─────────────────────────────────────────────
    event_id = _first_id("events.list")
    team_id = _first_id("teams.list")
    league_id = _first_id("leagues.list")
    print(f"\n[probe] resolved event_id={event_id} team_id={team_id} league_id={league_id}\n")

    # ── Event sub-resources (inline schemas captured here) ────────────────
    if event_id:
        print("[probe] event sub-resources:")
        probe("event.detail", f"/v2/events/{event_id}/")
        probe("event.prediction", f"/v2/events/{event_id}/prediction/")
        probe("event.stats", f"/v2/events/{event_id}/stats/")
        probe("event.lineups", f"/v2/events/{event_id}/lineups/")
        probe("event.incidents", f"/v2/events/{event_id}/incidents/")
        probe("event.h2h", f"/v2/events/{event_id}/h2h/")
        probe("event.odds", f"/v2/events/{event_id}/odds/")
        probe("event.odds_comparison", f"/v2/events/{event_id}/odds/comparison/")
        probe("event.polymarket", f"/v2/events/{event_id}/polymarket/")
        probe("event.player_stats", f"/v2/events/{event_id}/player-stats/")

    if team_id:
        print("\n[probe] team sub-resources:")
        probe("team.detail", f"/v2/teams/{team_id}/")
        probe("team.fixtures", f"/v2/teams/{team_id}/fixtures/", {"limit": 5})
        probe("team.squad", f"/v2/teams/{team_id}/squad/")
    if league_id:
        probe("league.standings", f"/v2/leagues/{league_id}/standings/")

    # ── Extractor validation against the REAL prediction payload ──────────
    print("\n[probe] extractor + compatibility checks:")
    checks: dict = {}
    pred_row = REPORT["endpoints"].get("event.prediction", {})
    pred_body = None
    if event_id:
        try:
            pred_body = _client.get(f"v2/events/{event_id}/prediction/").json()
        except Exception:
            pred_body = None
    checks["legacy_extractor"] = bzzoiro.extract_ml_probabilities(pred_body or {})
    checks["corrected_extractor"] = _corrected_ml(pred_body or {})
    print(f"  legacy extract_ml_probabilities -> {checks['legacy_extractor']}")
    print(f"  corrected markets.match_result  -> {checks['corrected_extractor']}")

    # map_event bridge from a Sportmonks-style fixture.
    try:
        events_body = _client.get("v2/events/", params={"limit": 5}).json()
        results = events_body.get("results", events_body if isinstance(events_body, list) else [])
        if results:
            e = results[0]
            home = e.get("home_team") or e.get("home_team_name") or ""
            away = e.get("away_team") or e.get("away_team_name") or ""
            ko = e.get("event_date") or ""
            kickoff = datetime.fromisoformat(ko.replace("Z", "+00:00")) if ko else datetime.now(timezone.utc)
            mapping = bzzoiro_mapper.map_event("SM-TEST-1", str(home), str(away), kickoff, results)
            checks["map_event"] = {"bzzoiro_event_id": mapping.bzzoiro_event_id,
                                   "confidence": mapping.confidence,
                                   "method": mapping.mapping_method,
                                   "home_match": mapping.home_match, "away_match": mapping.away_match}
            print(f"  map_event('{home}' vs '{away}') -> conf={mapping.confidence} id={mapping.bzzoiro_event_id}")
    except Exception as exc:
        checks["map_event"] = {"error": repr(exc)[:150]}

    # ProviderSnapshot contract + home_state mapping demo from a team fixture.
    if team_id:
        try:
            fixtures = _client.get(f"v2/teams/{team_id}/fixtures/", params={"limit": 5}).json()
            rows = fixtures.get("results", fixtures if isinstance(fixtures, list) else [])
            snap = create_bzzoiro_snapshot("team_fixtures", str(team_id), rows, success=bool(rows))
            checks["provider_snapshot"] = {"provider": snap.provider, "hash": snap.payload_hash[:12],
                                           "success": snap.success, "n_rows": len(rows)}
            checks["home_state_from_fixture"] = _home_state_demo(rows)
            print(f"  ProviderSnapshot ok hash={snap.payload_hash[:12]} rows={len(rows)}")
            print(f"  home_state keys from team fixtures -> {list((checks['home_state_from_fixture'] or {}).keys())}")
        except Exception as exc:
            checks["provider_snapshot"] = {"error": repr(exc)[:150]}

    REPORT["checks"] = checks

    # ── Write report ──────────────────────────────────────────────────────
    out = ROOT / "storage" / "backtests" / "bzzoiro_probe_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(REPORT, indent=2, default=str))
    ok = sum(1 for r in REPORT["endpoints"].values() if r.get("status") == 200)
    total = len(REPORT["endpoints"])
    print(f"\n[probe] {ok}/{total} endpoints returned 200. report -> {out}")
    return 0


def _corrected_ml(prediction: dict) -> dict | None:
    """Real prediction path: markets.match_result.prob_home/draw/away, with a
    fallback to the flat ``prob_home_win`` list-schema fields."""
    if not isinstance(prediction, dict):
        return None
    mr = ((prediction.get("markets") or {}).get("match_result")) or {}
    home = mr.get("prob_home")
    draw = mr.get("prob_draw")
    away = mr.get("prob_away")
    if home is None:  # flat schema fallback
        home, draw, away = prediction.get("prob_home_win"), prediction.get("prob_draw"), prediction.get("prob_away_win")
    try:
        h, d, a = float(home), float(draw), float(away)
    except (TypeError, ValueError):
        return None
    tot = h + d + a or 1.0
    return {"home": round(h / tot, 4), "draw": round(d / tot, 4), "away": round(a / tot, 4),
            "sums_to_one": round(h + d + a, 3)}


def _home_state_demo(fixtures: list) -> dict | None:
    """Show how team fixtures map onto deterministic home_state keys (goals form
    + neutral/travel). Demonstrates feeding chronological Elo / rolling form."""
    if not fixtures:
        return None
    gf = ga = n = 0.0
    neutral = None
    travel = None
    for e in fixtures:
        if not isinstance(e, dict):
            continue
        hs, as_ = e.get("home_score"), e.get("away_score")
        if hs is None or as_ is None:
            continue
        gf += hs; ga += as_; n += 1
        neutral = e.get("is_neutral_ground", neutral)
        travel = e.get("travel_distance_km", travel)
    if n == 0:
        return {"matches": 0, "note": "no finished fixtures in sample"}
    return {"matches": int(n), "goals_for": round(gf / n, 3), "goals_against": round(ga / n, 3),
            "neutral": neutral, "travel_distance_km": travel,
            "maps_to": ["goals_for", "goals_against", "neutral", "travel_distance_km"]}


if __name__ == "__main__":
    raise SystemExit(main())
