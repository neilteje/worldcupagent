"""Read-only coverage audit for every World Cup group-stage fixture.

Checks the data surfaces the live runner depends on:
  - Sportmonks schedule/fixture participants
  - Supabase country-id resolver
  - BZZOIRO event mapping, prediction, stats, and lineups endpoint availability
  - Polymarket fixture mapping, token IDs, and CLOB midpoints
  - Arena match timing endpoint

No orders, ledgers, or state writes.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

config.BZZOIRO_ENABLED = True


def _safe(label: str, fn, *args, **kwargs) -> tuple[Any, str | None]:
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:
        return None, f"{label}: {type(exc).__name__}: {exc}"


def _participant(fixture: dict, location: str) -> dict:
    participants = fixture.get("participants") or []
    return next((p for p in participants if (p.get("meta") or {}).get("location") == location), {})


def _is_group_stage(fixture: dict) -> bool:
    stage = str((fixture.get("stage") or {}).get("name") or fixture.get("stage") or "").lower()
    name = str(fixture.get("name") or "")
    # Current arena schedule has exactly the 72 group-stage fixtures. Keep this
    # tolerant for future includes where the stage field is populated.
    return not stage or "group" in stage or bool(name)


def _ok_payload(payload: Any) -> bool:
    return bool(payload) and not (isinstance(payload, dict) and payload.get("error"))


def main() -> int:
    from data import bzzoiro, bzzoiro_mapper, polymarket, sportmonks, supabase_client
    from data.team_codes import fifa_code
    from live.arena_client import ArenaClient
    from live.runner import flatten_schedule

    schedule, schedule_err = _safe("schedule", sportmonks.get_season_schedule)
    fixtures = [f for f in flatten_schedule(schedule or []) if _is_group_stage(f)]
    reader = ArenaClient()
    mappings, mappings_err = _safe("polymarket mappings", polymarket.get_all_mappings)
    mapping_by_fixture = {
        str(m.get("sportmonks_fixture_id")): m
        for m in (mappings or [])
        if m.get("sportmonks_fixture_id") is not None
    }

    now = datetime.now(timezone.utc)
    rows: list[dict] = []
    countries: dict[str, dict] = {}
    failures: list[dict] = []

    for i, fixture in enumerate(fixtures, start=1):
        fid = int(fixture["id"])
        detail, detail_err = _safe(f"fixture {fid}", sportmonks.get_fixture, fid)
        fx = detail if isinstance(detail, dict) and detail else fixture

        home = _participant(fx, "home")
        away = _participant(fx, "away")
        home_name = home.get("name") or fixture.get("home_country") or "HOME"
        away_name = away.get("name") or fixture.get("away_country") or "AWAY"
        home_code = fifa_code(home.get("short_code"), "HOME")
        away_code = fifa_code(away.get("short_code"), "AWAY")
        kickoff = str(fx.get("starting_at") or fixture.get("starting_at") or "")
        match_date = kickoff[:10]
        kickoff_dt = None
        try:
            kickoff_dt = datetime.fromisoformat(kickoff.replace("Z", "")).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        is_future = bool(kickoff_dt and kickoff_dt > now)

        home_country_id = supabase_client.resolve_country_id(home_name)
        away_country_id = supabase_client.resolve_country_id(away_name)
        countries[home_name] = {
            "code": home_code,
            "country_id": home_country_id,
            "known_missing": supabase_client.known_missing_country(home_name),
        }
        countries[away_name] = {
            "code": away_code,
            "country_id": away_country_id,
            "known_missing": supabase_client.known_missing_country(away_name),
        }

        event_id, bz_map_err = _safe(
            f"bzzoiro map {fid}",
            bzzoiro_mapper.get_bzzoiro_event_id,
            home_name,
            away_name,
            match_date,
        )
        bz_event = bz_prediction = bz_stats = bz_lineups = None
        bz_errors = [e for e in [bz_map_err] if e]
        if event_id:
            bz_event, err = _safe(f"bzzoiro event {event_id}", bzzoiro.get_event, int(event_id))
            if err:
                bz_errors.append(err)
            bz_prediction, err = _safe(f"bzzoiro prediction {event_id}", bzzoiro.get_event_prediction, int(event_id))
            if err:
                bz_errors.append(err)
            bz_stats, err = _safe(f"bzzoiro stats {event_id}", bzzoiro.get_event_stats, int(event_id))
            if err:
                bz_errors.append(err)
            bz_lineups, err = _safe(f"bzzoiro lineups {event_id}", bzzoiro.get_event_lineups, int(event_id))
            if err:
                bz_errors.append(err)

        bz_probs = bzzoiro.extract_ml_probabilities(bz_prediction or {})
        bz_lineup_status = (bz_lineups or {}).get("lineup_status") if isinstance(bz_lineups, dict) else None
        bz_has_lineups = bool(isinstance(bz_lineups, dict) and bz_lineups.get("lineups"))

        mapping = mapping_by_fixture.get(str(fid)) or {}
        pm_tokens = {
            "home": mapping.get("polymarket_home_token_yes"),
            "draw": mapping.get("polymarket_draw_token_yes"),
            "away": mapping.get("polymarket_away_token_yes"),
            "slug": mapping.get("polymarket_event_slug", ""),
        }
        pm_three_way, pm_err = _safe("polymarket three-way", polymarket.get_three_way_market_probs, fid, pm_tokens)

        arena_match, arena_err = _safe("arena match", reader.match, fid)

        row = {
            "fixture_id": fid,
            "fixture_name": fixture.get("name") or f"{home_name} vs {away_name}",
            "kickoff": kickoff,
            "is_future": is_future,
            "home": {"name": home_name, "code": home_code, "country_id": home_country_id},
            "away": {"name": away_name, "code": away_code, "country_id": away_country_id},
            "sportmonks_fixture_ok": detail_err is None and bool(detail),
            "supabase_resolver_ok": bool(home_country_id and away_country_id),
            "supabase_known_coverage_gap": bool(
                (not home_country_id and supabase_client.known_missing_country(home_name))
                or (not away_country_id and supabase_client.known_missing_country(away_name))
            ),
            "bzzoiro": {
                "event_id": event_id,
                "event_ok": _ok_payload(bz_event),
                "prediction_ok": bool(bz_probs),
                "stats_ok": _ok_payload(bz_stats),
                "lineups_ok": _ok_payload(bz_lineups),
                "lineup_status": bz_lineup_status,
                "has_lineups": bz_has_lineups,
                "errors": bz_errors,
            },
            "polymarket": {
                "mapping_ok": bool(mapping),
                "slug": mapping.get("polymarket_event_slug"),
                "tokens_ok": all(pm_tokens.get(k) for k in ("home", "draw", "away")),
                "midpoints_ok": bool(isinstance(pm_three_way, dict) and pm_three_way.get("complete")),
                "reason": (pm_three_way or {}).get("reason") if isinstance(pm_three_way, dict) else pm_err,
                "raw_midpoints": (pm_three_way or {}).get("raw_midpoints") if isinstance(pm_three_way, dict) else None,
            },
            "arena": {
                "match_ok": _ok_payload(arena_match),
                "current_window": (arena_match or {}).get("current_window") if isinstance(arena_match, dict) else None,
                "server_ts_utc": (arena_match or {}).get("server_ts_utc") if isinstance(arena_match, dict) else None,
                "error": arena_err,
            },
            "errors": [e for e in [detail_err, pm_err, arena_err] if e],
        }

        row["ready_for_future_live_cycle"] = bool(
            row["is_future"]
            and row["sportmonks_fixture_ok"]
            and row["bzzoiro"]["event_ok"]
            and row["bzzoiro"]["prediction_ok"]
            and row["polymarket"]["mapping_ok"]
            and row["polymarket"]["tokens_ok"]
            and row["polymarket"]["midpoints_ok"]
            and row["arena"]["match_ok"]
        )
        row["complete_historical_coverage"] = bool(
            row["sportmonks_fixture_ok"]
            and row["bzzoiro"]["event_ok"]
            and row["bzzoiro"]["prediction_ok"]
            and row["polymarket"]["mapping_ok"]
            and row["polymarket"]["tokens_ok"]
            and row["polymarket"]["midpoints_ok"]
            and row["arena"]["match_ok"]
        )
        row["past_market_expired"] = bool(
            not row["is_future"]
            and row["polymarket"]["mapping_ok"]
            and row["polymarket"]["tokens_ok"]
            and not row["polymarket"]["midpoints_ok"]
        )
        rows.append(row)

        if row["is_future"] and not row["ready_for_future_live_cycle"]:
            failures.append(row)

        print(
            f"[{i:02d}/{len(fixtures)}] {row['fixture_name']}: "
            f"SM={'ok' if row['sportmonks_fixture_ok'] else 'MISS'} "
            f"SB={'ok' if row['supabase_resolver_ok'] else 'gap' if row['supabase_known_coverage_gap'] else 'MISS'} "
            f"BZ={'ok' if row['bzzoiro']['prediction_ok'] else 'MISS'} "
            f"PM={'ok' if row['polymarket']['midpoints_ok'] else row['polymarket']['reason']} "
            f"Arena={'ok' if row['arena']['match_ok'] else 'MISS'} "
            f"{'future-ready' if row['ready_for_future_live_cycle'] else 'past-expired' if row['past_market_expired'] else ''}"
        )

    unresolved_countries = {
        name: info for name, info in sorted(countries.items())
        if not info["country_id"] and not info["known_missing"]
    }
    known_gaps = {
        name: info for name, info in sorted(countries.items())
        if not info["country_id"] and info["known_missing"]
    }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fixture_count": len(rows),
        "country_count": len(countries),
        "schedule_error": schedule_err,
        "mappings_error": mappings_err,
        "future_fixture_count": sum(1 for r in rows if r["is_future"]),
        "future_ready_count": sum(1 for r in rows if r["ready_for_future_live_cycle"]),
        "complete_historical_coverage_count": sum(1 for r in rows if r["complete_historical_coverage"]),
        "past_market_expired_count": sum(1 for r in rows if r["past_market_expired"]),
        "sportmonks_ok": sum(1 for r in rows if r["sportmonks_fixture_ok"]),
        "supabase_resolver_ok": sum(1 for r in rows if r["supabase_resolver_ok"]),
        "bzzoiro_event_ok": sum(1 for r in rows if r["bzzoiro"]["event_ok"]),
        "bzzoiro_prediction_ok": sum(1 for r in rows if r["bzzoiro"]["prediction_ok"]),
        "bzzoiro_lineups_ok": sum(1 for r in rows if r["bzzoiro"]["lineups_ok"]),
        "polymarket_mapping_ok": sum(1 for r in rows if r["polymarket"]["mapping_ok"]),
        "polymarket_midpoints_ok": sum(1 for r in rows if r["polymarket"]["midpoints_ok"]),
        "arena_match_ok": sum(1 for r in rows if r["arena"]["match_ok"]),
        "unresolved_countries": unresolved_countries,
        "known_supabase_coverage_gaps": known_gaps,
        "failures": [
            {
                "fixture_id": r["fixture_id"],
                "fixture_name": r["fixture_name"],
                "kickoff": r["kickoff"],
                "future_ready": r["ready_for_future_live_cycle"],
                "supabase_ok": r["supabase_resolver_ok"],
                "bzzoiro": r["bzzoiro"],
                "polymarket": r["polymarket"],
                "arena": r["arena"],
                "errors": r["errors"],
            }
            for r in failures
        ],
        "fixtures": rows,
    }

    out_dir = ROOT / "storage" / "reviews"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "worldcup_group_stage_coverage.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n=== summary ===")
    for key in (
        "fixture_count",
        "future_fixture_count",
        "future_ready_count",
        "complete_historical_coverage_count",
        "past_market_expired_count",
        "sportmonks_ok",
        "supabase_resolver_ok",
        "bzzoiro_event_ok",
        "bzzoiro_prediction_ok",
        "bzzoiro_lineups_ok",
        "polymarket_mapping_ok",
        "polymarket_midpoints_ok",
        "arena_match_ok",
    ):
        print(f"{key}: {summary[key]}")
    print(f"unresolved_countries: {len(unresolved_countries)}")
    print(f"known_supabase_coverage_gaps: {len(known_gaps)}")
    print(f"failures: {len(failures)}")
    print(f"report: {out_path}")

    return 0 if not failures and not unresolved_countries else 1


if __name__ == "__main__":
    raise SystemExit(main())
