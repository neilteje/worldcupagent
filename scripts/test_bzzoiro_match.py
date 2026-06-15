"""Test BZZOIRO coverage for a single match.

Read-only: searches BZZOIRO events, then fetches stats, prediction, and lineups
for the mapped event. No arena orders and no ledger writes.

Examples:
    python3 scripts/test_bzzoiro_match.py
    python3 scripts/test_bzzoiro_match.py --home Spain --away "Cabo Verde" --date 2026-06-15
    python3 scripts/test_bzzoiro_match.py --home Portugal --away Nigeria --date 2026-06-15 --raw
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

config.BZZOIRO_ENABLED = True


def _dump(label: str, payload: object, *, raw: bool = False) -> None:
    print(f"\n=== {label} ===")
    if payload is None:
        print("  <none>")
        return

    if isinstance(payload, str):
        print(payload)
        return

    text = json.dumps(payload, indent=2, default=str, ensure_ascii=False)
    if not raw and len(text) > 3000:
        print(text[:3000] + "\n... truncated ...")
    else:
        print(text)


def _mark(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default="Spain")
    ap.add_argument("--away", default="Cabo Verde")
    ap.add_argument("--date", default="2026-06-15")
    ap.add_argument("--raw", action="store_true", help="print full raw BZZOIRO responses")
    args = ap.parse_args()

    from data import bzzoiro, bzzoiro_mapper, fixture_bundle

    print("Testing BZZOIRO match coverage")
    print(f"  fixture: {args.home} vs {args.away}")
    print(f"  date:    {args.date}")
    print(f"  key:     {'set' if config.BZZOIRO_KEY else 'MISSING'}")

    try:
        kickoff = datetime.fromisoformat(args.date.replace("Z", "+00:00"))
    except ValueError:
        print("FAIL: --date must be parseable as an ISO date/datetime")
        return 2

    results = list(bzzoiro.search_events(args.home, args.away, "", ""))
    _dump("search_events(team_name=home)", results, raw=args.raw)
    print(f"  {_mark(bool(results))} search returned {len(results)} result(s)")

    mapped = bzzoiro_mapper.map_event(
        internal_fixture_id="cli",
        home_name=args.home,
        away_name=args.away,
        kickoff=kickoff,
        bzzoiro_search_results=results,
    )
    print("\n=== mapping ===")
    print(json.dumps(mapped.__dict__, indent=2, default=str))
    print(f"  {_mark(mapped.bzzoiro_event_id is not None)} mapped_event_id={mapped.bzzoiro_event_id}")

    if not mapped.bzzoiro_event_id:
        print("\nNo confident BZZOIRO event mapping found.")
        return 1

    event_id = int(mapped.bzzoiro_event_id)
    event = bzzoiro.get_event(event_id)
    stats = bzzoiro.get_event_stats(event_id)
    prediction = bzzoiro.get_event_prediction(event_id)
    lineups = bzzoiro.get_event_lineups(event_id)

    _dump("event", event, raw=args.raw)
    _dump("stats", stats, raw=args.raw)
    _dump("prediction", prediction, raw=args.raw)
    _dump("lineups", lineups, raw=args.raw)

    digest = fixture_bundle.build_bzzoiro_digest(args.home, args.away, args.date) or {}
    _dump("fixture_bundle digest", digest, raw=args.raw)

    checks = {
        "event": bool(event and "error" not in event),
        "stats": bool(stats and "error" not in stats),
        "prediction": bool(prediction and "error" not in prediction),
        "lineups": bool(lineups and "error" not in lineups),
        "ml_prediction": bool(digest.get("ml_prediction")),
        "prediction_summary": bool(digest.get("prediction_summary")),
        "stats_summary": bool(digest.get("stats_summary")),
        "has_lineups": bool(digest.get("has_lineups")),
    }

    print("\n=== summary ===")
    for key, ok in checks.items():
        print(f"  {_mark(ok)} {key}")

    live = sum(1 for ok in checks.values() if ok)
    print(f"\n{live}/{len(checks)} BZZOIRO checks passed.")
    return 0 if mapped.bzzoiro_event_id and checks["ml_prediction"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
