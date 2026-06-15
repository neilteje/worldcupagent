"""Recursive live verification of every council input signal.

Hits the REAL providers with the configured keys and prints a content-quality
report per source — so you can confirm the council is actually being fed (and
not silently degrading to one API). Read-only: no orders, no ledger writes.

    python3 scripts/verify_signals.py
    python3 scripts/verify_signals.py --home "Portugal" --away "Nigeria" --date 2026-06-15
    python3 scripts/verify_signals.py --council     # also run the full LLM council (costs tokens)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

# BZZOIRO is opt-in; the live runner enables it via .env. Force it on for the
# probe so a missing flag does not masquerade as a dead API.
config.BZZOIRO_ENABLED = True

OK = "OK  "
WARN = "!!  "
DEAD = "XX  "


def _mark(ok: bool, partial: bool = False) -> str:
    return OK if ok else (WARN if partial else DEAD)


def _short(obj, n: int = 220) -> str:
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return s if len(s) <= n else s[:n] + "…"


def check_bzzoiro(home: str, away: str, date: str) -> bool:
    print("\n=== BZZOIRO API ===")
    from data import bzzoiro_mapper, fixture_bundle
    eid = bzzoiro_mapper.get_bzzoiro_event_id(home, away, date)
    print(f"  mapped event_id: {eid}")
    if not eid:
        print(f"{WARN}no event mapped (fixture may not be in BZZOIRO yet)")
        return False
    dig = fixture_bundle.build_bzzoiro_digest(home, away, date) or {}
    ml = dig.get("ml_prediction")
    ps = dig.get("prediction_summary") or {}
    unavailable = dig.get("unavailable_players") or {}
    n_out = sum(len(v) for v in unavailable.values()) if isinstance(unavailable, dict) else len(unavailable)
    print(f"{_mark(bool(ml))}ml_prediction: {ml}")
    print(f"{_mark(bool(ps))}prediction_summary: xg={ps.get('expected_goals')} "
          f"score={ps.get('most_likely_score')} conf={ps.get('model_confidence')} "
          f"ver={ps.get('model_version')}")
    print(f"{_mark(n_out > 0, partial=True)}lineup_status={dig.get('lineup_status')} "
          f"unavailable_players={n_out}")
    if n_out:
        names = []
        for side in ("home", "away"):
            for p in (unavailable.get(side) or []):
                names.append(f"{p.get('name')}({p.get('status')})")
        print(f"      → {', '.join(names[:8])}")
    return bool(ml)


def check_web(home: str, away: str, date: str) -> bool:
    print("\n=== WEB SEARCH (Serper) ===")
    from data import web_search
    res = web_search.gather_research(home, away, date)
    n = res.get("total_results", 0)
    print(f"{_mark(n > 0)}backend={res.get('backend')} total_results={n} "
          f"sources={len(res.get('sources') or [])}")
    print(f"      sources: {', '.join((res.get('sources') or [])[:10])}")
    for r in (res.get("previews") or [])[:3]:
        print(f"      • [{r.get('source')}] {_short(r.get('title'), 90)}")
        print(f"        {_short(r.get('snippet'), 160)}")
    return n > 0


def check_reddit(home: str, away: str) -> bool:
    print("\n=== REDDIT (r/soccer) ===")
    from data import reddit_sentiment
    res = reddit_sentiment.get_sentiment_bundle(home, away)
    n = res.get("comments_found", 0)
    print(f"{_mark(n > 0, partial=True)}source={res.get('source')} "
          f"threads={res.get('threads_found')} comments={n}")
    for c in (res.get("top_comments") or [])[:3]:
        print(f"      • {_short(c, 160)}")
    return n > 0


def check_grok(fixture: str, home: str, away: str, kickoff: str) -> bool:
    print("\n=== GROK (live X / news pulse) ===")
    from reasoning import llm
    from reasoning import prompts
    try:
        res = llm.call_grok(prompts.SOCIAL_PULSE_SYS,
                            prompts.social_pulse_input(fixture, home, away, kickoff))
        parsed = res.parsed if res else {}
    except Exception as exc:
        print(f"{DEAD}grok call failed: {exc!r}")
        return False
    ok = bool(parsed.get("summary") or parsed.get("breaking") or parsed.get("overall_lean"))
    print(f"{_mark(ok)}overall_lean={parsed.get('overall_lean')} "
          f"confidence={parsed.get('confidence')} breaking={len(parsed.get('breaking') or [])}")
    print(f"      summary: {_short(parsed.get('summary'), 220)}")
    for b in (parsed.get("breaking") or [])[:3]:
        print(f"      breaking: {_short(b, 140)}")
    return ok


def run_council(fixture: str, home: str, away: str, date: str) -> None:
    print("\n=== FULL COUNCIL (live LLM calls) ===")
    from data import fixture_bundle, web_search, reddit_sentiment
    from reasoning import council
    web = web_search.gather_research(home, away, date)
    reddit = reddit_sentiment.get_sentiment_bundle(home, away)
    ctx = fixture_bundle.build_context(home, away, home[:3].upper(), away[:3].upper(),
                                       match_date=date, fixture_name=fixture)
    cr = council.run_council(
        fixture, home[:3].upper(), away[:3].upper(), home, away,
        f"{date}T16:00:00Z", ctx.get("sportmonks_digest"), ctx.get("supabase_digest"),
        {"data_availability": "no_market"}, web, reddit,
        deterministic_context=None, bz_digest=ctx.get("bzzoiro_digest"),
    )
    print(f"  probabilities: {cr.probabilities}")
    print(f"  outcome={cr.outcome} p={cr.probability} confidence={cr.confidence}")
    print(f"  scout_flags: {len(cr.scout_flags or [])}")
    print(f"  summary: {_short(getattr(cr, 'summary', '') or cr.grounding.get('council_summary',''), 280)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default="Spain")
    ap.add_argument("--away", default="Cabo Verde")
    ap.add_argument("--date", default="2026-06-15")
    ap.add_argument("--council", action="store_true", help="also run the full LLM council")
    args = ap.parse_args()

    fixture = f"{args.home} vs {args.away}"
    print(f"Verifying signals for: {fixture}  ({args.date})")
    print(f"keys: bzzoiro={'set' if config.BZZOIRO_KEY else 'MISSING'} "
          f"serper={'set' if config.SERPER_API_KEY else 'MISSING'} "
          f"xai={'set' if config.XAI_KEY else 'MISSING'}")

    results = {
        "bzzoiro": check_bzzoiro(args.home, args.away, args.date),
        "web_search": check_web(args.home, args.away, args.date),
        "reddit": check_reddit(args.home, args.away),
        "grok": check_grok(fixture, args.home, args.away, f"{args.date}T16:00:00Z"),
    }

    if args.council:
        run_council(fixture, args.home, args.away, args.date)

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {_mark(v)}{k}")
    live = sum(1 for v in results.values() if v)
    print(f"\n{live}/{len(results)} signals returned usable content.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
