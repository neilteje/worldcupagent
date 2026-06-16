"""
CLI for the live arena runner.

    python -m live run [--dry-run] [--agents monk,anchor]
    python -m live once --fixture-id 19609127 [--window PRE_MATCH] [--dry-run]
    python -m live status
    python -m live report
    python -m live test
"""
from __future__ import annotations
import argparse
import json


def _parse_agents(arg: str | None) -> list[str] | None:
    if not arg:
        return None
    return [a.strip().lower() for a in arg.split(",") if a.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m live",
                                 description="World Cup live 4-agent runner")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="resumable end-to-end loop until the final")
    p_run.add_argument("--dry-run", action="store_true",
                       help="full pipeline, but no orders and no ledger submit")
    p_run.add_argument("--agents", help="comma list (default: all with keys set)")

    p_once = sub.add_parser("once", help="run one fixture × window now")
    p_once.add_argument("--fixture-id", type=int, required=True)
    p_once.add_argument("--window", default="PRE_MATCH",
                        choices=["PRE_MATCH", "HT"])
    p_once.add_argument("--dry-run", action="store_true")
    p_once.add_argument("--agents", help="comma list (default: all with keys set)")
    p_once.add_argument("--force", action="store_true",
                        help="run even if state says this window is done")

    sub.add_parser("status", help="state summary: done / pending / settled")
    sub.add_parser("report", help="retrospective evaluation report")
    sub.add_parser("test", help="connectivity check for every agent key")

    args = ap.parse_args()

    if args.cmd == "run":
        from live.roster import load_roster
        from live.runner import LiveRunner
        agents = load_roster(only=_parse_agents(args.agents))
        LiveRunner(agents=agents, dry_run=args.dry_run).run_forever()

    elif args.cmd == "once":
        from live.roster import load_roster
        from live.state import LiveState
        from live.cycle import run_window_cycle
        agents = load_roster(only=_parse_agents(args.agents))
        state = LiveState()
        if state.window_done(args.fixture_id, args.window) and not args.force:
            print(f"{args.fixture_id}:{args.window} already done "
                  f"(use --force to rerun)")
            return
        result = run_window_cycle(args.fixture_id, args.window, agents,
                                  dry_run=args.dry_run)
        state.mark_window(args.fixture_id, args.window, "dry_run" if args.dry_run else "done",
                          fixture_name=result.get("fixture_name", ""),
                          agents=result.get("agents"))
        print(json.dumps(result, indent=2, default=str)[:4000])

    elif args.cmd == "status":
        from live.state import LiveState
        from live.runner import LiveRunner, flatten_schedule
        state = LiveState()
        print(json.dumps(state.summary(), indent=2, default=str))
        try:
            from data import sportmonks
            fixtures = flatten_schedule(sportmonks.get_season_schedule())
            pending = [f for f in fixtures
                       if not state.window_done(f["id"], "PRE_MATCH")][:5]
            print("\nNext unprocessed fixtures:")
            for f in pending:
                print(f"  {f['id']}  {f.get('starting_at')}  {f.get('name')}")
        except Exception as exc:
            print(f"(schedule unavailable: {exc!r})")

    elif args.cmd == "report":
        from live.report import build_report
        print(build_report())

    elif args.cmd == "test":
        from live.roster import load_roster
        from live.arena_client import ArenaClient
        agents = load_roster()
        ok = True
        for a in agents:
            chk = ArenaClient(a.api_key, a.name).check()
            if chk["ok"]:
                print(f"  {a.name:8s} OK   agent={chk.get('display_name')}  "
                      f"phase={chk.get('phase')}  "
                      f"available=${chk.get('available', 0):.2f}  "
                      f"locked=${chk.get('locked', 0):.2f}")
            else:
                ok = False
                print(f"  {a.name:8s} FAIL {chk.get('error')}")
        # Shared data endpoints through the first key
        from live.runner import flatten_schedule
        try:
            from data import sportmonks
            n = len(flatten_schedule(sportmonks.get_season_schedule()))
            print(f"  schedule OK   {n} fixtures")
        except Exception as exc:
            ok = False
            print(f"  schedule FAIL {exc!r}")
        try:
            m = ArenaClient(agents[0].api_key).matches()
            print(f"  matches  OK   {len(m)} entries")
        except Exception as exc:
            print(f"  matches  WARN {exc!r} (runner falls back to local clock)")
        print("\nAll good." if ok else "\nFix the failures above before match day.")


if __name__ == "__main__":
    main()
