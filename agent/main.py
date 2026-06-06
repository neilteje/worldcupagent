from __future__ import annotations
import argparse, sys, time
from agent.config import load_settings
from agent.scheduler import run_daemon, run_once
from backtesting.runner import run_backtest
from reasoning.review_writer import write_run_review


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m agent.main")
    p.add_argument("--once", action="store_true")
    p.add_argument("--daemon", action="store_true")
    p.add_argument("--interval-seconds", type=int, default=300)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--fixture-code")
    p.add_argument("--window", choices=["PRE_MATCH", "HT", "HALFTIME", "prematch", "halftime"], default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--use-synthetic-fixtures", action="store_true")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--backtest-sample", type=int, default=50)
    p.add_argument("--use-claude", action="store_true", help="Allow optional Claude critique in backtest, capped by BACKTEST_LLM_BUDGET_USD.")
    args = p.parse_args()
    settings = load_settings(dry_run_override=True if args.dry_run else None)
    start = time.time()
    command = " ".join(sys.argv)
    decisions = []
    exit_status = 0
    try:
        if args.backtest:
            run_backtest(settings, sample_size=args.backtest_sample, use_claude=args.use_claude)
        elif args.daemon:
            decisions = run_daemon(settings, args.interval_seconds, args.fixture_code, args.window, use_synthetic_fixtures=args.use_synthetic_fixtures, verbose=args.verbose, max_iterations=args.max_iterations)
        else:
            decisions = run_once(settings, args.fixture_code, args.window, use_synthetic_fixtures=args.use_synthetic_fixtures, verbose=args.verbose)
    except Exception:
        exit_status = 1
        raise
    finally:
        if decisions:
            review = write_run_review(
                settings.storage_dir,
                command=command,
                dry_run=settings.dry_run,
                decisions=decisions,
                runtime_seconds=time.time() - start,
                exit_status=exit_status,
                commands_run=[command],
                ideas_implemented=["Added structured run review artifacts for every agent run."],
            )
            if args.verbose:
                print(f"Review written: {review['markdown']} / {review['json']}")

if __name__ == "__main__":
    main()
