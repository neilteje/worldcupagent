from __future__ import annotations
import argparse
from agent.config import load_settings
from agent.scheduler import run_daemon, run_once
from backtesting.runner import run_backtest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--daemon", action="store_true")
    p.add_argument("--interval-seconds", type=int, default=300)
    p.add_argument("--fixture-code")
    p.add_argument("--window", choices=["PRE_MATCH", "HT", "HALFTIME", "prematch", "halftime"], default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--backtest-sample", type=int, default=50)
    p.add_argument("--use-claude", action="store_true", help="Allow optional Claude critique in backtest, capped by BACKTEST_LLM_BUDGET_USD.")
    args = p.parse_args()
    settings = load_settings(dry_run_override=True if args.dry_run else None)
    if args.backtest:
        run_backtest(settings, sample_size=args.backtest_sample, use_claude=args.use_claude)
    elif args.daemon:
        run_daemon(settings, args.interval_seconds, args.fixture_code, args.window)
    else:
        run_once(settings, args.fixture_code, args.window)

if __name__ == "__main__":
    main()
