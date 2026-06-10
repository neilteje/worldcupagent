"""
CLI for the paper-trading harness.

Typical flow for tomorrow:
  python -m harness init                         # snapshot fixtures + schedule
  python -m harness run                          # live: trades each window at its time
  # …after matches finish, edit storage/harness/<date>/results.json…
  python -m harness settle                       # grade + write reports/plots

Useful extras:
  python -m harness now --fixture FRD-POR-NGA --window PRE_MATCH   # run one window
  python -m harness run --start-now              # ignore wait times (smoke test)
  python -m harness run --engine deterministic   # skip LLM calls
  python -m harness report                       # regenerate plots/CSV
"""
from __future__ import annotations
import argparse

from harness import runner
from harness.backtest import run_harness_backtest, run_harness_backtest_comparison
from harness.fixtures import DEFAULT_DATE


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--date", default=DEFAULT_DATE, help="Match date YYYY-MM-DD (default tomorrow's friendlies).")
    p.add_argument("--session", default=None, help="Session name (default = date).")
    p.add_argument("--fixtures", default=None, help="Override fixtures JSON path.")
    p.add_argument("--profiles", default=None, help="Override agent-profiles JSON path.")
    p.add_argument("--engine", default="council", choices=["council", "deterministic", "market"],
                   help="Prediction engine (falls back automatically).")
    p.add_argument("--market", default="auto", choices=["auto", "real", "synthetic"],
                   help="Market source: auto (real else demo), real, or synthetic demo.")
    p.add_argument("--refresh", action="store_true", help="Ignore cached predictions and recompute.")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create session, snapshot fixtures, write results template.")
    _common(p_init)

    p_now = sub.add_parser("now", help="Run one (fixture, window) immediately.")
    _common(p_now)
    p_now.add_argument("--fixture", required=True, help="Fixture code, e.g. FRD-POR-NGA.")
    p_now.add_argument("--window", choices=["PRE_MATCH", "HT"], default=None,
                       help="Window (default: both).")

    p_run = sub.add_parser("run", help="Live loop: execute each window at its trigger time.")
    _common(p_run)
    p_run.add_argument("--start-now", action="store_true", help="Skip waits; run all windows now.")
    p_run.add_argument("--poll-seconds", type=float, default=15.0, help="Sleep granularity while waiting.")

    p_settle = sub.add_parser("settle", help="Settle open trades against results.json, then report.")
    _common(p_settle)

    p_report = sub.add_parser("report", help="Regenerate performance CSV/summary/plots.")
    _common(p_report)

    p_backtest = sub.add_parser("backtest", help="Replay historical rows through harness profiles and paper ledger.")
    p_backtest.add_argument("--dataset", choices=["synthetic", "wc2022"], default="wc2022")
    p_backtest.add_argument("--sample", type=int, default=20)
    p_backtest.add_argument("--session", default=None)
    p_backtest.add_argument("--profiles", default=None)
    p_backtest.add_argument("--engine", choices=["deterministic", "council", "market", "compare"], default="deterministic")

    args = ap.parse_args()
    {
        "init": runner.cmd_init,
        "now": runner.cmd_now,
        "run": runner.cmd_run,
        "settle": runner.cmd_settle,
        "report": runner.cmd_report,
        "backtest": lambda a: print(run_harness_backtest_comparison(
            dataset=a.dataset,
            sample_size=a.sample,
            session=a.session,
            profiles_path=a.profiles,
        ) if a.engine == "compare" else run_harness_backtest(
            dataset=a.dataset,
            sample_size=a.sample,
            session=a.session,
            profiles_path=a.profiles,
            engine=a.engine,
        )),
    }[args.cmd](args)


if __name__ == "__main__":
    main()
