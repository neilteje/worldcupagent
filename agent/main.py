from __future__ import annotations
import argparse, sys, time
from agent.config import load_settings
from agent.scheduler import run_daemon, run_once
from backtesting.runner import run_backtest, run_mode_comparison_backtest
from models.critic_policy import merge_critic_review
from reasoning.anthropic_review import anthropic_health_check, anthropic_key_status, critique_decisions_with_anthropic
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
    p.add_argument("--anthropic-health-check", action="store_true", help="Call Anthropic with a tiny health-check prompt and report non-secret status.")
    p.add_argument("--use-anthropic-critic", action="store_true", help="Ask Anthropic to critique dry-run decisions. The critic cannot authorize orders.")
    p.add_argument("--use-llm-claims", action="store_true", help="Use Anthropic to extract typed claims from text sources before deterministic blending.")
    p.add_argument("--use-llm-analyst", action="store_true", help="Use Anthropic Sonnet as a required second analyst for each decision.")
    p.add_argument("--decision-mode", choices=["deterministic", "llm_central"], default=None, help="Select deterministic-first forecasting or an LLM-central forecast mode that synthesizes all features.")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--compare-modes", action="store_true", help="Run a deterministic vs llm_central comparison backtest on the same synthetic sample.")
    p.add_argument("--backtest-sample", type=int, default=50)
    p.add_argument("--use-claude", action="store_true", help="Allow optional Claude critique in backtest, capped by BACKTEST_LLM_BUDGET_USD.")
    args = p.parse_args()
    settings = load_settings(dry_run_override=True if args.dry_run else None)
    if args.decision_mode:
        settings = type(settings)(**{**settings.__dict__, "decision_mode": args.decision_mode})
    start = time.time()
    command = " ".join(sys.argv)
    decisions = []
    anthropic_status = {"key": anthropic_key_status(settings)}
    if args.anthropic_health_check:
        anthropic_status["health_check"] = anthropic_health_check(settings)
    exit_status = 0
    try:
        if args.backtest:
            run_backtest(settings, sample_size=args.backtest_sample, use_claude=args.use_claude)
        elif args.compare_modes:
            run_mode_comparison_backtest(settings, sample_size=args.backtest_sample, include_claude_report=True)
        elif args.daemon:
            decisions = run_daemon(settings, args.interval_seconds, args.fixture_code, args.window, use_synthetic_fixtures=args.use_synthetic_fixtures, verbose=args.verbose, max_iterations=args.max_iterations, use_llm_analyst=args.use_llm_analyst, use_llm_claims=args.use_llm_claims, decision_mode=settings.decision_mode)
        else:
            decisions = run_once(settings, args.fixture_code, args.window, use_synthetic_fixtures=args.use_synthetic_fixtures, verbose=args.verbose, use_llm_analyst=args.use_llm_analyst, use_llm_claims=args.use_llm_claims, decision_mode=settings.decision_mode)
        if args.use_anthropic_critic and decisions:
            anthropic_status["critic"] = critique_decisions_with_anthropic(settings, decisions, storage_dir=settings.storage_dir)
            decisions = merge_critic_review(decisions, anthropic_status["critic"])
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
                ideas_implemented=[
                    "Added Anthropic Sonnet signal analyst as a required second decision input." if args.use_llm_analyst else
                    "Added Anthropic structured claim extraction as capped deterministic model input." if args.use_llm_claims else
                    "Added Anthropic API health check and non-authorizing LLM critic path." if (args.anthropic_health_check or args.use_anthropic_critic) else
                    "Added structured run review artifacts for every agent run."
                ],
                llm_review=anthropic_status,
            )
            if args.verbose:
                print(f"Review written: {review['markdown']} / {review['json']}")

if __name__ == "__main__":
    main()
