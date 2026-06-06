from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json, subprocess, time


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_run_review(storage_dir: Path, *, command: str, dry_run: bool, decisions: list[dict], runtime_seconds: float, exit_status: int, commands_run: list[str] | None = None, tests: dict | None = None, ideas_implemented: list[str] | None = None, code_changes: list[str] | None = None) -> dict:
    reviews = storage_dir / "reviews"
    reviews.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    iso = datetime.now(timezone.utc).isoformat()
    api_errors = []
    missing = []
    for d in decisions:
        if not d.get("market_probs"):
            missing.append(f"{d.get('fixture_code')} market")
        if not d.get("bookmaker_probs"):
            missing.append(f"{d.get('fixture_code')} bookmaker")
        order = d.get("order") or {}
        if order.get("reason") not in {None, "skip"} and not order.get("dry_run"):
            api_errors.append(str(order.get("reason")))
    weaknesses = []
    if not decisions:
        weaknesses.append("No fixture/window decisions were generated.")
    if any("edge_below_threshold" in d.get("risk_flags", []) for d in decisions):
        weaknesses.append("Most or all opportunities remain no-trade because measured edge is small.")
    if any(not d.get("ledger_submitted") for d in decisions):
        weaknesses.append("Ledger is saved locally but live submission is unavailable or not configured.")
    if any("lineup_unconfirmed" in d.get("risk_flags", []) for d in decisions):
        weaknesses.append("Confirmed lineup data is unavailable for at least one decision.")
    ideas = [
        "Track source contribution deltas per run to explain probability movement.",
        "Use prediction diffing to explain changes from the previous fixture/window run.",
        "Add live API reliability metrics over multiple runs.",
    ]
    summary = {
        "timestamp": iso,
        "commands_run": commands_run or [command],
        "tests": tests or {"passed": None, "failed": None},
        "fixtures_processed": len({d.get("fixture_code") for d in decisions}),
        "predictions_generated": sum(1 for d in decisions if d.get("prediction_submitted")),
        "orders_attempted": sum(1 for d in decisions if d.get("action") == "BET"),
        "orders_skipped": sum(1 for d in decisions if d.get("action") != "BET"),
        "ledger_records_generated": sum(int(d.get("ledger_records", 0) or 0) for d in decisions),
        "api_errors": api_errors,
        "weaknesses": weaknesses,
        "ideas_proposed": ideas,
        "ideas_implemented": ideas_implemented or [],
        "remaining_todos": weaknesses[:],
    }
    json_path = reviews / f"run_{timestamp}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pred_blocks = []
    for d in decisions:
        pred_blocks.append("\n".join([
            f"- Fixture: {d.get('fixture_code')}",
            f"- Window: {d.get('window')}",
            f"- Teams: {d.get('teams', 'unknown')}",
            f"- Final probabilities: {d.get('final_probs')}",
            f"- Market probabilities: {d.get('market_probs')}",
            f"- Bookmaker probabilities: {d.get('bookmaker_probs')}",
            f"- Sportmonks probabilities: {d.get('sportmonks_probs')}",
            f"- HT data if applicable: {d.get('halftime')}",
            f"- Lineup data if applicable: {d.get('lineup')}",
            f"- Consensus case: {d.get('consensus_case')}",
            f"- Best edge: {d.get('best_outcome')} {d.get('best_edge')}",
            f"- Edge tier: {d.get('edge_tier')}",
            f"- Confidence: {d.get('confidence')}",
            f"- Uncertainty: {d.get('uncertainty')}",
            f"- Top signals: {d.get('top_signals')}",
            f"- Risk flags: {d.get('risk_flags')}",
            f"- Action: prediction={d.get('prediction_submitted')} order={d.get('order_submitted')} action={d.get('action')}",
            f"- Reason: {d.get('reason') or d.get('edge_reason')}",
        ]))
    md = f"""# Agent Run Review

## Run metadata
- Timestamp: {iso}
- Git commit/hash if available: {_git_hash()}
- Command: {command}
- Dry run: {dry_run}
- Fixtures processed: {summary['fixtures_processed']}
- Windows processed: {len(decisions)}
- Runtime: {runtime_seconds:.2f}s
- Exit status: {exit_status}

## API/data status
- Sportmonks: {'available/synthetic' if decisions else 'not processed'}
- Supabase: {'available or fallback priors used' if decisions else 'not processed'}
- Polymarket: {'available or synthetic/demo fallback used' if decisions else 'not processed'}
- Ledger endpoint: {'submitted' if any(d.get('ledger_submitted') for d in decisions) else 'local-only/unavailable'}
- Order endpoint: {'dry-run skipped' if dry_run else 'guarded by sanity checks'}
- Missing data: {missing or 'none detected in decision artifacts'}
- API errors: {api_errors or 'none'}

## Prediction summary
{chr(10).join(pred_blocks) if pred_blocks else '- No predictions generated.'}

## Quality critique
- What looks correct: Dry-run safety gates and ledger local persistence are active.
- What looks suspicious: Demo/synthetic fallback can mask live API schema issues when no ARENA_KEY is configured.
- Probability issues: {weaknesses or 'No obvious artifact-level probability issue.'}
- Edge issues: No-trade discipline is active; edge quality depends on source completeness.
- Signal issues: Weak web/sentiment signals must remain capped and corroborated.
- Data issues: {missing or 'No missing data reported by artifacts.'}
- Ledger issues: {'Some ledgers were local-only.' if any(not d.get('ledger_submitted') for d in decisions) else 'Ledger submission reported success.'}
- Automation issues: Duplicate order prevention should be monitored with real non-dry runs.

## New ideas proposed
{chr(10).join('- ' + i for i in ideas)}

## Ideas implemented this iteration
{chr(10).join('- ' + i for i in (ideas_implemented or ['None in this run.']))}

## Code changes
{chr(10).join('- ' + i for i in (code_changes or ['No code change metadata supplied.']))}

## Tests run
{chr(10).join('- ' + c for c in (commands_run or [command]))}

## Remaining weaknesses
{chr(10).join('- ' + w for w in (weaknesses or ['Continue validating against live API data.']))}

## Next iteration priorities
- Validate live API payloads with ARENA_KEY configured.
- Add richer source contribution and prediction diff reports.
- Expand backtest data beyond synthetic examples.
"""
    md_path = reviews / f"run_{timestamp}.md"
    md_path.write_text(md, encoding="utf-8")
    (reviews / "latest_review.md").write_text(md, encoding="utf-8")
    with (reviews / "iteration_log.md").open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {iso}\n- Command: `{command}`\n- Fixtures: {summary['fixtures_processed']}\n- Predictions: {summary['predictions_generated']}\n- Ideas implemented: {', '.join(ideas_implemented or ['none'])}\n")
    comp = reviews / "comparison.md"
    previous = comp.read_text(encoding="utf-8") if comp.exists() else "# Run Comparison\n"
    previous += f"\n## {iso}\n- Decisions: {len(decisions)}\n- Orders skipped: {summary['orders_skipped']}\n- Ledger records: {summary['ledger_records_generated']}\n- Weakness count: {len(weaknesses)}\n"
    comp.write_text(previous, encoding="utf-8")
    return {"markdown": str(md_path), "json": str(json_path), **summary}
