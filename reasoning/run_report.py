from __future__ import annotations

def fmt_probs(p):
    return " / ".join(f"{k.upper()} {float((p or {}).get(k,0)):.2f}" for k in ("home","draw","away"))

def print_run_report(report: dict) -> str:
    lines = [
        f"Fixture: {report.get('fixture_code')}",
        f"Window: {report.get('window')}",
        f"Archetype: {((report.get('archetype') or {}).get('match_archetype'))}",
        f"Market regime: {((report.get('archetype') or {}).get('market_regime'))}",
        f"Arbiter: {((report.get('arbiter') or {}).get('mode'))}",
        f"Final: {fmt_probs(report.get('final_probs'))}",
        f"Market: {fmt_probs(report.get('market_probs'))}",
        f"Bookmaker: {fmt_probs(report.get('bookmaker_probs'))}",
        f"Consensus: {report.get('consensus_case')}",
        f"Best edge: {str(report.get('best_outcome')).upper()} {float(report.get('best_edge',0)):+.2f}",
        f"Edge tier: {report.get('edge_tier')}",
        f"Action: {report.get('action')}",
        f"Risk flags: {report.get('risk_flags')}",
        f"Prediction submitted: {report.get('prediction_submitted')}",
        f"Order submitted: {report.get('order_submitted')}",
        f"Dry run: {report.get('dry_run')}",
        f"Ledger: {report.get('ledger_submitted')} ({report.get('ledger_records')} records)",
        f"Trace score: {float((report.get('trace_quality') or {}).get('score', 0.0)):.2f}",
    ]
    text = "\n".join(lines)
    print(text)
    return text
