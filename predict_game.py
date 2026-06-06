"""
predict_game.py — standalone, sandbox predictor for ANY match.

Pick any game (World Cup, friendly, club — anything with a Polymarket event),
feed it team names, and run the full reasoning council against live research +
market prices. It NEVER touches the arena: no ledger submission, no orders, no
wallet. Pure analysis you can run on tonight's fixture before risking anything.

It reuses the production brains:
  - data/web_search   — injuries / lineups / previews across many sources
  - data/reddit_sentiment, data/kalshi
  - reasoning/council — Grok pulse → Scout → Analyst → Devil → Judge
  - reasoning/gates   — deterministic edge / consensus / veto gates
  - betting/kelly     — sizing (hypothetical only)

It works with or without a Polymarket slug:
  • With  --pm-slug : pulls live mids, computes edge, gives a hypothetical bet.
  • Without          : research + council still run; market-blind verdict only.

Sportmonks/Supabase are OPTIONAL here — friendlies often have neither, so the
council reasons from web + Grok + general knowledge when they're absent.

Examples
--------
  # Tonight's friendly, live market, no arena writes:
  python predict_game.py --home Argentina --away Honduras \\
      --home-code ARG --away-code HND --pm-slug fif-arg-hnd-2026-06-06

  # Any game, no slug (market-blind council only):
  python predict_game.py --home Brazil --away Morocco \\
      --home-code BRA --away-code MAR --date 2026-06-13

  # Just peek at live Polymarket prices for a slug:
  python predict_game.py --pm-slug fif-arg-hnd-2026-06-06 --prices-only
"""
from __future__ import annotations
import argparse
import datetime as _dt
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config
from data import web_search, reddit_sentiment, kalshi
from data import polymarket as pm
from reasoning import council, gates
from betting.kelly import kelly_usd

console = Console()


def _norm_mids(moneyline: dict | None) -> dict[str, float | None]:
    """Raw YES mids per slot {home, draw, away}."""
    if not moneyline:
        return {"home": None, "draw": None, "away": None}
    out = {}
    for k in ("home", "draw", "away"):
        out[k] = (moneyline.get("outcomes", {}).get(k) or {}).get("current_mid_yes")
    return out


def _normalized_probs(mids: dict[str, float | None]) -> dict[str, float | None]:
    vals = {k: v for k, v in mids.items() if isinstance(v, (int, float))}
    s = sum(vals.values())
    if s <= 0:
        return mids
    return {k: (round(v / s, 4) if isinstance(v, (int, float)) else None)
            for k, v in mids.items()}


def _outcome_to_slot(outcome: str, home_code: str, away_code: str) -> str | None:
    if outcome == "draw":
        return "draw"
    if outcome == home_code:
        return "home"
    if outcome == away_code:
        return "away"
    return None


def show_prices(slug: str) -> dict | None:
    ml = pm.get_moneyline_by_slug(slug)
    if not ml:
        console.print(f"[red]No Polymarket event found for slug:[/red] {slug}")
        return None
    mids = _norm_mids(ml)
    norm = _normalized_probs(mids)
    t = Table(title=f"Polymarket — {ml.get('fixture') or slug}")
    t.add_column("Outcome", style="cyan")
    t.add_column("Raw YES mid")
    t.add_column("Normalized")
    for slot in ("home", "draw", "away"):
        tc = (ml.get("outcomes", {}).get(slot) or {}).get("team_code", slot)
        raw = mids.get(slot)
        nrm = norm.get(slot)
        t.add_row(f"{slot} ({tc})",
                  f"{raw:.3f}" if raw is not None else "—",
                  f"{nrm:.1%}" if nrm is not None else "—")
    console.print(t)
    return ml


def run(args: argparse.Namespace) -> None:
    home, away = args.home, args.away
    home_code = (args.home_code or home[:3]).upper()
    away_code = (args.away_code or away[:3]).upper()
    date = args.date or _dt.date.today().isoformat()
    fixture_name = f"{home} vs {away}"

    console.print(Panel(
        f"[bold cyan]SANDBOX PREDICT[/bold cyan]  {fixture_name}\n"
        f"[dim]codes {home_code}/{away_code} · {date} · "
        f"{'slug ' + args.pm_slug if args.pm_slug else 'no market slug'} · "
        f"NO arena writes[/dim]",
        expand=False,
    ))

    # ── Market (optional) ──────────────────────────────────────────────────
    moneyline = None
    if args.pm_slug:
        console.print("[dim]Fetching live Polymarket prices…[/dim]")
        moneyline = pm.get_moneyline_by_slug(args.pm_slug)
        if moneyline:
            mids = _normalized_probs(_norm_mids(moneyline))
            console.print(f"  Market: " + "  ".join(
                f"{k}={v:.1%}" for k, v in mids.items() if v is not None))
        else:
            console.print("  [yellow]Slug returned no event — going market-blind.[/yellow]")

    pm_digest = None
    if moneyline:
        pm_digest = {
            "fixture": moneyline.get("fixture"),
            "market_handle": moneyline.get("polymarket_event_slug"),
            "implied_win_prob": {
                home_code: (_norm_mids(moneyline)["home"]),
                "draw": (_norm_mids(moneyline)["draw"]),
                away_code: (_norm_mids(moneyline)["away"]),
            },
            "data_availability": "mids_available",
        }

    # ── Research (always works on team names) ──────────────────────────────
    console.print("[dim]Researching injuries / lineups / previews…[/dim]")
    web = web_search.gather_research(home, away, date)
    console.print(f"  Web: {web['total_results']} results from "
                  f"{len(web['sources'])} sources via {web['backend']}")

    console.print("[dim]Pulling crowd sentiment…[/dim]")
    reddit = reddit_sentiment.get_sentiment_bundle(home, away)
    console.print(f"  Reddit/social: {len(reddit['top_comments'])} comments "
                  f"({reddit['source']})")

    console.print("[dim]Checking Kalshi cross-market…[/dim]")
    kalshi_ml = kalshi.get_moneyline(home, away)
    console.print(f"  Kalshi: {kalshi_ml['markets_found']} markets")

    # ── Council ────────────────────────────────────────────────────────────
    console.print("[dim]Convening reasoning council "
                  "(Grok → Scout → Analyst → Devil → Judge)…[/dim]")
    cr = council.run_council(
        fixture_name, home_code, away_code,
        home, away, date,
        None,            # no Sportmonks digest in sandbox
        None,            # no Supabase digest in sandbox
        pm_digest, kalshi_ml,
        web, reddit,
    )

    pulse_lean = (cr.social_pulse or {}).get("overall_lean", "n/a")
    console.print(f"  Grok pulse lean: {pulse_lean}")
    console.print(f"  Scout flags: {len(cr.scout_flags)}")

    # ── Verdict ────────────────────────────────────────────────────────────
    vt = Table(title="Council Verdict", show_header=False)
    vt.add_column("k", style="cyan"); vt.add_column("v")
    vt.add_row("Pick", f"[bold]{cr.outcome}[/bold]")
    vt.add_row("Probability", f"[green]{cr.probability:.1%}[/green]")
    vt.add_row("Confidence", cr.confidence)
    vt.add_row("Market alignment", cr.market_alignment)
    if cr.probabilities:
        vt.add_row("Full distribution",
                   "  ".join(f"{k}={float(v):.1%}" for k, v in cr.probabilities.items()))
    console.print(vt)
    if cr.council_summary:
        console.print(Panel(cr.council_summary, title="Why", expand=False))

    # ── Hypothetical trade (only if we have a market) ──────────────────────
    if moneyline:
        slot = _outcome_to_slot(cr.outcome, home_code, away_code)
        pm_mid = _norm_mids(moneyline).get(slot) if slot else None
        kalshi_mid = (kalshi_ml.get(slot) if slot else None)
        gate = gates.evaluate_gates(
            outcome=cr.outcome, model_prob=cr.probability,
            pm_mid=pm_mid, kalshi_mid=kalshi_mid,
            scout_flags=cr.scout_flags, confidence=cr.confidence,
            wallet_balance=args.bankroll,
        )
        size = 0.0
        if gate.should_trade and pm_mid:
            size = round(min(kelly_usd(cr.probability, pm_mid, args.bankroll)
                             * gate.bet_multiplier, config.MAX_BET_USD, args.bankroll), 2)
            if size < 1.0:
                gate.should_trade = False
                size = 0.0

        gt = Table(title="Hypothetical Trade (NOT submitted)", show_header=False)
        gt.add_column("k", style="cyan"); gt.add_column("v")
        gt.add_row("Our prob", f"{cr.probability:.1%}")
        gt.add_row("Market mid", f"{pm_mid:.1%}" if pm_mid is not None else "—")
        gt.add_row("Edge", f"{gate.edge*100:+.1f}pp")
        gt.add_row("Market agreement", gate.market_agreement)
        gt.add_row("Multiplier", f"×{gate.bet_multiplier}")
        gt.add_row("Decision", "[green]TRADE[/green]" if gate.should_trade
                   else f"[yellow]SKIP[/yellow] ({gate.veto_reason or 'no edge'})")
        if gate.should_trade:
            gt.add_row("Suggested size", f"${size:.2f} on {cr.outcome} "
                       f"(limit ≤ {min(round(pm_mid + 0.02, 4), 0.99)})")
        console.print(gt)
        for r in gate.reasons:
            console.print(f"    [dim]· {r}[/dim]")

    if args.json_out:
        payload = {
            "fixture": fixture_name,
            "outcome": cr.outcome,
            "probability": cr.probability,
            "confidence": cr.confidence,
            "probabilities": cr.probabilities,
            "market_alignment": cr.market_alignment,
            "social_pulse": cr.social_pulse,
            "scout_flags": cr.scout_flags,
            "summary": cr.council_summary,
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        console.print(f"[dim]Wrote {args.json_out}[/dim]")

    console.print(Panel("[bold green]Done — sandbox only, nothing submitted.[/bold green]",
                        expand=False))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sandbox match predictor — full council, no arena writes.")
    p.add_argument("--home", help="Home team name, e.g. Argentina")
    p.add_argument("--away", help="Away team name, e.g. Honduras")
    p.add_argument("--home-code", help="Short code (defaults to first 3 letters)")
    p.add_argument("--away-code", help="Short code (defaults to first 3 letters)")
    p.add_argument("--date", help="Match date YYYY-MM-DD (defaults to today)")
    p.add_argument("--pm-slug", help="Polymarket event slug (any product line)")
    p.add_argument("--bankroll", type=float, default=100.0,
                   help="Hypothetical bankroll for sizing (default 100)")
    p.add_argument("--prices-only", action="store_true",
                   help="Just print live Polymarket prices for --pm-slug and exit")
    p.add_argument("--json-out", help="Optional path to dump the verdict as JSON")
    args = p.parse_args()

    if args.prices_only:
        if not args.pm_slug:
            p.error("--prices-only requires --pm-slug")
        show_prices(args.pm_slug)
        return

    if not args.home or not args.away:
        p.error("--home and --away are required (unless using --prices-only)")
    run(args)


if __name__ == "__main__":
    main()
