"""
predict_game.py — standalone, sandbox predictor for ANY match.

Pick any game (World Cup, friendly, club — anything with a Polymarket event),
feed it team names, and run the full reasoning council against live research +
market prices. It NEVER touches the arena: no ledger submission, no orders, no
wallet. Pure analysis you can run on tonight's fixture before risking anything.

It reuses the production brains:
  - data/web_search   — injuries / lineups / previews across many sources
  - data/reddit_sentiment
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
from data import web_search, reddit_sentiment
from data import polymarket as pm
from reasoning import council, gates
from betting.kelly import kelly_usd
from betting import decision as ev_decision

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
        try:
            moneyline = pm.get_moneyline_by_slug(args.pm_slug)
        except Exception as exc:
            console.print(f"  [yellow]Polymarket fetch failed ({exc}) — "
                          f"going market-blind.[/yellow]")
            moneyline = None
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

    # ── Structured grounding (Sportmonks if a fixture id was given; Supabase
    # priors resolve by team name, so they work for any international match) ─
    console.print("[dim]Building structured context (Sportmonks + Supabase)…[/dim]")
    from data import fixture_bundle
    ctx = fixture_bundle.build_context(
        home, away, home_code, away_code,
        sportmonks_fixture_id=args.fixture_id, fixture_name=fixture_name)
    sm_digest, sb_digest = ctx["sportmonks_digest"], ctx["supabase_digest"]
    console.print(f"  Context: sportmonks={'yes' if sm_digest else 'no'}  "
                  f"supabase={'yes' if sb_digest else 'no'}")

    # ── Council ────────────────────────────────────────────────────────────
    console.print("[dim]Convening reasoning council "
                  "(Grok → Scout → Analyst → Devil → Judge)…[/dim]")
    cr = council.run_council(
        fixture_name, home_code, away_code,
        home, away, date,
        sm_digest, sb_digest,
        pm_digest,
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
    g = cr.grounding or {}
    if g:
        anchor = g.get("anchor") or {}
        vt.add_row("Anchor", anchor.get("source") or "none")
        if g.get("shrink_lambda"):
            vt.add_row("Shrink λ", f"{g['shrink_lambda']:.2f} toward anchor (low confidence)")
        if g.get("sanity_flags"):
            vt.add_row("Sanity flags", "; ".join(g["sanity_flags"]))
    console.print(vt)
    if cr.council_summary:
        console.print(Panel(cr.council_summary, title="Why", expand=False))

    # ── EV-ranked decision across ALL outcomes (only if we have a market) ──
    if moneyline:
        from harness.profiles import get_profile
        profile = get_profile(args.profile)
        console.print(f"  [dim]profile: {profile.name} "
                      f"(edge≥{profile.min_edge_vs_fair*100:.1f}pp vs fair, "
                      f"kelly×{profile.kelly_fraction})[/dim]")
        # Evaluate home/draw/away (not just the council's pick): de-vig the
        # market, rank every outcome by EV, and pick the best tradable side.
        decision = ev_decision.evaluate_game(
            cr.probabilities, moneyline, home_code, away_code, args.bankroll,
            kelly_fraction=profile.kelly_fraction,
            min_edge_vs_fair=profile.min_edge_vs_fair,
        )

        et = Table(title="Per-outcome EV (de-vigged)")
        et.add_column("Outcome", style="cyan")
        et.add_column("Our prob"); et.add_column("Pay (raw)"); et.add_column("Fair")
        et.add_column("Edge vs fair"); et.add_column("EV / $1"); et.add_column("Kelly $")
        for e in decision.ranked:
            mark = " ★" if (decision.best and e.slot == decision.best.slot
                            and decision.should_trade) else ""
            et.add_row(
                f"{e.code}{mark}", f"{e.our_prob:.0%}",
                f"{e.raw_mid:.0%}" if e.raw_mid is not None else "—",
                f"{e.fair_prob:.0%}" if e.fair_prob is not None else "—",
                f"{e.edge_vs_fair*100:+.1f}pp", f"{e.ev_per_dollar*100:+.1f}%",
                f"${e.kelly_usd:.2f}" if e.kelly_usd else "—",
            )
        console.print(et)
        ov = (f"{decision.overround*100:+.1f}%"
              if decision.overround is not None else "n/a")
        console.print(f"  [dim]market overround (vig): {ov}[/dim]")

        # Risk overlay (scout veto / consensus / confidence) on the chosen side.
        best = decision.best
        size = 0.0
        gate = None
        if best:
            gate = gates.evaluate_gates(
                outcome=best.code, model_prob=best.our_prob,
                pm_mid=best.raw_mid,
                scout_flags=cr.scout_flags, confidence=cr.confidence,
                wallet_balance=args.bankroll,
                min_edge=None,                       # edge bar lives in decision.py
                scout_veto=profile.skip_on_high_scout_flag,
            )
            should_trade = decision.should_trade and gate.should_trade
            if should_trade and best.raw_mid:
                size = round(min(best.kelly_usd * gate.bet_multiplier,
                                 profile.max_bet_usd,
                                 config.MAX_BET_USD, args.bankroll), 2)
                if size < 1.0:
                    should_trade = False
                    size = 0.0
        else:
            should_trade = False

        gt = Table(title="Hypothetical Decision (NOT submitted)", show_header=False)
        gt.add_column("k", style="cyan"); gt.add_column("v")
        gt.add_row("Best action", decision.summary)
        if best and should_trade:
            limit = min(round(best.raw_mid + 0.02, 4), 0.99)
            gt.add_row("Decision", "[green]TRADE[/green]")
            gt.add_row("Side", best.code)
            gt.add_row("Suggested size", f"${size:.2f} (limit ≤ {limit})")
            gt.add_row("Multiplier", f"×{gate.bet_multiplier}")
        else:
            reason = (gate.veto_reason if gate and gate.veto_reason
                      else "no +EV side clears the bar")
            gt.add_row("Decision", f"[yellow]HOLD[/yellow] ({reason})")
        console.print(gt)
        if gate:
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
            "grounding": cr.grounding,
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
    p.add_argument("--fixture-id", type=int, default=None,
                   help="Sportmonks fixture id (enables real ML/odds/xG digest)")
    p.add_argument("--profile", choices=["monk", "anchor", "hunter", "blitz"],
                   default=None,
                   help="Trading profile for the hypothetical decision "
                        "(default: AGENT_PROFILE env, else 'anchor')")
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
