"""
World Cup 2026 – Today's Fixtures Explorer
===========================================

Pulls all fixtures scheduled for *today* (UTC) and enriches each one with:
  • Sportmonks ML predictions
  • Bookmaker implied probabilities
  • Polymarket CLOB midpoints (via arena proxy)
  • Kalshi market midpoints (public API)
  • Live / half-time stats (if in-play)
  • Lineup availability flags

Run directly::

    python -m scripts.exploration

Or from the repo root::

    python -m worldcupagent.scripts.exploration
"""

from __future__ import annotations

import datetime as _dt
import sys
import os
from datetime import timezone
from typing import Any, Mapping, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box

# ── Ensure repo root is on path when run as a script ──────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data import sportmonks
from data import polymarket
from data import kalshi

from dotenv import load_dotenv

load_dotenv(override=True)  # Load environment variables from .env file if present

console = Console()


# ── Helpers ────────────────────────────────────────────────────────────────

def _is_today(iso_ts: Optional[str]) -> bool:
    """Return True if the ISO-8601 timestamp falls on the current UTC date."""
    if not iso_ts:
        return False
    try:
        dt = _dt.datetime.fromisoformat(iso_ts.rstrip("Z"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.date() == _dt.datetime.now(timezone.utc).date()
    except Exception:
        return False


def _is_today_ms(kickoff_ms: Optional[float]) -> bool:
    """Return True if a millisecond UTC timestamp falls on the current UTC date."""
    if kickoff_ms is None:
        return False
    try:
        dt = _dt.datetime.fromtimestamp(float(kickoff_ms) / 1000, tz=timezone.utc)
        return dt.date() == _dt.datetime.now(timezone.utc).date()
    except Exception:
        return False


def _team_name(fixture: Mapping, side: str) -> str:
    """Extract a readable team name from a fixture dict."""
    key = f"{side}_team"
    team = fixture.get(key) or {}
    return (
        team.get("name")
        or team.get("country")
        or team.get("short_code")
        or str(team)
    )


def _format_prob(p: Optional[float]) -> str:
    """Format a probability (0-1) as a percentage with one decimal place."""
    if p is None:
        return "—"
    return f"{p * 100:.1f}%"


def _format_prob_color(p: Optional[float], high: float = 0.6, low: float = 0.3) -> Text:
    """Format a probability with color coding."""
    if p is None:
        return Text("—", style="dim")
    text = f"{p * 100:.1f}%"
    if p >= high:
        return Text(text, style="bold green")
    elif p <= low:
        return Text(text, style="red")
    return Text(text, style="yellow")


def _safe_kickoff(kickoff_ms: Any) -> tuple[Optional[str], Optional[str]]:
    """Convert ms kickoff to (iso_str, hh_mm_str)."""
    if kickoff_ms is None:
        return None, None
    try:
        dt = _dt.datetime.fromtimestamp(float(kickoff_ms) / 1000, tz=timezone.utc)
        return dt.isoformat(), dt.strftime("%H:%M")
    except Exception:
        return str(kickoff_ms), str(kickoff_ms)


# ── Data enrichment ───────────────────────────────────────────────────────

def enrich_fixture(fixture: dict) -> dict:
    """
    Enrich a fixture dict with all available data sources.
    Returns a dict with keys: detail, ml_probs, book_probs, poly_probs, kalshi_probs, live_stats
    """
    fixture_id = fixture.get("id")
    home_country = fixture.get("home_country") or _team_name(fixture, "home")
    away_country = fixture.get("away_country") or _team_name(fixture, "away")

    result = {
        "detail": {},
        "ml_probs": {},
        "book_probs": {},
        "poly_probs": {},
        "kalshi_probs": {},
        "live_stats": {},
    }

    # 1. Sportmonks detail (predictions, odds, participants, stats)
    try:
        detail = sportmonks.get_fixture_detail_safe(fixture_id)
        result["detail"] = detail
        result["ml_probs"] = sportmonks.extract_ml_probabilities(detail) or {}
        result["book_probs"] = sportmonks.extract_bookmaker_odds(detail) or {}
        result["live_stats"] = sportmonks.extract_ht_stats(detail)
    except Exception:
        pass

    # 2. Polymarket moneyline
    try:
        ml = polymarket.get_moneyline(int(fixture_id))
        if ml:
            result["poly_probs"] = polymarket.extract_implied_probs(ml)
    except Exception:
        pass

    # 3. Kalshi moneyline
    try:
        kalshi_markets = kalshi.search_fixture_markets(home_country, away_country, strict=False)
        if kalshi_markets:
            kalshi_probs: dict[str, float] = {}
            for m in kalshi_markets:
                mid = kalshi._yes_mid(m)
                if mid is None:
                    continue
                text = kalshi._text(m)
                # Try to identify which outcome this market is for
                home_lower = home_country.lower()
                away_lower = away_country.lower()
                if "draw" in text or "tie" in text:
                    kalshi_probs["draw"] = mid
                elif home_lower in text and away_lower not in text:
                    kalshi_probs["home_win"] = mid
                elif away_lower in text and home_lower not in text:
                    kalshi_probs["away_win"] = mid
            if kalshi_probs:
                # Normalize
                total = sum(kalshi_probs.values())
                if total > 0:
                    kalshi_probs = {k: v / total for k, v in kalshi_probs.items()}
                result["kalshi_probs"] = kalshi_probs
    except Exception:
        pass

    return result


# ── UI Rendering ──────────────────────────────────────────────────────────

def render_fixture_card(fixture: dict, enriched: dict) -> Panel:
    """Render a single fixture as a rich Panel with all data."""
    home = fixture.get("home_country") or _team_name(fixture, "home")
    away = fixture.get("away_country") or _team_name(fixture, "away")
    kickoff_iso = fixture.get("starting_at") or ""
    try:
        kickoff_dt = _dt.datetime.fromisoformat(kickoff_iso.rstrip("Z"))
        kickoff_str = kickoff_dt.strftime("%H:%M UTC")
    except Exception:
        kickoff_str = kickoff_iso

    fixture_id = fixture.get("id", "?")

    # Build the main comparison table
    comp_table = Table(
        show_header=True,
        header_style="bold white",
        box=box.SIMPLE,
        pad_edge=False,
        padding=(0, 1),
    )
    comp_table.add_column("Source", style="bold cyan", width=14)
    comp_table.add_column(f"{home}", justify="right", width=10)
    comp_table.add_column("Draw", justify="right", width=10)
    comp_table.add_column(f"{away}", justify="right", width=10)

    ml = enriched.get("ml_probs", {})
    if ml:
        comp_table.add_row(
            "Sportmonks ML",
            _format_prob_color(ml.get("home_win")),
            _format_prob_color(ml.get("draw")),
            _format_prob_color(ml.get("away_win")),
        )

    book = enriched.get("book_probs", {})
    if book:
        comp_table.add_row(
            "Bookmaker",
            _format_prob_color(book.get("home_win")),
            _format_prob_color(book.get("draw")),
            _format_prob_color(book.get("away_win")),
        )

    poly = enriched.get("poly_probs", {})
    if poly:
        home_code = fixture.get("home_short_code", "").upper() or home[:3].upper()
        away_code = fixture.get("away_short_code", "").upper() or away[:3].upper()
        comp_table.add_row(
            "Polymarket",
            _format_prob_color(poly.get(home_code)),
            _format_prob_color(poly.get("draw")),
            _format_prob_color(poly.get(away_code)),
        )

    kalshi = enriched.get("kalshi_probs", {})
    if kalshi:
        comp_table.add_row(
            "Kalshi",
            _format_prob_color(kalshi.get("home_win")),
            _format_prob_color(kalshi.get("draw")),
            _format_prob_color(kalshi.get("away_win")),
        )

    # Live stats section
    live = enriched.get("live_stats", {})
    live_text = Text()
    if live:
        ht_score = live.get("ht_score", {})
        if ht_score:
            live_text.append(f"  HT Score: ", style="bold")
            live_text.append(f"{ht_score.get('home', '?')} - {ht_score.get('away', '?')}\n")
        for stat_name in ("Expected Goals", "Ball Possession", "Shots on Goal", "Yellow Cards", "Red Cards"):
            val = live.get(stat_name)
            if val is not None:
                live_text.append(f"  {stat_name}: ", style="bold")
                live_text.append(f"{val}\n")
    else:
        live_text.append("  Pre-match (no live stats yet)", style="dim")

    # Edge detection: compare ML vs Polymarket
    edge_text = Text()
    if ml and poly:
        home_code = fixture.get("home_short_code", "").upper() or home[:3].upper()
        away_code = fixture.get("away_short_code", "").upper() or away[:3].upper()
        for label, key, p_key in [
            (f"{home}", "home_win", home_code),
            ("Draw", "draw", "draw"),
            (f"{away}", "away_win", away_code),
        ]:
            ml_p = ml.get(key)
            poly_p = poly.get(p_key)
            if ml_p is not None and poly_p is None:
                edge_text.append(f"  {label}: ML={_format_prob(ml_p)} | Market=—\n", style="dim")
            elif ml_p is not None and poly_p is not None:
                edge = ml_p - poly_p
                if abs(edge) >= 0.05:
                    style = "bold green" if edge > 0 else "bold red"
                    direction = "▲ OVER" if edge > 0 else "▼ UNDER"
                    edge_text.append(
                        f"  {label}: ML={_format_prob(ml_p)} Poly={_format_prob(poly_p)} "
                        f"Edge={edge:+.1%} {direction}\n",
                        style=style,
                    )
                else:
                    edge_text.append(
                        f"  {label}: ML={_format_prob(ml_p)} Poly={_format_prob(poly_p)} "
                        f"Edge={edge:+.1%} (within noise)\n",
                        style="dim",
                    )

    # Assemble the panel content
    content = Text()
    content.append(f"🆔 Fixture ID: {fixture_id}\n", style="dim")
    content.append(f"🕐 Kickoff: {kickoff_str}\n\n", style="bold magenta")
    content.append(comp_table)
    content.append("\n\n📊 Live Stats:\n", style="bold")
    content.append(live_text)

    if edge_text.plain.strip():
        content.append("\n⚡ Edges (ML vs Polymarket):\n", style="bold")
        content.append(edge_text)

    title = f"[bold white]{home} vs {away}[/bold white]"
    return Panel(content, title=title, border_style="bright_blue", box=box.ROUNDED)


def render_summary_table(fixtures: list[dict], enriched_list: list[dict]) -> Table:
    """Render a compact summary table of all today's fixtures."""
    table = Table(
        title="⚽ World Cup 2026 — Today's Fixtures Summary",
        title_style="bold white",
        box=box.DOUBLE_EDGE,
        show_lines=True,
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Fixture", style="cyan", no_wrap=True)
    table.add_column("Kickoff", style="magenta", justify="center")
    table.add_column("ML", justify="center")
    table.add_column("Book", justify="center")
    table.add_column("Polymarket", justify="center")
    table.add_column("Kalshi", justify="center")
    table.add_column("Best Edge", justify="center")

    for i, (fixture, enriched) in enumerate(zip(fixtures, enriched_list), 1):
        home = fixture.get("home_country") or _team_name(fixture, "home")
        away = fixture.get("away_country") or _team_name(fixture, "away")
        kickoff_iso = fixture.get("starting_at") or ""
        try:
            kickoff_dt = _dt.datetime.fromisoformat(kickoff_iso.rstrip("Z"))
            kickoff_str = kickoff_dt.strftime("%H:%M")
        except Exception:
            kickoff_str = "?"

        def _mini_prob_bar(probs: dict) -> str:
            """Create a mini probability summary like '45/25/30'."""
            if not probs:
                return "—"
            h = probs.get("home_win", probs.get(fixture.get("home_short_code", "").upper(), 0)) or 0
            d = probs.get("draw", 0) or 0
            a = probs.get("away_win", probs.get(fixture.get("away_short_code", "").upper(), 0)) or 0
            return f"{h*100:.0f}/{d*100:.0f}/{a*100:.0f}"

        # Find best edge
        best_edge = "—"
        ml = enriched.get("ml_probs", {})
        poly = enriched.get("poly_probs", {})
        if ml and poly:
            home_code = fixture.get("home_short_code", "").upper() or home[:3].upper()
            away_code = fixture.get("away_short_code", "").upper() or away[:3].upper()
            edges = []
            for key, p_key in [("home_win", home_code), ("draw", "draw"), ("away_win", away_code)]:
                ml_p = ml.get(key)
                poly_p = poly.get(p_key)
                if ml_p is not None and poly_p is not None:
                    edges.append(abs(ml_p - poly_p))
            if edges:
                max_edge = max(edges)
                best_edge = f"{max_edge:+.1%}" if max_edge >= 0.02 else "<2%"

        table.add_row(
            str(i),
            f"{home} vs {away}",
            kickoff_str,
            _mini_prob_bar(enriched.get("ml_probs", {})),
            _mini_prob_bar(enriched.get("book_probs", {})),
            _mini_prob_bar(enriched.get("poly_probs", {})),
            _mini_prob_bar(enriched.get("kalshi_probs", {})),
            best_edge,
        )

    return table


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    console.print()
    console.print(
        Panel(
            "[bold]World Cup 2026 — Today's Fixtures Explorer[/bold]\n"
            f"📅 {_dt.datetime.now(timezone.utc).strftime('%A, %B %d %Y')} (UTC)\n"
            "Pulling fixtures and enriching with Sportmonks, Polymarket & Kalshi data…",
            border_style="green",
        )
    )
    console.print()

    # ── Strategy 1: Use the arena mapping endpoint (has kickoff timestamps) ──
    mappings = polymarket.get_all_mappings()
    today_fixtures: list[dict] = []

    if mappings:
        for m in mappings:
            kickoff_ms = m.get("sportmonks_kickoff_utc")
            if not _is_today_ms(kickoff_ms):
                continue
            kickoff_iso, _ = _safe_kickoff(kickoff_ms)
            today_fixtures.append({
                "id": int(m["sportmonks_fixture_id"]),
                "fixture_code": str(m["sportmonks_fixture_id"]),
                "name": m.get("sportmonks_match_name", ""),
                "starting_at": kickoff_iso,
                "home_team_code": m.get("home_short_code", "HOME"),
                "away_team_code": m.get("away_short_code", "AWAY"),
                "home_country": m.get("home_country", m.get("home_short_code", "HOME")),
                "away_country": m.get("away_country", m.get("away_short_code", "AWAY")),
                "home_short_code": m.get("home_short_code", "HOME"),
                "away_short_code": m.get("away_short_code", "AWAY"),
                "polymarket_event_slug": m.get("polymarket_event_slug"),
                "polymarket_home_token_yes": m.get("polymarket_home_token_yes"),
                "polymarket_draw_token_yes": m.get("polymarket_draw_token_yes"),
                "polymarket_away_token_yes": m.get("polymarket_away_token_yes"),
            })

    # ── Strategy 2: Fallback to Sportmonks season schedule ────────────────
    if not today_fixtures:
        try:
            all_fixtures = sportmonks.get_fixtures_by_season()
            today_fixtures = [f for f in all_fixtures if _is_today(f.get("starting_at"))]
        except Exception:
            pass

    if not today_fixtures:
        console.print(
            Panel(
                "[bold yellow]⚠ No fixtures scheduled for today.[/bold yellow]\n\n"
                "This could mean:\n"
                "  • There are no World Cup matches today\n"
                "  • The ARENA_KEY environment variable is not set\n"
                "  • The mapping endpoint returned no results\n\n"
                "Try setting your [bold]STAIR_API_KEY[/bold] in the .env file.",
                border_style="yellow",
            )
        )
        return

    # Sort by kickoff time
    today_fixtures.sort(key=lambda f: f.get("starting_at") or "")

    console.print(f"[dim]Found {len(today_fixtures)} fixture(s) for today. Enriching with data…[/dim]\n")

    # Enrich each fixture
    enriched_list: list[dict] = []
    for fixture in today_fixtures:
        with console.status(f"[bold green]Fetching data for {fixture.get('home_country', '?')} vs {fixture.get('away_country', '?')}…"):
            enriched = enrich_fixture(fixture)
            enriched_list.append(enriched)

    # ── Render summary table ──────────────────────────────────────────────
    summary = render_summary_table(today_fixtures, enriched_list)
    console.print(summary)
    console.print()

    # ── Render detailed cards ─────────────────────────────────────────────
    console.print("[bold white]━" * 70)
    console.print("[bold]Detailed Fixture Cards[/bold]")
    console.print("[bold white]━" * 70)
    console.print()

    for fixture, enriched in zip(today_fixtures, enriched_list):
        card = render_fixture_card(fixture, enriched)
        console.print(card)
        console.print()

    # ── Legend ────────────────────────────────────────────────────────────
    legend = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    legend.add_column()
    legend.add_column()
    legend.add_row("[bold green]■[/bold green] High probability (>60%)", "[bold]Legend:[/bold]")
    legend.add_row("[bold yellow]■[/bold yellow] Medium probability (30-60%)", "")
    legend.add_row("[bold red]■[/bold red] Low probability (<30%)", "")
    legend.add_row("[dim]—[/dim] Data not available", "")
    console.print(legend)
    console.print()


if __name__ == "__main__":
    main()
