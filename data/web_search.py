"""
Targeted pre-match web research — injury news, lineup confirmations, previews.

This is deliberately narrow: every query is a fixed template built from the two
team names and the match date. No open-ended browsing. The output is a flat list
of {title, snippet, url, query} result objects that an LLM scout reads to flag
availability/form signals.

Primary backend  : Serper (serper.dev) — Google results as JSON, needs SERPER_API_KEY.
Fallback backend : DuckDuckGo Instant Answer (no key) — thinner, but free.

Every public function fails soft: on any error it returns [] / {} so the agent
run never aborts because the web was unreachable.
"""
from __future__ import annotations
import httpx
import config

_SERPER_URL = "https://google.serper.dev/search"
_DDG_URL = "https://api.duckduckgo.com/"

_INJURY_TEMPLATES = (
    "{home} vs {away} injury news {date}",
    "{home} team news injuries {date}",
    "{away} team news injuries {date}",
)
_LINEUP_TEMPLATES = (
    "{home} predicted lineup {date}",
    "{away} predicted lineup {date}",
)
_PREVIEW_TEMPLATES = (
    "{home} vs {away} match preview prediction {date}",
)


def _serper_search(query: str, num: int = 5) -> list[dict]:
    if not config.SERPER_API_KEY:
        return []
    resp = httpx.post(
        _SERPER_URL,
        headers={"X-API-KEY": config.SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": query, "num": num},
        timeout=config.RESEARCH_TIMEOUT_SECONDS,
    )
    if not resp.is_success:
        return []
    organic = resp.json().get("organic") or []
    return [
        {
            "title": r.get("title", ""),
            "snippet": r.get("snippet", ""),
            "url": r.get("link", ""),
            "query": query,
        }
        for r in organic[:num]
    ]


def _ddg_search(query: str, num: int = 5) -> list[dict]:
    """DuckDuckGo Instant Answer fallback. Shallow but keyless."""
    resp = httpx.get(
        _DDG_URL,
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        timeout=config.RESEARCH_TIMEOUT_SECONDS,
    )
    if not resp.is_success:
        return []
    data = resp.json()
    out: list[dict] = []
    abstract = data.get("AbstractText")
    if abstract:
        out.append({
            "title": data.get("Heading", query),
            "snippet": abstract,
            "url": data.get("AbstractURL", ""),
            "query": query,
        })
    for topic in (data.get("RelatedTopics") or []):
        if len(out) >= num:
            break
        text = topic.get("Text")
        if text:
            out.append({
                "title": text[:80],
                "snippet": text,
                "url": topic.get("FirstURL", ""),
                "query": query,
            })
    return out


def _search(query: str, num: int = 5) -> list[dict]:
    try:
        results = _serper_search(query, num)
        if results:
            return results
        return _ddg_search(query, num)
    except Exception:
        return []


def _run_templates(templates, home: str, away: str, date: str, per_query: int) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for tmpl in templates:
        for r in _search(tmpl.format(home=home, away=away, date=date), per_query):
            key = r.get("url") or r.get("snippet")
            if key and key not in seen:
                seen.add(key)
                out.append(r)
    return out


def fetch_injury_news(home: str, away: str, match_date: str, per_query: int = 4) -> list[dict]:
    return _run_templates(_INJURY_TEMPLATES, home, away, match_date, per_query)


def fetch_lineup_news(home: str, away: str, match_date: str, per_query: int = 4) -> list[dict]:
    return _run_templates(_LINEUP_TEMPLATES, home, away, match_date, per_query)


def fetch_previews(home: str, away: str, match_date: str, per_query: int = 4) -> list[dict]:
    return _run_templates(_PREVIEW_TEMPLATES, home, away, match_date, per_query)


def gather_research(
    home: str,
    away: str,
    match_date: str,
    have_confirmed_lineups: bool = False,
) -> dict:
    """
    One-shot bundle for the agent. When Sportmonks already exposes confirmed
    lineups we skip the lineup queries — the structured API data beats scraped
    headlines and we save quota.

    Returns:
      {
        "backend": "serper" | "ddg" | "none",
        "injuries": [result, ...],
        "lineups":  [result, ...],
        "previews": [result, ...],
        "total_results": int,
      }
    """
    backend = "serper" if config.SERPER_API_KEY else "ddg"
    injuries = fetch_injury_news(home, away, match_date)
    lineups = [] if have_confirmed_lineups else fetch_lineup_news(home, away, match_date)
    previews = fetch_previews(home, away, match_date)
    total = len(injuries) + len(lineups) + len(previews)
    return {
        "backend": backend if total else "none",
        "injuries": injuries,
        "lineups": lineups,
        "previews": previews,
        "total_results": total,
    }
