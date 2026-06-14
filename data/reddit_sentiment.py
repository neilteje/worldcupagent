"""
Reddit crowd-sentiment scraping for a fixture.

r/soccer match-preview and discussion threads carry high-signal fan analysis.
Reddit serves JSON for any URL when you append `.json` — no API key, no OAuth.

We fetch the most relevant threads, pull their top-voted comments, and return
the raw comment text. We deliberately do NOT score sentiment here — that's the
LLM scout's job. We only attach cheap structural signals (mention counts).

All functions fail soft: on any error they return empty structures so a Reddit
outage never aborts the agent run.
"""
from __future__ import annotations
import httpx
import config

# Reddit 403s user-agents containing "bot"/"python"; use a realistic browser UA.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
_SEARCH_URL = "https://www.reddit.com/r/soccer/search.json"
_TIMEOUT = config.RESEARCH_TIMEOUT_SECONDS
_TEAM_ALIASES = {
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
}


def _alias(name: str) -> str:
    return _TEAM_ALIASES.get(str(name).strip().lower(), name)


def search_reddit_threads(home: str, away: str, limit: int = 3) -> list[dict]:
    """Top r/soccer threads for the fixture, ranked by relevance then score."""
    try:
        resp = httpx.get(
            _SEARCH_URL,
            headers=_HEADERS,
            params={
                "q": f"{_alias(home)} {_alias(away)}",
                "restrict_sr": 1,
                "sort": "relevance",
                "t": "month",
                "limit": 10,
            },
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        if not resp.is_success:
            return []
        children = (resp.json().get("data") or {}).get("children") or []
    except Exception:
        return []

    threads = []
    for c in children:
        d = c.get("data") or {}
        threads.append({
            "title": d.get("title", ""),
            "permalink": d.get("permalink", ""),
            "score": int(d.get("score") or 0),
            "num_comments": int(d.get("num_comments") or 0),
        })
    threads.sort(key=lambda t: t["score"], reverse=True)
    return threads[:limit]


def fetch_top_comments(permalink: str, limit: int = 20) -> list[str]:
    """Top-voted comments for a thread permalink (markdown stripped to plain text)."""
    if not permalink:
        return []
    try:
        resp = httpx.get(
            f"https://www.reddit.com{permalink}.json",
            headers=_HEADERS,
            params={"sort": "top", "limit": limit},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        if not resp.is_success:
            return []
        listings = resp.json()
    except Exception:
        return []

    if not isinstance(listings, list) or len(listings) < 2:
        return []

    comments: list[str] = []
    for c in (listings[1].get("data") or {}).get("children") or []:
        if c.get("kind") != "t1":
            continue
        body = (c.get("data") or {}).get("body")
        if body and body not in ("[deleted]", "[removed]"):
            comments.append(" ".join(body.split()))
        if len(comments) >= limit:
            break
    return comments


def _search_fallback(home: str, away: str, limit: int = 8) -> list[str]:
    """
    Reddit blocks unauthenticated JSON from many IPs (403). When the direct API
    is unavailable we surface r/soccer discussion via the web-search backend
    (site:reddit.com), returning result snippets as pseudo-comments.
    """
    try:
        from data import web_search
    except Exception:
        return []
    home_q, away_q = _alias(home), _alias(away)
    queries = [
        f"site:reddit.com/r/soccer {home_q} vs {away_q} match thread",
        f"site:reddit.com/r/soccer {home_q} {away_q} World Cup",
    ]
    results = []
    seen = set()
    for query in queries:
        for r in web_search._search(query, num=limit):
            key = r.get("url") or r.get("snippet")
            if key and key not in seen:
                seen.add(key)
                results.append(r)
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break
    return [f"{r.get('title','')} — {r.get('snippet','')}".strip(" —")
            for r in results if r.get("snippet")]


def get_sentiment_bundle(home: str, away: str, max_comments: int = 25) -> dict:
    """
    Aggregate raw crowd chatter for the LLM scout.

    Returns:
      {
        "source": "reddit_api" | "web_search" | "none",
        "threads_found": int,        # direct Reddit threads only
        "comments_found": int,       # direct comments or fallback snippets
        "threads": [{title, score, num_comments}, ...],
        "top_comments": [str, ...],     # raw, for the LLM to interpret
        "home_mentions": int,
        "away_mentions": int,
      }
    """
    threads = search_reddit_threads(home, away)
    comments: list[str] = []
    for t in threads:
        if len(comments) >= max_comments:
            break
        comments.extend(fetch_top_comments(t["permalink"], limit=max_comments))
    comments = comments[:max_comments]
    source = "reddit_api" if comments else "none"

    if not comments:
        fb = _search_fallback(home, away)
        if fb:
            comments = fb[:max_comments]
            source = "web_search"

    blob = " ".join(comments).lower()
    return {
        "source": source,
        "threads_found": len(threads),
        "comments_found": len(comments),
        "threads": [{k: t[k] for k in ("title", "score", "num_comments")} for t in threads],
        "top_comments": comments,
        "home_mentions": blob.count(home.lower()),
        "away_mentions": blob.count(away.lower()),
    }
