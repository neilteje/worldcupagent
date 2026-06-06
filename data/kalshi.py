"""
Kalshi prediction-market odds for a fixture (a second market to triangulate
against Polymarket).

Kalshi's trade API v2 serves public market reads with no auth. We locate the
3-way winner markets for a fixture by scanning open markets and matching the two
team names in the market title/subtitle, then convert YES bid/ask cents into a
mid probability.

The signal that matters downstream is the *cross-market spread*: when Kalshi and
Polymarket agree, that's confirmation; when they diverge, that's a risk flag.

Everything fails soft — Kalshi frequently has no market for a given fixture, in
which case we return None mids and the agent proceeds on Polymarket alone.
"""
from __future__ import annotations
import httpx
import config

_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_TIMEOUT = config.RESEARCH_TIMEOUT_SECONDS
_MAX_PAGES = 4
_PAGE_SIZE = 1000


def _headers() -> dict:
    h = {"Accept": "application/json"}
    if config.KALSHI_API_KEY:
        h["Authorization"] = f"Bearer {config.KALSHI_API_KEY}"
    return h


def _scan_open_markets() -> list[dict]:
    """Bounded scan of open markets across a few cursor pages. Fails soft to []."""
    markets: list[dict] = []
    cursor = None
    try:
        for _ in range(_MAX_PAGES):
            params = {"status": "open", "limit": _PAGE_SIZE}
            if cursor:
                params["cursor"] = cursor
            resp = httpx.get(f"{_BASE}/markets", headers=_headers(),
                             params=params, timeout=_TIMEOUT)
            if not resp.is_success:
                break
            body = resp.json()
            markets.extend(body.get("markets") or [])
            cursor = body.get("cursor")
            if not cursor:
                break
    except Exception:
        return markets
    return markets


def _yes_mid(market: dict) -> float | None:
    """Mid of YES bid/ask in probability units (Kalshi quotes cents 1..99)."""
    bid = market.get("yes_bid")
    ask = market.get("yes_ask")
    if bid in (None, 0) and ask in (None, 0):
        last = market.get("last_price")
        return float(last) / 100.0 if last else None
    bid = bid or ask
    ask = ask or bid
    return round((float(bid) + float(ask)) / 200.0, 4)


def _text(market: dict) -> str:
    return " ".join(
        str(market.get(k, "")) for k in ("title", "subtitle", "yes_sub_title", "ticker")
    ).lower()


def search_fixture_markets(home: str, away: str, strict: bool = True) -> list[dict]:
    """
    Open Kalshi markets for this exact fixture.

    strict=True requires BOTH team names in the market text — this filters out
    the many geopolitical/parlay markets on the elections host that merely
    mention a single country name. A clean head-to-head match market names both
    sides (and a single-game soccer market is a small 2-3 outcome set, not a
    20-team futures parlay).
    """
    h, a = home.lower(), away.lower()
    out = []
    for m in _scan_open_markets():
        t = _text(m)
        has_h, has_a = h in t, a in t
        if (has_h and has_a) if strict else (has_h or has_a):
            # Reject obvious multi-game/futures parlays (lots of "yes <team>" legs).
            if t.count("yes ") >= 4:
                continue
            out.append({
                "ticker": m.get("ticker"),
                "title": m.get("title"),
                "subtitle": m.get("subtitle") or m.get("yes_sub_title"),
                "yes_mid": _yes_mid(m),
                "text": t,
            })
    return out


def get_moneyline(home: str, away: str) -> dict:
    """
    Best-effort 3-way moneyline from Kalshi for the exact fixture.

    Returns {"home": float|None, "draw": float|None, "away": float|None,
             "markets_found": int}. Mids are YES probabilities for each outcome.
    Returns all-None when Kalshi has no clean market for the pairing (common for
    fixtures far in the future) — the agent then proceeds on Polymarket alone.
    """
    candidates = search_fixture_markets(home, away, strict=True)
    h, a = home.lower(), away.lower()
    result = {"home": None, "draw": None, "away": None, "markets_found": len(candidates)}

    for c in candidates:
        text = c["text"]
        sub = (c["subtitle"] or "").lower()
        mid = c["yes_mid"]
        if mid is None:
            continue
        # The YES side resolves for whichever outcome the subtitle names.
        if "draw" in sub or "tie" in sub:
            result["draw"] = result["draw"] or mid
        elif h in sub:
            result["home"] = result["home"] or mid
        elif a in sub:
            result["away"] = result["away"] or mid
    return result


def cross_market_signal(pm_mid: float | None, kalshi_mid: float | None) -> dict:
    """
    Compare our outcome's Polymarket vs Kalshi mid.

    Returns {"available": bool, "spread": float|None,
             "agreement": "consensus"|"contested"|"normal"|"n/a"}.
    """
    if pm_mid is None or kalshi_mid is None:
        return {"available": False, "spread": None, "agreement": "n/a"}
    spread = abs(float(pm_mid) - float(kalshi_mid))
    if spread <= config.MARKET_CONSENSUS_SPREAD:
        agreement = "consensus"
    elif spread >= config.MARKET_CONTESTED_SPREAD:
        agreement = "contested"
    else:
        agreement = "normal"
    return {"available": True, "spread": round(spread, 4), "agreement": agreement}
