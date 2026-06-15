"""
Polymarket data via the arena's Gamma and CLOB proxies.

Two upstream APIs:
  Gamma  — /api/v1/data/proxy/polymarket-gamma/events?slug=...
           Returns the event + 3 child winner markets (conditionId + clobTokenIds)
  CLOB   — /api/v1/data/proxy/polymarket-clob/midpoint?token_id=...
           Returns the live mid price for a YES token

Both have the arena envelope: response.json()["body"] is the real payload.

Flow:
  1. GET /api/v1/web/mapping?fixture_id=...  → polymarket_event_slug
  2. GET gamma/events?slug=...               → event + markets
  3. GET clob/midpoint?token_id=...          → mid price per YES token (×3)
  4. Assemble moneyline dict
"""
from __future__ import annotations
import json
import re
import httpx
from typing import Any
import config
from data.team_codes import fifa_code

_ARENA         = config.ARENA_BASE
_ARENA_API     = config.ARENA_API
_GAMMA         = f"{_ARENA_API}/v1/data/proxy/polymarket-gamma"
_CLOB          = f"{_ARENA_API}/v1/data/proxy/polymarket-clob"
_HEADERS       = {"x-api-key": config.ARENA_KEY}

_TICKER_RE = re.compile(r"^fifwc-([a-z]{2,4})-([a-z]{2,4})-(\d{4}-\d{2}-\d{2})$")
# Generic 3-way soccer ticker: any prefix (fifwc-, fif-, …), home-away-date.
_TICKER_RE_GENERIC = re.compile(r"^([a-z]+)-([a-z]{2,4})-([a-z]{2,4})-(\d{4}-\d{2}-\d{2})$")


# ── Mapping ────────────────────────────────────────────────────────────────

def get_event_slug(fixture_id: int) -> str | None:
    """
    Look up the Polymarket event slug for a Sportmonks fixture id.
    GET /api/v1/web/mapping?fixture_id=...
    Returns the polymarket_event_slug string or None if not mapped.
    """
    resp = httpx.get(
        f"{_ARENA_API}/v1/web/mapping",
        headers=_HEADERS,
        params={"fixture_id": fixture_id},
        timeout=15,
    )
    resp.raise_for_status()
    mappings = resp.json().get("mappings") or []
    if not mappings:
        return None
    return mappings[0].get("polymarket_event_slug")


# ── Gamma + CLOB ───────────────────────────────────────────────────────────

def _clob_mid(token_id: str) -> float | None:
    """Single CLOB midpoint call. Returns None on failure."""
    if not token_id:
        return None
    try:
        resp = httpx.get(
            f"{_CLOB}/midpoint",
            headers=_HEADERS,
            params={"token_id": token_id},
            timeout=10,
        )
        if not resp.is_success:
            return None
        body = resp.json().get("body")
        if isinstance(body, dict) and "mid" in body:
            return float(body["mid"])
    except Exception:
        pass
    return None


def _outcome_from_slug(market_slug: str, ticker: str,
                       home_code: str, away_code: str) -> str | None:
    """Map 'fifwc-mex-rsa-2026-06-11-mex' → 'home' | 'draw' | 'away'."""
    if not market_slug.startswith(ticker + "-"):
        return None
    suffix = market_slug[len(ticker) + 1:]
    if suffix == home_code:
        return "home"
    if suffix == "draw":
        return "draw"
    if suffix == away_code:
        return "away"
    return None


def get_moneyline(fixture_id: int) -> dict | None:
    """
    Build the 3-way moneyline dict for a fixture, mirroring the notebook flow.

    Returns:
        {
          "sportmonks_match_id": int,
          "fixture": str,
          "kickoff_utc": str,
          "polymarket_event_slug": str,
          "outcomes": {
            "home": {"team_code": str, "condition_id": str, "token_yes": str, "current_mid_yes": float|None},
            "draw": {...},
            "away": {...},
          }
        }
        or None if no mapping / no event found.
    """
    slug = get_event_slug(fixture_id)
    if not slug:
        return None

    # Gamma: fetch event + child markets
    resp = httpx.get(
        f"{_GAMMA}/events",
        headers=_HEADERS,
        params={"slug": slug},
        timeout=15,
    )
    resp.raise_for_status()
    events = resp.json().get("body") or []
    event = events[0] if events else None
    if not event:
        return None

    ticker = (event.get("ticker") or "").lower()
    m = _TICKER_RE.match(ticker)
    if not m:
        return None

    pm_home, pm_away, _ = m.groups()
    outcomes: dict[str, dict] = {}

    for mkt in (event.get("markets") or []):
        key = _outcome_from_slug(
            (mkt.get("slug") or "").lower(), ticker, pm_home, pm_away
        )
        if key is None:
            continue
        try:
            token_ids = json.loads(mkt.get("clobTokenIds") or "[]")
        except json.JSONDecodeError:
            token_ids = []
        token_yes = token_ids[0] if token_ids else None
        outcomes[key] = {
            "team_code":       "draw" if key == "draw" else (
                                    fifa_code(pm_home) if key == "home" else fifa_code(pm_away)),
            "condition_id":    mkt.get("conditionId"),
            "token_yes":       token_yes,
            "current_mid_yes": _clob_mid(token_yes),
        }

    return {
        "sportmonks_match_id":   fixture_id,
        "fixture":               event.get("title"),
        "kickoff_utc":           event.get("startDate"),
        "polymarket_event_slug": slug,
        "outcomes":              outcomes,
    }


def get_moneyline_by_slug(slug: str) -> dict | None:
    """
    Build the 3-way moneyline from a Polymarket event slug directly — works for
    ANY product line (World Cup `fifwc-…`, friendlies `fif-…`, etc.), bypassing
    the arena's curated fixture→slug mapping.

    Returns the same shape as get_moneyline(), or None if the event isn't found.
    """
    if not slug:
        return None

    resp = httpx.get(f"{_GAMMA}/events", headers=_HEADERS,
                     params={"slug": slug}, timeout=15)
    resp.raise_for_status()
    events = resp.json().get("body") or []
    event = events[0] if events else None
    if not event:
        return None

    ticker = (event.get("ticker") or "").lower()
    m = _TICKER_RE_GENERIC.match(ticker)
    if not m:
        return None
    _prefix, pm_home, pm_away, _date = m.groups()

    outcomes: dict[str, dict] = {}
    for mkt in (event.get("markets") or []):
        key = _outcome_from_slug((mkt.get("slug") or "").lower(),
                                 ticker, pm_home, pm_away)
        if key is None:
            continue
        try:
            token_ids = json.loads(mkt.get("clobTokenIds") or "[]")
        except json.JSONDecodeError:
            token_ids = []
        token_yes = token_ids[0] if token_ids else None
        outcomes[key] = {
            "team_code":       "draw" if key == "draw" else (
                                    fifa_code(pm_home) if key == "home" else fifa_code(pm_away)),
            "condition_id":    mkt.get("conditionId"),
            "token_yes":       token_yes,
            "current_mid_yes": _clob_mid(token_yes),
        }

    return {
        "sportmonks_match_id":   None,
        "fixture":               event.get("title"),
        "kickoff_utc":           event.get("startDate"),
        "polymarket_event_slug": slug,
        "outcomes":              outcomes,
    }


def extract_implied_probs(moneyline: dict) -> dict[str, float]:
    """
    Pull the three mid prices out of a moneyline dict.
    Returns {home_team_code: p, 'draw': p, away_team_code: p}, normalised.
    """
    outcomes = moneyline.get("outcomes") or {}
    probs: dict[str, float] = {}
    for key in ("home", "draw", "away"):
        entry = outcomes.get(key, {})
        mid = entry.get("current_mid_yes")
        team = entry.get("team_code", key)
        if mid is not None:
            probs[team] = float(mid)
    # Normalise
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    return probs

# ── Robust strategy-agent helpers (three-way home/draw/away keys) ──────────
def _body(resp_json):
    return resp_json.get("body", resp_json) if isinstance(resp_json, dict) else resp_json


def get_mapping(fixture_id: int | str) -> dict | None:
    try:
        r = httpx.get(f"{_ARENA_API}/v1/web/mapping", headers=_HEADERS, params={"fixture_id": fixture_id}, timeout=15)
        if not r.is_success: return None
        mappings = r.json().get("mappings") or []
        return mappings[0] if mappings else None
    except Exception:
        return None


def get_gamma_event(slug: str) -> dict | None:
    try:
        r = httpx.get(f"{_GAMMA}/events", headers=_HEADERS, params={"slug": slug}, timeout=15)
        if not r.is_success: return None
        body = _body(r.json()) or []
        return body[0] if isinstance(body, list) and body else body if isinstance(body, dict) else None
    except Exception:
        return None


def extract_yes_token_ids(event: dict) -> dict[str, str | None]:
    tokens = {"home": None, "draw": None, "away": None}
    markets = event.get("markets") or event.get("data", {}).get("markets") or []
    ticker = (event.get("ticker") or event.get("slug") or "").lower()
    parsed = _TICKER_RE.match(ticker)
    home_code, away_code = (parsed.group(1), parsed.group(2)) if parsed else ("home", "away")
    for mkt in markets:
        slug = (mkt.get("slug") or mkt.get("market_slug") or "").lower()
        title = (mkt.get("question") or mkt.get("title") or "").lower()
        key = _outcome_from_slug(slug, ticker, home_code, away_code) if ticker else None
        if key is None:
            if "draw" in slug or "draw" in title: key = "draw"
            elif home_code in slug or " home" in title: key = "home"
            elif away_code in slug or " away" in title: key = "away"
        if key in tokens:
            raw = mkt.get("clobTokenIds") or mkt.get("clob_token_ids") or mkt.get("tokens") or []
            if isinstance(raw, str):
                try: raw = json.loads(raw)
                except json.JSONDecodeError: raw = [raw]
            if raw and isinstance(raw[0], dict): token = raw[0].get("token_id") or raw[0].get("id")
            else: token = raw[0] if raw else None
            tokens[key] = str(token) if token else None
    return tokens


def get_midpoint(token_id: str) -> float | None:
    return _clob_mid(str(token_id))


def get_three_way_from_tokens(home_token: str | None, draw_token: str | None, away_token: str | None, slug: str = "") -> dict:
    """
    Fetch CLOB midpoints for three pre-known YES token IDs and return a normalised
    market probability dict.  This is the fast path when token IDs come from the
    arena mapping endpoint directly (no Gamma API call needed).
    """
    tokens = {"home": home_token, "draw": draw_token, "away": away_token}
    if not all(tokens.values()):
        return {"complete": False, "raw_midpoints": {}, "normalized_probs": None,
                "reason": "missing_token_ids", "slug": slug}
    raw: dict[str, float | None] = {k: _clob_mid(str(v)) for k, v in tokens.items()}
    if any(raw.get(k) is None for k in ("home", "draw", "away")):
        return {"complete": False, "raw_midpoints": raw, "normalized_probs": None,
                "tokens": tokens, "slug": slug, "reason": "clob_midpoint_missing"}
    total = sum(float(raw[k]) for k in ("home", "draw", "away"))  # type: ignore[arg-type]
    if not total:
        return {"complete": False, "raw_midpoints": raw, "normalized_probs": None,
                "tokens": tokens, "slug": slug, "reason": "zero_total"}
    norm = {k: float(raw[k]) / total for k in ("home", "draw", "away")}  # type: ignore[arg-type]
    return {"complete": True, "raw_midpoints": raw, "normalized_probs": norm,
            "tokens": tokens, "slug": slug, "reason": "ok"}


def get_three_way_market_probs(fixture_id: int | str, tokens: dict | None = None) -> dict:
    """
    Return normalised 3-way (home/draw/away) market probabilities for a fixture.

    Fast path: if caller supplies a ``tokens`` dict with keys
    ``home``, ``draw``, ``away`` (token ID strings from the arena mapping),
    skip the Gamma API entirely and go straight to the CLOB.

    Slow path: look up the mapping, then try Gamma (deprecated -- may return empty).
    """
    # Fast path: caller already has token IDs from the mapping
    if tokens and all(tokens.get(k) for k in ("home", "draw", "away")):
        return get_three_way_from_tokens(
            tokens["home"], tokens["draw"], tokens["away"],
            slug=tokens.get("slug", ""),
        )

    # Slow path: mapping lookup + Gamma (deprecated but kept as fallback)
    mapping = get_mapping(fixture_id)
    if mapping:
        # Mapping already contains the token IDs -- use fast path
        home_t = mapping.get("polymarket_home_token_yes")
        draw_t = mapping.get("polymarket_draw_token_yes")
        away_t = mapping.get("polymarket_away_token_yes")
        slug   = mapping.get("polymarket_event_slug", "")
        if home_t and draw_t and away_t:
            return get_three_way_from_tokens(home_t, draw_t, away_t, slug=slug)

    slug = (mapping or {}).get("polymarket_event_slug") or (mapping or {}).get("slug")
    if not slug:
        return {"complete": False, "raw_midpoints": {}, "normalized_probs": None, "reason": "mapping_missing"}
    event = get_gamma_event(slug)
    if not event:
        return {"complete": False, "raw_midpoints": {}, "normalized_probs": None, "reason": "gamma_event_missing", "slug": slug}
    extracted = extract_yes_token_ids(event)
    raw: dict[str, float | None] = {k: get_midpoint(v) if v else None for k, v in extracted.items()}
    if any(raw.get(k) is None for k in ("home", "draw", "away")):
        return {"complete": False, "raw_midpoints": raw, "normalized_probs": None, "tokens": extracted, "slug": slug, "reason": "midpoint_missing"}
    total = sum(float(raw[k]) for k in ("home", "draw", "away"))  # type: ignore[arg-type]
    norm = {k: float(raw[k]) / total for k in ("home", "draw", "away")} if total else None  # type: ignore[arg-type]
    return {"complete": bool(norm), "raw_midpoints": raw, "normalized_probs": norm, "tokens": extracted, "slug": slug, "reason": "ok" if norm else "zero_total"}


def get_all_mappings() -> list[dict]:
    """
    Return all arena fixture mappings (all 72 WC2026 group-stage fixtures).
    Includes Sportmonks fixture IDs, team codes, token IDs, and kickoff times.
    Call with no params -- the endpoint returns the complete list.
    """
    try:
        r = httpx.get(f"{_ARENA_API}/v1/web/mapping", headers=_HEADERS, timeout=15)
        if not r.is_success:
            return []
        return r.json().get("mappings") or []
    except Exception:
        return []
