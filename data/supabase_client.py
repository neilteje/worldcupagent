"""
Stair AI aggregated dataset via Supabase PostgREST.

Two schemas:
  public           — catalog_full, catalog_tables, catalog_columns
  world_cup_arena  — priors (ads_a_*) + live checkpoints (d_*)

IMPORTANT: Team identifiers differ between systems.
  Sportmonks uses its own numeric ids (e.g. Mexico = 458, South Africa = 146).
  This database (StatsBomb-derived) uses country_id numbers (e.g. Mexico = 147, South Africa = 211).
  Always filter by country_id, NOT by Sportmonks team id.

Available tables (from catalog):
  PRIORS (ads_a_*):
    ads_a_country_struct  — formation, tactical approach, recent tournament record
    ads_a_country_style   — set-piece efficiency, group/KO goals-per-game
    ads_a_h2h_continent   — head-to-head by continent pair
    ads_a_h2h_country     — head-to-head by country pair (most useful)
    ads_a_ko_pattern      — knockout stage performance pattern
    ads_a_manager         — manager career record
    ads_a_special_match   — historical performance in special scenarios
    ads_a_stage_record    — performance by tournament stage

  LIVE CHECKPOINTS (d_*):
    d_checkpoint_snapshot — per-team per-match at HT/FT/ET1/ET2
    d_match_scores        — score at each checkpoint
    dim_match             — match dimension table
"""
from __future__ import annotations
import httpx
import re
import unicodedata
from typing import Any
import config

_BASE    = config.SUPABASE_URL         # https://ezvbmtvrvzageqixvdak.supabase.co/rest/v1
_H_PUB   = {"apikey": config.SUPABASE_KEY}
_H_ARENA = {"apikey": config.SUPABASE_KEY, "Accept-Profile": "world_cup_arena"}


def _get(table: str, params: dict | None = None, arena: bool = True) -> list[dict]:
    headers = _H_ARENA if arena else _H_PUB
    resp = httpx.get(
        f"{_BASE}/{table}",
        headers=headers,
        params=params or {},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ── Catalog ────────────────────────────────────────────────────────────────

def get_catalog() -> list[dict]:
    """All tables with descriptions and row counts."""
    return _get(
        "catalog_full",
        params={"select": "table_name,category,row_count,table_description",
                "order":  "category,table_name"},
        arena=False,
    )


# ── Country-name → StatsBomb country_id resolution ──────────────────────────
# The priors tables key on StatsBomb country_id (Mexico=147), which differs from
# Sportmonks team ids (Mexico=458). The ads_a_h2h_country table carries BOTH the
# id and the name on each side, so we build a name→id map from it once and cache.

_COUNTRY_ID_MAP: dict[str, int] | None = None

# Hand-tuned aliases for names that differ between Sportmonks and StatsBomb.
_NAME_ALIASES = {
    "usa": "united states",
    "united states of america": "united states",
    "south korea": "korea republic",
    "north korea": "korea dpr",
    "cote d ivoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "côte d'ivoire": "ivory coast",
    "iran": "iran",
    "ir iran": "iran",
    "czechia": "czech republic",
}

# Countries expected in WC fixtures but absent from the current StatsBomb prior
# snapshot. Treat these as coverage gaps, not resolver surprises.
_KNOWN_MISSING_COUNTRIES = {
    "cote d ivoire",
    "cote d'ivoire",
    "côte d'ivoire",
    "ivory coast",
}


def _norm_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


def _build_country_id_map() -> dict[str, int]:
    rows = _get("ads_a_h2h_country",
                params={"select": "country_id_a,country_name_a,country_id_b,country_name_b"})
    m: dict[str, int] = {}
    for r in rows:
        if r.get("country_name_a") and r.get("country_id_a") is not None:
            name = r["country_name_a"].strip().lower()
            m[name] = int(r["country_id_a"])
            m[_norm_name(name)] = int(r["country_id_a"])
        if r.get("country_name_b") and r.get("country_id_b") is not None:
            name = r["country_name_b"].strip().lower()
            m[name] = int(r["country_id_b"])
            m[_norm_name(name)] = int(r["country_id_b"])
    return m


def known_missing_country(team_name: str) -> bool:
    """True when the current Supabase prior snapshot is known not to cover it."""
    name = _norm_name(team_name)
    alias = _NAME_ALIASES.get(name, name)
    return name in _KNOWN_MISSING_COUNTRIES or alias in _KNOWN_MISSING_COUNTRIES


def resolve_country_id(team_name: str) -> int | None:
    """
    Map a Sportmonks team name to its StatsBomb country_id, or None if unknown.
    Tries exact match, alias table, then a loose contains-match.
    """
    global _COUNTRY_ID_MAP
    if not team_name:
        return None
    if _COUNTRY_ID_MAP is None:
        try:
            _COUNTRY_ID_MAP = _build_country_id_map()
        except Exception:
            _COUNTRY_ID_MAP = {}

    name = _norm_name(team_name)
    if name in _COUNTRY_ID_MAP:
        return _COUNTRY_ID_MAP[name]
    if name in _NAME_ALIASES and _NAME_ALIASES[name] in _COUNTRY_ID_MAP:
        return _COUNTRY_ID_MAP[_NAME_ALIASES[name]]
    for known, cid in _COUNTRY_ID_MAP.items():
        if name in known or known in name:
            return cid
    return None


# ── Priors — multi-table fetch ─────────────────────────────────────────────

def get_country_style(country_id_a: int, country_id_b: int) -> list[dict]:
    """Set-piece efficiency, group/KO goals-per-game (ads_a_country_style)."""
    return _get("ads_a_country_style",
                params={"country_id": f"in.({country_id_a},{country_id_b})", "select": "*"})


def get_country_struct(country_id_a: int, country_id_b: int) -> list[dict]:
    """Tactical/formation priors (ads_a_country_struct)."""
    return _get("ads_a_country_struct",
                params={"country_id": f"in.({country_id_a},{country_id_b})", "select": "*"})


def get_h2h_country(country_id_a: int, country_id_b: int) -> list[dict]:
    """Head-to-head record between two countries (ads_a_h2h_country)."""
    rows = _get("ads_a_h2h_country",
                params={"country_id_a": f"eq.{country_id_a}",
                        "country_id_b": f"eq.{country_id_b}",
                        "select": "*"})
    if not rows:
        rows = _get("ads_a_h2h_country",
                    params={"country_id_a": f"eq.{country_id_b}",
                            "country_id_b": f"eq.{country_id_a}",
                            "select": "*"})
    return rows


def get_ko_pattern(country_id_a: int, country_id_b: int) -> list[dict]:
    """Knockout stage performance (ads_a_ko_pattern)."""
    return _get("ads_a_ko_pattern",
                params={"country_id": f"in.({country_id_a},{country_id_b})", "select": "*"})


def get_stage_record(country_id_a: int, country_id_b: int) -> list[dict]:
    """Per-stage performance record (ads_a_stage_record)."""
    return _get("ads_a_stage_record",
                params={"country_id": f"in.({country_id_a},{country_id_b})", "select": "*"})


def get_all_priors(country_id_a: int, country_id_b: int) -> dict[str, list[dict]]:
    """
    Fetch all relevant priors tables for both teams.

    IMPORTANT: StatsBomb country_ids differ from Sportmonks ids.
    We fetch ALL rows from smaller tables and let the calling code (LLM)
    identify the relevant rows. For larger tables we query by country_id
    and retry with all rows if empty.
    """
    results: dict[str, Any] = {}

    for table in ["ads_a_country_style", "ads_a_country_struct",
                  "ads_a_ko_pattern", "ads_a_stage_record"]:
        try:
            rows = _get(table,
                        params={"country_id": f"in.({country_id_a},{country_id_b})",
                                "select": "*"})
            if not rows:
                # Fallback: fetch all (tables are small, ≤200 rows)
                rows = _get(table, params={"select": "*"})
            results[table] = rows
        except Exception as e:
            results[table] = {"error": str(e)}

    # H2H — try both orderings
    for table in ["ads_a_h2h_country"]:
        try:
            rows = _get(table, params={"country_id_a": f"eq.{country_id_a}",
                                       "country_id_b": f"eq.{country_id_b}",
                                       "select": "*"})
            if not rows:
                rows = _get(table, params={"country_id_a": f"eq.{country_id_b}",
                                           "country_id_b": f"eq.{country_id_a}",
                                           "select": "*"})
            results[table] = rows
        except Exception as e:
            results[table] = {"error": str(e)}

    return results


# ── Live checkpoints ───────────────────────────────────────────────────────

def get_ht_snapshot(sportmonks_fixture_id: int) -> list[dict]:
    """
    Half-time snapshot: per-team goals, shots, xG, possession, cards.
    d_checkpoint_snapshot WHERE match_id = X AND checkpoint = 'HT'
    """
    return _get("d_checkpoint_snapshot",
                params={"sportmonks_fixture_id": f"eq.{sportmonks_fixture_id}",
                        "checkpoint": "eq.HT",
                        "select": "*"})


def get_ht_score(sportmonks_fixture_id: int) -> list[dict]:
    """Half-time score from d_match_scores."""
    return _get("d_match_scores",
                params={"sportmonks_fixture_id": f"eq.{sportmonks_fixture_id}",
                        "checkpoint": "eq.HT",
                        "select": "*"})


def get_dim_match(sportmonks_fixture_id: int) -> list[dict]:
    """Match dimension row (teams, stage, group info)."""
    return _get("dim_match",
                params={"sportmonks_fixture_id": f"eq.{sportmonks_fixture_id}",
                        "select": "*"})
