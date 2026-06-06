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
