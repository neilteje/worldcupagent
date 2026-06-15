"""
Supabase data access for the World Cup Arena.

Catalog: public schema (no Accept-Profile) -- tables: catalog_tables, catalog_full, catalog_columns
Data:    world_cup_arena schema (Accept-Profile: world_cup_arena)

Real tables (verified 2026-06-06):

  PRIORS (ads_a_* -- StatsBomb historical, keyed by country_id or country name):
    ads_a_h2h_country      1484 rows  head-to-head records between country pairs
    ads_a_stage_record      180 rows  win/draw/loss at each tournament stage per country
    ads_a_country_style      71 rows  set-piece, goals-per-game metrics
    ads_a_ko_pattern         71 rows  modal KO exit stage
    ads_a_special_match      36 rows  extra-time / penalty shootout history
    ads_a_country_struct     66 rows  last final appearance date
    ads_a_manager           206 rows  manager coaching stats
    ads_a_h2h_continent      28 rows  continent-level h2h records

  LIVE CHECKPOINTS (d_* -- Sportmonks live, keyed by Sportmonks match_id integer):
    d_checkpoint_snapshot   260 rows  per-team stats at HT/FT/ET1/ET2 (grows live during tournament)
    d_match_scores          206 rows  score at each checkpoint
    d_checkpoint_runs       130 rows  pipeline run status
    d_checkpoint_minutes    130 rows  minute bounds per checkpoint
    sm_match_meta           ???  rows  Sportmonks match metadata mirror

  DIMENSIONS (dim_*):
    dim_match                65 rows  match metadata keyed by Sportmonks match_id
    dim_checkpoint            4 rows  checkpoint code definitions (HT/FT/ET1/ET2)

Key: country_id values in ads_a_* are StatsBomb internal IDs, NOT Sportmonks country IDs.
     Use country name string matching (ilike) to look up priors reliably.
     Live d_* tables are keyed by Sportmonks fixture integer IDs (match_id).
"""
from __future__ import annotations
import httpx
from agent.config import Settings


# --------------------------------------------------------------------------
# Low-level fetch helpers
# --------------------------------------------------------------------------

def _fetch_wca(settings: Settings, table: str, params: dict) -> list[dict]:
    """Fetch rows from the world_cup_arena schema."""
    headers = {
        "apikey": settings.supabase_publishable_key,
        "Authorization": f"Bearer {settings.supabase_publishable_key}",
        "Accept-Profile": "world_cup_arena",
    }
    try:
        r = httpx.get(
            f"{settings.supabase_url.rstrip('/')}/{table}",
            headers=headers,
            params=params,
            timeout=15,
        )
        if not r.is_success:
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _fetch_pub(settings: Settings, table: str, params: dict) -> list[dict]:
    """Fetch rows from the public schema (catalog tables)."""
    headers = {
        "apikey": settings.supabase_publishable_key,
        "Authorization": f"Bearer {settings.supabase_publishable_key}",
    }
    try:
        r = httpx.get(
            f"{settings.supabase_url.rstrip('/')}/{table}",
            headers=headers,
            params=params,
            timeout=15,
        )
        if not r.is_success:
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


# --------------------------------------------------------------------------
# Priors from historical StatsBomb data
# --------------------------------------------------------------------------

def _h2h_lookup(settings: Settings, name_a: str, name_b: str) -> dict | None:
    """Fetch one h2h row for (country_a vs country_b). Tries both orderings."""
    for a, b, flip in [(name_a, name_b, False), (name_b, name_a, True)]:
        rows = _fetch_wca(settings, "ads_a_h2h_country", {
            "country_name_a": f"ilike.{a}",
            "country_name_b": f"ilike.{b}",
            "match_scope": "eq.all",
            "select": "country_id_a,country_id_b,wins_a_weighted,draws_weighted,losses_a_weighted,total_weight,total_matches",
            "limit": "1",
        })
        if rows:
            row = rows[0]
            row["_flipped"] = flip
            return row
    return None


def _stage_win_rate(settings: Settings, country_id: int, stage: str = "group") -> float | None:
    """Win rate for a country at a given tournament stage."""
    rows = _fetch_wca(settings, "ads_a_stage_record", {
        "country_id": f"eq.{country_id}",
        "stage_canonical": f"eq.{stage}",
        "select": "wins,draws,losses,matches,win_rate",
        "limit": "1",
    })
    if not rows:
        return None
    r = rows[0]
    return float(r["win_rate"]) if r.get("win_rate") is not None else None


def _style_stats(settings: Settings, country_id: int) -> dict:
    """Style metrics (goals per game, set-piece efficiency) for a country."""
    rows = _fetch_wca(settings, "ads_a_country_style", {
        "country_id": f"eq.{country_id}",
        "select": "group_gpg,ko_gpg,set_piece_goals,set_piece_shots,conversion_rate",
        "limit": "1",
    })
    return rows[0] if rows else {}


def get_priors(
    settings: Settings,
    fixture_code: str,
    home_name: str | None = None,
    away_name: str | None = None,
    stage: str = "group",
) -> dict | None:
    """
    Return 3-way probability priors from historical StatsBomb data.

    Sources used (in priority order):
    1. Head-to-head win rates from ads_a_h2h_country
    2. Stage-level win rates from ads_a_stage_record
    3. Returns None if neither source is available (caller uses its own fallback)

    h2h is blended with a uniform prior weighted by match count:
      - 1+ matches:  h2h weight = min(0.6, matches/10 * 0.6)
      - 0 matches:   fall through to stage records only
    """
    if not home_name or not away_name:
        return None

    h2h = _h2h_lookup(settings, home_name, away_name)

    h2h_home = h2h_draw = h2h_away = None
    home_cid = away_cid = None

    if h2h:
        total_w = float(h2h.get("total_weight") or 0)
        n_matches = int(h2h.get("total_matches") or 0)
        if total_w > 0:
            wins_a = float(h2h.get("wins_a_weighted") or 0)
            draws  = float(h2h.get("draws_weighted") or 0)
            loss_a = float(h2h.get("losses_a_weighted") or 0)
            if h2h.get("_flipped"):
                h2h_home = loss_a / total_w
                h2h_draw = draws  / total_w
                h2h_away = wins_a / total_w
                home_cid = h2h.get("country_id_b")
                away_cid = h2h.get("country_id_a")
            else:
                h2h_home = wins_a / total_w
                h2h_draw = draws  / total_w
                h2h_away = loss_a / total_w
                home_cid = h2h.get("country_id_a")
                away_cid = h2h.get("country_id_b")

        h2h_weight = min(0.60, n_matches / 10 * 0.60) if n_matches > 0 else 0.0
    else:
        h2h_weight = 0.0
        n_matches = 0

    # Stage records (need country_id from h2h or skip)
    stage_home = stage_away = None
    if home_cid:
        stage_home = _stage_win_rate(settings, home_cid, stage)
    if away_cid:
        stage_away = _stage_win_rate(settings, away_cid, stage)

    # Convert stage win rates to 3-way probs
    # Neutral-venue WC: no home advantage adjustment
    # avg draw rate at group stage ~0.25
    AVG_DRAW = 0.25
    stage_p_home = stage_p_draw = stage_p_away = None
    if stage_home is not None and stage_away is not None:
        remainder = 1.0 - AVG_DRAW
        denom = stage_home + stage_away
        if denom > 0:
            stage_p_home = remainder * (stage_home / denom)
            stage_p_away = remainder * (stage_away / denom)
            stage_p_draw = AVG_DRAW

    # Blend: h2h + stage + uniform
    stage_weight = 0.25 if stage_p_home is not None else 0.0
    uniform_weight = max(0.0, 1.0 - h2h_weight - stage_weight)

    p_home = p_draw = p_away = None

    if h2h_weight > 0 and h2h_home is not None:
        p_home = h2h_weight * h2h_home
        p_draw = h2h_weight * h2h_draw
        p_away = h2h_weight * h2h_away
    else:
        p_home = p_draw = p_away = 0.0

    if stage_weight > 0 and stage_p_home is not None:
        p_home += stage_weight * stage_p_home
        p_draw += stage_weight * stage_p_draw
        p_away += stage_weight * stage_p_away

    if uniform_weight > 0:
        p_home += uniform_weight * 0.37   # slight home-coding edge
        p_draw += uniform_weight * 0.26
        p_away += uniform_weight * 0.37

    total = p_home + p_draw + p_away
    if total <= 0:
        return None

    style_home = _style_stats(settings, home_cid) if home_cid else {}
    style_away = _style_stats(settings, away_cid) if away_cid else {}

    return {
        "home":  round(p_home / total, 4),
        "draw":  round(p_draw / total, 4),
        "away":  round(p_away / total, 4),
        "source": "supabase_h2h_stage",
        "h2h_matches": n_matches,
        "h2h_weight": round(h2h_weight, 2),
        "stage_weight": round(stage_weight, 2),
        "home_group_gpg": style_home.get("group_gpg"),
        "away_group_gpg": style_away.get("group_gpg"),
    }


# --------------------------------------------------------------------------
# Live checkpoint data (populated during tournament)
# --------------------------------------------------------------------------

def get_live_checkpoint(settings: Settings, fixture_code: str) -> dict | None:
    """
    Fetch HT stats for a live fixture from d_* tables.

    fixture_code must be the Sportmonks integer fixture ID (as a string).
    Returns None if no HT checkpoint exists yet (pre-match or not yet ingested).

    Data comes from:
      d_match_scores         -- score (home_goals, away_goals)
      d_checkpoint_snapshot  -- richer stats (xG, shots, possession, cards)
    """
    try:
        match_id = int(fixture_code)
    except (ValueError, TypeError):
        return None

    # Score
    score_rows = _fetch_wca(settings, "d_match_scores", {
        "match_id": f"eq.{match_id}",
        "checkpoint_code": "eq.HT",
        "select": "home_goals,away_goals",
        "limit": "1",
    })
    if not score_rows:
        return None

    result: dict = {
        "home_goals": int(score_rows[0].get("home_goals") or 0),
        "away_goals": int(score_rows[0].get("away_goals") or 0),
    }

    # Richer per-team stats
    snap_rows = _fetch_wca(settings, "d_checkpoint_snapshot", {
        "match_id": f"eq.{match_id}",
        "checkpoint_code": "eq.HT",
        "select": "is_home,cum_xg,raw_cum_shots_total,raw_cum_shots_on_target,"
                  "raw_cum_possession_pct,cum_red_cards,cum_yellow_cards",
        "limit": "2",
    })
    for row in snap_rows:
        pfx = "home" if row.get("is_home") else "away"
        if row.get("cum_xg") is not None:
            result[f"{pfx}_xg"] = float(row["cum_xg"])
        if row.get("raw_cum_shots_total") is not None:
            result[f"{pfx}_shots"] = int(row["raw_cum_shots_total"])
        if row.get("raw_cum_shots_on_target") is not None:
            result[f"{pfx}_sot"] = int(row["raw_cum_shots_on_target"])
        if row.get("raw_cum_possession_pct") is not None:
            result[f"{pfx}_possession"] = float(row["raw_cum_possession_pct"])
        if row.get("cum_red_cards") is not None:
            result[f"{pfx}_red"] = int(row["cum_red_cards"])

    return result


# --------------------------------------------------------------------------
# Catalog helper (public schema, no Accept-Profile)
# --------------------------------------------------------------------------

def get_catalog(settings: Settings) -> list[dict]:
    """Return the full data catalog (table names, descriptions, row counts)."""
    return _fetch_pub(settings, "catalog_full", {"limit": "100"})
