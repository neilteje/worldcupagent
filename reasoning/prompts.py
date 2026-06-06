"""
Prompt templates for the World Cup Arena agent.

FIVE LLM calls per fixture window (matching the notebook architecture):
  1. digest_sportmonks   — compress Sportmonks payload → clean JSON
  2. digest_polymarket   — compress Polymarket moneyline → clean JSON + execution handles
  3. digest_supabase     — compress multi-table priors → per-team profile JSON
  4. predict             — form independent prediction from digests (market-blind)
  5. strategy            — compare prediction vs market → trade decision

The digest + predict steps are kept intentionally separate:
  - Prediction is formed WITHOUT market prices (prevents anchoring)
  - Strategy step then compares prediction to market to find edge
  - This mirrors how professional analysts separate view-formation from execution

For the Stair AI reasoning score, the full internal chain-of-thought from each
Claude call is captured and submitted in the ledger Thinking records.
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# 1. SPORTMONKS DIGEST
# ═══════════════════════════════════════════════════════════════════════════

SPORTMONKS_DIGEST_SYS = (
    "You are a soccer analyst. You receive a raw Sportmonks pre-match payload for "
    "one fixture and must distil it into a self-contained JSON digest that a "
    "downstream LLM (with no other context about Sportmonks) will read.\n\n"

    "## Input shape\n"
    "  - fixture       : match name (e.g. 'Mexico vs South Africa')\n"
    "  - home_code     : home team short code\n"
    "  - away_code     : away team short code\n"
    "  - predictions[] : Sportmonks ML rows. The Full-Time-Result type carries\n"
    "                    win/draw/loss probabilities in a `predictions` object.\n"
    "                    May be empty.\n"
    "  - odds[]        : bookmaker odds rows. market_id=1 is the 1X2 winner market.\n"
    "                    label: '1'=home, 'X'=draw, '2'=away. `probability` is\n"
    "                    0-100. Average across bookmakers for consensus.\n"
    "  - xGFixture[]   : expected-goals entries per team. May be empty.\n\n"

    "## Output schema (return ONLY valid JSON — no prose, no code fences)\n"
    "{\n"
    "  'fixture'                       : str,\n"
    "  'home_team'                     : str,\n"
    "  'away_team'                     : str,\n"
    "  'sportmonks_ml_win_prob'        : {home_code: float, 'draw': float, away_code: float} | null,\n"
    "  'bookmaker_consensus_win_prob'  : {home_code: float, 'draw': float, away_code: float} | null,\n"
    "  'bookmaker_count'               : int | null,\n"
    "  'expected_goals'                : {home_code: float, away_code: float} | null,\n"
    "  'data_availability': {\n"
    "    'sportmonks_ml'       : 'available' | 'missing',\n"
    "    'bookmaker_consensus' : 'available' | 'missing',\n"
    "    'expected_goals'      : 'available' | 'missing'\n"
    "  },\n"
    "  'summary': str\n"
    "}\n\n"
    "All probabilities in 0..1. Use null (not 0) for missing data. Never fabricate."
)


def sportmonks_digest_input(fixture: dict, home_code: str, away_code: str) -> str:
    import json
    odds_1x2 = sorted(
        [o for o in (fixture.get("odds") or []) if o.get("market_id") == 1],
        key=lambda o: o.get("latest_bookmaker_update") or "",
    )[-50:]  # last 50 1X2 rows
    return json.dumps({
        "fixture":     fixture.get("name"),
        "home_code":   home_code,
        "away_code":   away_code,
        "predictions": fixture.get("predictions"),
        "odds":        odds_1x2,
        "xGFixture":   fixture.get("xgfixture"),
    }, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# 2. POLYMARKET DIGEST
# ═══════════════════════════════════════════════════════════════════════════

POLYMARKET_DIGEST_SYS = (
    "You are an analyst digesting a Polymarket moneyline (3-way match-winner) "
    "market response into a self-contained JSON for a downstream LLM.\n\n"

    "## Input shape\n"
    "  - outcomes.{home,draw,away}\n"
    "      team_code        : team short code (or 'draw')\n"
    "      condition_id     : Polymarket condition id\n"
    "      token_yes        : YES-side token id\n"
    "      current_mid_yes  : mid price 0..1 (implied probability). null if unavailable.\n\n"

    "## Output schema (return ONLY valid JSON — no prose, no code fences)\n"
    "{\n"
    "  'fixture'           : str,\n"
    "  'market_handle'     : str,\n"
    "  'implied_win_prob'  : {home_code: float, 'draw': float, away_code: float} | null,\n"
    "  'sum_implied_prob'  : float | null,\n"
    "  'execution_handles' : {home_code: {condition_id, token_yes},\n"
    "                         'draw':    {condition_id, token_yes},\n"
    "                         away_code: {condition_id, token_yes}},\n"
    "  'data_availability' : 'mids_available' | 'mids_partial' | 'mids_missing' | 'no_market',\n"
    "  'summary'           : str\n"
    "}\n\n"
    "Use null for missing. Identify the favorite (highest mid). Note if sum ≠ ~1."
)


# ═══════════════════════════════════════════════════════════════════════════
# 3. SUPABASE DIGEST
# ═══════════════════════════════════════════════════════════════════════════

SUPABASE_DIGEST_SYS = (
    "You are an analyst aggregating multi-table Supabase priors for one fixture "
    "into a self-contained per-team JSON profile for a downstream LLM.\n\n"

    "## Tables you may receive (all from the world_cup_arena schema)\n"
    "  ads_a_country_style  — set_piece_shots/goals/conversion_rate, group_gpg, ko_gpg\n"
    "  ads_a_country_struct — formation, tactical indicators, recent wld record\n"
    "  ads_a_h2h_country    — head-to-head wins/draws/losses, goals, last result\n"
    "  ads_a_ko_pattern     — knockout advancement rate, avg goals in KO stage\n"
    "  ads_a_stage_record   — per-stage win%, goals_for, goals_against\n\n"

    "## Output schema (return ONLY valid JSON — no prose, no code fences)\n"
    "{\n"
    "  'fixture'          : str,\n"
    "  'tables_used'      : [str],\n"
    "  'teams': {\n"
    "    home_code: {\n"
    "      'set_piece_efficiency' : float | null,\n"
    "      'set_piece_sample'     : int   | null,\n"
    "      'group_goals_per_game' : float | null,\n"
    "      'ko_goals_per_game'    : float | null,\n"
    "      'ko_advancement_rate'  : float | null,\n"
    "      'h2h_wins'             : int   | null,\n"
    "      'h2h_draws'            : int   | null,\n"
    "      'h2h_losses'           : int   | null,\n"
    "      'recent_form_notes'    : str   | null\n"
    "    },\n"
    "    away_code: { same shape }\n"
    "  },\n"
    "  'h2h_summary'     : str | null,\n"
    "  'data_availability': 'rich' | 'partial' | 'sparse',\n"
    "  'summary'         : str\n"
    "}\n\n"
    "Call out small sample sizes. Do NOT output a win probability — that's for the prediction step."
)


def supabase_digest_input(
    fixture_name: str,
    home_code: str,
    away_code: str,
    home_country_id: int,
    away_country_id: int,
    home_name: str,
    away_name: str,
    priors: dict,
) -> str:
    import json
    return json.dumps({
        "fixture":           fixture_name,
        "home_code":         home_code,
        "away_code":         away_code,
        "home_country_id":   home_country_id,
        "away_country_id":   away_country_id,
        "home_team_name":    home_name,    # for Claude to identify rows
        "away_team_name":    away_name,    # (StatsBomb IDs may differ from Sportmonks)
        "note":              "country_id values are Sportmonks IDs. "
                             "StatsBomb tables may use different IDs. "
                             "Use team names to identify relevant rows when IDs don't match.",
        "priors_by_table":   priors,
    }, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# 4. PREDICTION (market-blind — this is the core PSL-scored output)
# ═══════════════════════════════════════════════════════════════════════════

PREDICT_SYS = (
    "You are a soccer match analyst. You receive two pre-distilled digests for "
    "one fixture and must produce the agent's own outcome prediction.\n\n"

    "## Critical: the market is deliberately excluded from your input.\n"
    "Form your view from the data signals alone. Anchoring to the market price "
    "defeats the purpose — edge only exists if your view is independent.\n\n"

    "## Input\n"
    "  - sportmonks_digest : ML probabilities, bookmaker consensus, xG\n"
    "  - supabase_digest   : historical priors, H2H, set-piece efficiency\n\n"

    "## Output schema (return ONLY valid JSON — no prose, no code fences)\n"
    "{\n"
    "  'fixture'          : str,\n"
    "  'outcome'          : str,       // home_code | 'draw' | away_code\n"
    "  'probability'      : float,     // 0..1; your confidence in this outcome\n"
    "  'rationale'        : str,       // 2-4 sentences; name signals and caveats\n"
    "  'used_signals': {\n"
    "    'sportmonks' : 'leaned_on' | 'unavailable',\n"
    "    'supabase'   : 'leaned_on' | 'unavailable'\n"
    "  },\n"
    "  'confidence_level' : 'high' | 'medium' | 'low'\n"
    "}\n\n"
    "Probability must reflect what the data says alone. Be honest about uncertainty: "
    "sparse evidence → low confidence → probability closer to the base rate (~0.40/0.28/0.32 "
    "for home/draw/away in international football)."
)


def predict_input(
    fixture_name: str,
    home_code: str,
    away_code: str,
    sportmonks_digest: dict | None,
    supabase_digest: dict | None,
) -> str:
    import json
    return json.dumps({
        "fixture":           fixture_name,
        "home_code":         home_code,
        "away_code":         away_code,
        "sportmonks_digest": sportmonks_digest,
        "supabase_digest":   supabase_digest,
    })


# ═══════════════════════════════════════════════════════════════════════════
# 5. STRATEGY (compare prediction vs market → trade decision)
# ═══════════════════════════════════════════════════════════════════════════

STRATEGY_SYS = (
    "You are a bankroll manager for a demo prediction-market account. "
    "You receive the agent's own prediction and the current Polymarket market, "
    "and decide whether to trade and on what terms.\n\n"

    "## How to decide\n"
    "  1. Edge = prediction.probability − market.implied_win_prob[prediction.outcome]\n"
    "     Positive edge → market UNDERprices → long (buy YES on the outcome).\n"
    "     Negative edge → market OVERprices → consider shorting, but the arena\n"
    "     API only supports buy-YES. For a short: pick the cheapest other outcome\n"
    "     that makes analytical sense (e.g. to short MEX, long RSA or draw).\n"
    "  2. Size discipline (wallet ~$100, max $5 per trade):\n"
    "       |edge| < 5 pp              → skip (noise)\n"
    "       |edge| 5-15 pp             → $1-2\n"
    "       |edge| > 15 pp             → $3-5\n"
    "     Halve size if confidence_level == 'low'. Up to 1.5× (capped $5) if 'high'.\n"
    "     Skip if data_availability != 'mids_available'.\n"
    "  3. limit_price: worst acceptable price per share.\n"
    "     Long: just above the YES mid (e.g. mid 0.38 → limit 0.40).\n"
    "     Short-via-long: just above the target outcome's YES mid.\n\n"

    "## Output schema (return ONLY valid JSON — no prose, no code fences)\n"
    "{\n"
    "  'should_trade'  : bool,\n"
    "  'outcome'       : str,       // the outcome to LONG (team_code or 'draw')\n"
    "  'direction'     : 'long' | 'short',  // 'short' only means we're longing a different side\n"
    "  'team_code'     : str,       // the team_code to pass to the order API (for long: same as outcome; for short: alternative outcome)\n"
    "  'size_usdc'     : float,     // 0 if not trading; ≤5\n"
    "  'limit_price'   : float,     // 0..1\n"
    "  'edge_pp'       : float,     // (agent_prob − market_prob) × 100\n"
    "  'market_handle' : str,       // echo polymarket_digest.market_handle\n"
    "  'rationale'     : str        // 2-3 sentences: edge, size logic, limit logic\n"
    "}\n\n"
    "Be conservative: skip > bet on noise. State the team_code to long explicitly."
)


def strategy_input(prediction: dict, polymarket_digest: dict) -> str:
    import json
    return json.dumps({
        "prediction":        prediction,
        "polymarket_digest": polymarket_digest,
    })


# ═══════════════════════════════════════════════════════════════════════════
# 6. HALF-TIME UPDATE (replaces predict + strategy at HT)
# ═══════════════════════════════════════════════════════════════════════════

HT_PREDICT_SYS = (
    "You are a soccer analyst making a HALF-TIME prediction update. "
    "You have live match state and must update the pre-match prediction with "
    "a Bayesian update.\n\n"

    "## The key HT insight\n"
    "  xG often diverges from the scoreline — the team with higher xG is "
    "structurally more likely to win the second half. Don't anchor to the score alone.\n\n"

    "## Output schema (return ONLY valid JSON — no prose, no code fences)\n"
    "{\n"
    "  'fixture'          : str,\n"
    "  'outcome'          : str,\n"
    "  'probability'      : float,\n"
    "  'rationale'        : str,\n"
    "  'used_signals': {\n"
    "    'ht_score'    : 'available' | 'missing',\n"
    "    'ht_xg'       : 'available' | 'missing',\n"
    "    'ht_snapshot' : 'available' | 'missing'\n"
    "  },\n"
    "  'confidence_level'      : 'high' | 'medium' | 'low',\n"
    "  'changed_from_prematch' : bool,\n"
    "  'change_explanation'    : str\n"
    "}"
)


def ht_predict_input(
    fixture_name: str,
    home_code: str,
    away_code: str,
    prematch_prediction: dict | None,
    ht_snapshot: list[dict],
    ht_score: list[dict],
    ht_stats_sportmonks: dict,
) -> str:
    import json
    return json.dumps({
        "fixture":              fixture_name,
        "home_code":            home_code,
        "away_code":            away_code,
        "prematch_prediction":  prematch_prediction,
        "ht_snapshot_supabase": ht_snapshot,
        "ht_score_supabase":    ht_score,
        "ht_stats_sportmonks":  ht_stats_sportmonks,
    }, default=str)
