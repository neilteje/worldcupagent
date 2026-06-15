"""
Shared prediction provider.

Produces ONE forecast per (fixture, window) that every agent profile then trades.
Engine selection, in order of fidelity, with automatic fallback so a live window
never hard-fails:

  council        — the real World Cup brain: Grok pulse → Scout → Analyst → Devil
                   → Judge (reasoning/council.py), fed best-effort web/Reddit/Kalshi
                   research AND the deterministic_v2 ensemble as grounding. This is
                   what we run during the tournament.
  deterministic  — the deterministic_v2 ensemble alone (Elo + Poisson + market
                   prior), useful when LLM keys/budget are unavailable.
  market         — de-vigged market mids as the forecast (last resort / baseline).

Predictions are cached to the session dir so re-runs and both agents reuse the
exact same forecast (cheap + a clean A/B).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from harness.profiles import confidence_to_num


@dataclass
class Prediction:
    fixture_code: str
    window: str
    home_code: str
    away_code: str
    probabilities: dict           # {home_code: p, "draw": p, away_code: p}
    confidence_label: str
    confidence_num: float
    engine: str                   # which engine actually produced this
    scout_flags: list = field(default_factory=list)
    note: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize(probs: dict, home_code: str, away_code: str) -> dict | None:
    keys = (home_code, "draw", away_code)
    vals = {}
    for k in keys:
        v = (probs or {}).get(k)
        if not isinstance(v, (int, float)):
            return None
        vals[k] = max(0.0, float(v))
    s = sum(vals.values())
    if s <= 0:
        return None
    return {k: round(vals[k] / s, 4) for k in keys}


def _market_probs(moneyline: dict | None, home_code: str, away_code: str) -> dict | None:
    if not moneyline:
        return None
    raw = {}
    for slot, code in (("home", home_code), ("draw", "draw"), ("away", away_code)):
        mid = (moneyline.get("outcomes", {}).get(slot) or {}).get("current_mid_yes")
        if not isinstance(mid, (int, float)):
            return None
        raw[code] = float(mid)
    return _normalize(raw, home_code, away_code)


# ── Engine: council ─────────────────────────────────────────────────────────

def _predict_council(fx, window, moneyline) -> Prediction | None:
    try:
        from reasoning import council
    except Exception as exc:
        print(f"  [predictor] council import failed: {exc!r}")
        return None

    home, away = fx.home, fx.away
    home_code, away_code = fx.home_code, fx.away_code
    date = fx.date

    web = reddit = None
    kalshi_ml = None
    try:
        from data import web_search
        web = web_search.gather_research(home, away, date)
    except Exception as exc:
        print(f"  [predictor] web research failed: {exc!r}")
    try:
        from data import reddit_sentiment
        reddit = reddit_sentiment.get_sentiment_bundle(home, away)
    except Exception as exc:
        print(f"  [predictor] reddit sentiment failed: {exc!r}")
    try:
        from data import kalshi
        kalshi_ml = kalshi.get_moneyline(home, away)
    except Exception as exc:
        print(f"  [predictor] kalshi failed: {exc!r}")

    # Structured grounding — the same Sportmonks + Supabase digests agent.py
    # feeds the council. Supabase priors resolve by team NAME, so friendlies
    # get real H2H/style data too; Sportmonks needs a fixture id.
    sm_digest = sb_digest = None
    try:
        from data import fixture_bundle
        ctx = fixture_bundle.build_context(
            home, away, home_code, away_code,
            sportmonks_fixture_id=getattr(fx, "sportmonks_fixture_id", None),
            fixture_name=f"{home} vs {away}")
        sm_digest = ctx["sportmonks_digest"]
        sb_digest = ctx["supabase_digest"]
        print(f"  [predictor] grounding: sportmonks={'yes' if sm_digest else 'NO'} "
              f"supabase={'yes' if sb_digest else 'NO'}")
    except Exception as exc:
        print(f"  [predictor] fixture context failed: {exc!r}")

    pm_digest = None
    mkt = _market_probs(moneyline, home_code, away_code)
    if mkt:
        pm_digest = {
            "implied_win_prob": mkt,
            "market_handle": (moneyline or {}).get("polymarket_event_slug"),
            "data_availability": "mids_available",
        }

    try:
        cr = council.run_council(
            f"{home} vs {away}", home_code, away_code, home, away,
            f"{date} ({window})",
            sm_digest, sb_digest,
            pm_digest, kalshi_ml, web, reddit,
        )
    except Exception as exc:
        print(f"  [predictor] council run failed: {exc!r}")
        return None

    probs = _normalize(cr.probabilities, home_code, away_code)
    if not probs:
        print("  [predictor] council returned unusable probabilities")
        return None
    g = getattr(cr, "grounding", {}) or {}
    return Prediction(
        fixture_code=fx.fixture_code, window=window,
        home_code=home_code, away_code=away_code,
        probabilities=probs,
        confidence_label=str(cr.confidence or "low"),
        confidence_num=confidence_to_num(cr.confidence),
        engine="council",
        scout_flags=list(cr.scout_flags or []),
        note=(f"market_alignment={cr.market_alignment}; "
              f"anchor={(g.get('anchor') or {}).get('source', 'none')}; "
              f"shrink_lambda={g.get('shrink_lambda', 0)}; "
              f"flags={g.get('sanity_flags', [])}"),
    )


# ── Engine: deterministic ───────────────────────────────────────────────────

def _state_from_prior(prior: dict | None, side: str) -> dict:
    p = prior or {"home": 0.40, "draw": 0.28, "away": 0.32}
    diff = float(p.get("home", 0.40) or 0.40) - float(p.get("away", 0.32) or 0.32)
    return {
        "live_rating": diff if side == "home" else -diff,
        "matches": 0, "xg_for": 0.0, "xg_against": 0.0,
        "goals_for": 0.0, "goals_against": 0.0,
    }


def _predict_deterministic(fx, window, moneyline) -> Prediction | None:
    try:
        from models.deterministic_v2 import EnsembleConfig, predict_v2
    except Exception:
        return None
    mkt = _market_probs(moneyline, fx.home_code, fx.away_code)
    prior = None
    if mkt:
        prior = {"home": mkt[fx.home_code], "draw": mkt["draw"], "away": mkt[fx.away_code]}
    try:
        out = predict_v2(_state_from_prior(prior, "home"), _state_from_prior(prior, "away"),
                         market_probs=prior, cfg=EnsembleConfig())
        hda = out["probabilities"]
    except Exception:
        return None
    probs = _normalize({fx.home_code: hda["home"], "draw": hda["draw"], fx.away_code: hda["away"]},
                       fx.home_code, fx.away_code)
    if not probs:
        return None
    return Prediction(
        fixture_code=fx.fixture_code, window=window,
        home_code=fx.home_code, away_code=fx.away_code,
        probabilities=probs,
        confidence_label="medium", confidence_num=float(out.get("confidence", 0.55)),
        engine="deterministic_v2",
        note=f"deterministic_v2 ensemble (prior={'market' if prior else 'neutral'})",
    )


# ── Engine: market baseline ─────────────────────────────────────────────────

def _predict_market(fx, window, moneyline) -> Prediction:
    probs = _market_probs(moneyline, fx.home_code, fx.away_code) or {
        fx.home_code: 0.40, "draw": 0.28, fx.away_code: 0.32}
    return Prediction(
        fixture_code=fx.fixture_code, window=window,
        home_code=fx.home_code, away_code=fx.away_code,
        probabilities=probs, confidence_label="low", confidence_num=0.40,
        engine="market",
        note="market-implied baseline" if moneyline else "neutral prior (no market)",
    )


_ENGINES = {
    "council": _predict_council,
    "deterministic": _predict_deterministic,
    "market": _predict_market,
}
# Fallback chain when the requested engine returns nothing.
_FALLBACK = ["council", "deterministic", "market"]


def _cache_path(session_dir: Path, fixture_code: str, window: str) -> Path:
    return Path(session_dir) / "predictions" / f"{fixture_code}-{window}.json"


def predict(fx, window: str, *, engine: str = "council", moneyline: dict | None = None,
            session_dir: str | Path = "storage/harness", refresh: bool = False) -> Prediction:
    """Generate (or load cached) the shared forecast for one fixture window."""
    cache = _cache_path(Path(session_dir), fx.fixture_code, window)
    if cache.exists() and not refresh:
        try:
            return Prediction(**json.loads(cache.read_text(encoding="utf-8")))
        except Exception:
            pass

    order = [engine] + [e for e in _FALLBACK if e != engine]
    pred: Prediction | None = None
    for name in order:
        fn = _ENGINES.get(name)
        if not fn:
            continue
        pred = fn(fx, window, moneyline)
        if pred is not None:
            break
    if pred is None:
        pred = _predict_market(fx, window, moneyline)

    pred.created_at = datetime.now(timezone.utc).isoformat()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pred.to_dict(), indent=2), encoding="utf-8")
    return pred
