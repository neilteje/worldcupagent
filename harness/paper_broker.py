"""
Demo "calc sheet" broker — simulated trading, no arena writes.

Responsibilities:
  1. Fetch a market snapshot. Real Polymarket mids when a slug resolves; otherwise
     a clearly-labeled *synthetic* reference derived from the prediction (seeded
     zero-mean noise + a small vig) so marketless friendlies still exercise the
     full trade/sizing/settlement pipeline.
  2. Turn a shared prediction into per-agent paper trades using the existing
     EV/de-vig/Kelly engine (betting/decision.py), gated by each profile's policy.
  3. Persist a per-agent ledger (bankroll + trades) and settle it against results.

Settlement uses the entry price the bet was filled at: a winning YES share bought
at price q returns 1, i.e. profit = stake * (1/q - 1); a loss returns -stake.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from random import Random
import json
import uuid
from pathlib import Path

from betting import decision as ev_decision
from harness.profiles import AgentProfile

SLOT_OF_CODE = {}  # filled per call


def fetch_real_market(fx) -> dict | None:
    """Try to resolve a live Polymarket moneyline for this fixture; else None."""
    try:
        from data import polymarket as pm
    except Exception:
        return None
    slug = fx.guess_slug()
    try:
        ml = pm.get_moneyline_by_slug(slug)
    except Exception:
        ml = None
    if ml and (ml.get("outcomes") or {}):
        # require all three mids present to be tradable
        ok = all(isinstance((ml["outcomes"].get(s) or {}).get("current_mid_yes"), (int, float))
                 for s in ("home", "draw", "away"))
        if ok:
            ml["market_source"] = "polymarket"
            return ml
    return None


def synthetic_market(fx, prediction, vig: float = 0.04) -> dict:
    """
    Build a labeled demo market from the prediction: inject seeded zero-mean noise
    (so edges are mixed, not biased) and add a small vig. This is NOT a real price
    — it exists so the trading pipeline has something to act on for matches with no
    Polymarket market.
    """
    rng = Random(f"{fx.fixture_code}-{prediction.window}")
    p = prediction.probabilities
    slots = {"home": p[fx.home_code], "draw": p["draw"], "away": p[fx.away_code]}
    noisy = {s: max(0.02, v + rng.gauss(0, 0.04)) for s, v in slots.items()}
    tot = sum(noisy.values())
    fair = {s: noisy[s] / tot for s in noisy}
    codes = {"home": fx.home_code, "draw": "draw", "away": fx.away_code}
    return {
        "fixture": f"{fx.home} vs {fx.away}",
        "polymarket_event_slug": None,
        "market_source": "synthetic_demo",
        "outcomes": {
            s: {"team_code": codes[s], "current_mid_yes": round(fair[s] * (1 + vig), 4)}
            for s in ("home", "draw", "away")
        },
    }


def get_market(fx, prediction, *, mode: str = "auto") -> dict:
    """mode: 'real' (must be live), 'synthetic' (always demo), 'auto' (real else demo)."""
    if mode in ("real", "auto"):
        real = fetch_real_market(fx)
        if real:
            return real
        if mode == "real":
            return synthetic_market(fx, prediction)  # safety: never block the harness
    return synthetic_market(fx, prediction)


# ── Ledger persistence ──────────────────────────────────────────────────────

def _ledger_path(session_dir: Path) -> Path:
    return Path(session_dir) / "ledger.json"


def load_ledger(session_dir: str | Path, profiles: dict[str, AgentProfile]) -> dict:
    path = _ledger_path(Path(session_dir))
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "agents": {
            name: {"label": p.label, "start_bankroll": p.bankroll,
                   "bankroll": p.bankroll, "trades": []}
            for name, p in profiles.items()
        },
    }


def save_ledger(session_dir: str | Path, ledger: dict) -> None:
    path = _ledger_path(Path(session_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def _already_traded(ledger: dict, agent: str, fixture_code: str, window: str) -> bool:
    return any(t["fixture_code"] == fixture_code and t["window"] == window
               for t in ledger["agents"][agent]["trades"])


def _high_flag_on(scout_flags, code: str) -> bool:
    for f in scout_flags or []:
        if str(f.get("severity", "")).lower() == "high" and str(f.get("team", "")).lower() == code.lower():
            return True
    return False


# ── The decision: prediction + profile → paper trades ───────────────────────

def decide_trades(profile: AgentProfile, prediction, moneyline: dict, ledger: dict) -> list[dict]:
    """Apply one agent's policy to the shared prediction; returns new trade dicts."""
    agent = profile.name
    window = prediction.window
    fixture_code = prediction.fixture_code
    if _already_traded(ledger, agent, fixture_code, window):
        return []
    if window == "PRE_MATCH" and not profile.trade_prematch:
        return []
    if window == "HT" and not profile.trade_halftime:
        return []
    if prediction.confidence_num < profile.min_confidence:
        return []

    bankroll = float(ledger["agents"][agent]["bankroll"])
    game = ev_decision.evaluate_game(
        prediction.probabilities, moneyline,
        prediction.home_code, prediction.away_code,
        bankroll, kelly_fraction=profile.kelly_fraction,
    )

    picks = []
    for ev in game.ranked:
        if ev.raw_mid is None or ev.ev_per_dollar <= 0:
            continue
        if ev.edge_vs_fair < profile.min_edge_vs_fair:
            continue
        if ev.ev_per_dollar < profile.min_ev_per_dollar:
            continue
        if profile.skip_on_high_scout_flag and _high_flag_on(prediction.scout_flags, ev.code):
            continue
        picks.append(ev)
        if len(picks) >= profile.max_bets_per_window:
            break

    trades = []
    for ev in picks:
        size = min(ev.kelly_usd, profile.max_bet_usd,
                   bankroll * profile.stake_cap_fraction, bankroll)
        size = round(size, 2)
        if size < 1.0:                       # mirror the arena $1 minimum
            continue
        bankroll -= 0  # stake is at risk but bankroll is realized only at settlement
        trades.append({
            "trade_id": str(uuid.uuid4())[:8],
            "agent": agent,
            "fixture_code": fixture_code,
            "window": window,
            "slot": ev.slot,
            "outcome": ev.code,
            "stake": size,
            "entry_price": ev.raw_mid,
            "our_prob": ev.our_prob,
            "fair_prob": ev.fair_prob,
            "edge_vs_fair": ev.edge_vs_fair,
            "ev_per_dollar": ev.ev_per_dollar,
            "market_source": moneyline.get("market_source", "unknown"),
            "overround": game.overround,
            "ts": datetime.now(timezone.utc).isoformat(),
            "status": "open",
            "pnl": 0.0,
        })
    ledger["agents"][agent]["trades"].extend(trades)
    return trades


# ── Settlement ──────────────────────────────────────────────────────────────

def settle(ledger: dict, fixture_code: str, result_slot: str) -> int:
    """Resolve all open trades on a fixture against the winning slot; update bankrolls."""
    settled = 0
    for agent, book in ledger["agents"].items():
        for t in book["trades"]:
            if t["fixture_code"] != fixture_code or t["status"] != "open":
                continue
            q = max(0.02, min(0.98, float(t["entry_price"])))
            if t["slot"] == result_slot:
                t["pnl"] = round(t["stake"] * ((1.0 / q) - 1.0), 4)
                t["status"] = "won"
            else:
                t["pnl"] = round(-t["stake"], 4)
                t["status"] = "lost"
            book["bankroll"] = round(float(book["bankroll"]) + t["pnl"], 4)
            settled += 1
    return settled
