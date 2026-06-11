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
from betting import policy as bet_policy
from harness.profiles import AgentProfile


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


# ── The decision: prediction + profile → paper trades ───────────────────────

def decide_trades(profile: AgentProfile, prediction, moneyline: dict, ledger: dict) -> list[dict]:
    """Apply one agent's policy to the shared prediction; returns new trade dicts."""
    agent = profile.name
    window = prediction.window
    fixture_code = prediction.fixture_code
    if _already_traded(ledger, agent, fixture_code, window):
        return []

    is_synthetic = moneyline.get("market_source") == "synthetic_demo"
    bankroll = float(ledger["agents"][agent]["bankroll"])
    skip_reasons: list[str] = []
    picks = bet_policy.select_picks(
        profile, prediction.probabilities, moneyline,
        prediction.home_code, prediction.away_code, bankroll,
        window=window, confidence_num=prediction.confidence_num,
        scout_flags=prediction.scout_flags, skip_reasons=skip_reasons,
    )
    if not picks and skip_reasons:
        print(f"    [{agent}] no trade: {skip_reasons[0]}")

    # Overround is informational only; recompute once for the trade record.
    game = ev_decision.evaluate_game(
        prediction.probabilities, moneyline,
        prediction.home_code, prediction.away_code,
        bankroll, kelly_fraction=profile.kelly_fraction,
        min_edge_vs_fair=profile.min_edge_vs_fair,
    ) if picks else None

    trades = []
    for pk in picks:
        trades.append({
            "trade_id": str(uuid.uuid4())[:8],
            "agent": agent,
            "fixture_code": fixture_code,
            "window": window,
            "slot": pk.slot,
            "outcome": pk.code,
            "stake": pk.stake_usd,
            "entry_price": pk.entry_price,
            "our_prob": pk.our_prob,
            "fair_prob": pk.fair_prob,
            "edge_vs_fair": pk.edge_vs_fair,
            "ev_per_dollar": pk.ev_per_dollar,
            "market_source": moneyline.get("market_source", "unknown"),
            "synthetic_warning": is_synthetic,
            "overround": game.overround if game else None,
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
