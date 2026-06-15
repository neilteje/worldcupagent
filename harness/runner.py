"""
Harness orchestration: schedule windows, run the shared prediction, paper-trade
each agent, settle, and report.

Commands (see `__main__.py`):
  init     — create the session dir, snapshot fixtures, write a results template
  now      — run ONE (fixture, window) immediately (manual / testing)
  run      — live loop: wait for each window's trigger time, then execute it
  settle   — resolve open trades against results.json, then write reports
  report   — (re)generate performance CSV/summary/plots from the current state
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import time
from pathlib import Path

from harness import fixtures as fx_mod
from harness import predictor as predictor_mod
from harness import paper_broker as broker
from harness import performance
from harness.profiles import load_profiles
from agents.anchor import AnchorStrategy
from agents.blitz import BlitzStrategy
from agents.contracts import FixtureDataSnapshot, MarketContext
from agents.hunter import HunterStrategy
from agents.monk import MonkStrategy
from betting.portfolio import PortfolioLimits, allocate_jointly


STRATEGIES = {
    "monk": MonkStrategy(),
    "anchor": AnchorStrategy(),
    "hunter": HunterStrategy(),
    "blitz": BlitzStrategy(),
}


def session_dir_for(date: str, name: str | None) -> Path:
    return Path("storage/harness") / (name or date)


def _find_fixture(fixtures, code):
    for f in fixtures:
        if f.fixture_code == code:
            return f
    return None


def _load_fixtures_for_args(args):
    fixtures = fx_mod.load_fixtures(args.fixtures)
    date = getattr(args, "date", None)
    if str(getattr(args, "fixtures", "")).strip().lower() == "auto" and date:
        fixtures = [f for f in fixtures if f.date == date]
    return fixtures


def _market_for(fx, prediction, real_ml, market_mode):
    if market_mode == "synthetic":
        return broker.synthetic_market(fx, prediction)
    if real_ml:
        return real_ml
    if market_mode == "real":
        return None
    return broker.synthetic_market(fx, prediction)


def _market_context(moneyline: dict | None) -> MarketContext | None:
    if not moneyline:
        return None
    mids = {
        s: (moneyline.get("outcomes", {}).get(s) or {}).get("current_mid_yes")
        for s in ("home", "draw", "away")
    }
    if not all(isinstance(v, (int, float)) for v in mids.values()):
        return None
    return MarketContext(
        observed_at=datetime.now(timezone.utc),
        polymarket=moneyline if moneyline.get("market_source") == "polymarket" else None,
        kalshi=None,
        bookmaker_consensus=None,
        bookmaker_comparison=None,
        devigged_probabilities=mids,
        best_bid=mids,
        best_ask=mids,
        midpoint=mids,
        expected_fill_price=mids,
        movement={},
        dispersion={},
        overround=None,
    )


def _snapshot_for(fx, prediction, market) -> FixtureDataSnapshot:
    p = prediction.probabilities
    home_p = float(p[fx.home_code])
    draw_p = float(p["draw"])
    away_p = float(p[fx.away_code])
    evidence_ids = [
        str(flag.get("evidence_id") or flag.get("signal_id") or flag.get("signal"))
        for flag in prediction.scout_flags
        if isinstance(flag, dict) and (flag.get("evidence_id") or flag.get("signal_id") or flag.get("signal"))
    ]
    return FixtureDataSnapshot(
        fixture_id=fx.fixture_code,
        fixture_name=f"{fx.home} vs {fx.away}",
        window=prediction.window,
        kickoff=fx.kickoff_dt(),
        as_of_timestamp=datetime.now(timezone.utc),
        home_code=fx.home_code,
        away_code=fx.away_code,
        home_name=fx.home,
        away_name=fx.away,
        sportmonks=None,
        supabase=None,
        bzzoiro=None,
        web=None,
        reddit=None,
        social=None,
        football_context={
            "home_code": fx.home_code,
            "away_code": fx.away_code,
            "council_forecast": {
                "probabilities": {"home": home_p, "draw": draw_p, "away": away_p},
                "confidence": prediction.confidence_num,
                "evidence_ids": evidence_ids,
            },
            "independent_forecast": {
                "probabilities_by_code": {fx.home_code: home_p, "draw": draw_p, fx.away_code: away_p},
                "lower_bounds_by_code": {
                    fx.home_code: max(0.0, home_p - 0.10),
                    "draw": max(0.0, draw_p - 0.10),
                    fx.away_code: max(0.0, away_p - 0.10),
                },
                "upper_bounds_by_code": {
                    fx.home_code: min(1.0, home_p + 0.10),
                    "draw": min(1.0, draw_p + 0.10),
                    fx.away_code: min(1.0, away_p + 0.10),
                },
                "data_coverage_score": 0.75,
            },
            "forecast_snapshot_id": f"{fx.fixture_code}-{prediction.window}-{prediction.engine}",
            "evidence_ids": evidence_ids,
            "scout_flags": prediction.scout_flags,
            "event_signals": [
                flag for flag in prediction.scout_flags
                if isinstance(flag, dict) and (flag.get("event_type") or flag.get("trigger"))
            ],
        },
        live_context=None,
        market_context=_market_context(market),
        snapshot_id=f"snap_{fx.fixture_code}_{prediction.window}",
        snapshot_hash="",
    )


def _agent_recommendations(fx, prediction, market, profiles, ledger) -> list:
    snapshot = _snapshot_for(fx, prediction, market)
    market_ctx = snapshot.market_context
    recs = []
    is_synthetic = (market or {}).get("market_source") == "synthetic_demo"
    for name, profile in profiles.items():
        if name not in STRATEGIES:
            continue
        if broker.already_traded(ledger, name, fx.fixture_code, prediction.window):
            continue
        if is_synthetic and not profile.trade_synthetic:
            continue
        strategy = STRATEGIES[name]
        view = strategy.build_data_view(snapshot, None)
        forecast = strategy.build_forecast(view)
        candidates = strategy.generate_candidates(forecast, view, market_ctx)
        recs.extend(strategy.generate_recommendations(
            candidates,
            forecast,
            view,
            market_ctx,
            float(ledger["agents"][name]["bankroll"]),
        ))
    return recs


def execute_window(fx, window, *, engine, market_mode, profiles, session_dir, refresh=False) -> dict:
    """Run one (fixture, window): predict once, then paper-trade every profile."""
    print(f"\n=== {fx.fixture_code}  {window}  ({fx.home} vs {fx.away}) ===")
    real_ml = None if market_mode == "synthetic" else broker.fetch_real_market(fx)

    pred = predictor_mod.predict(fx, window, engine=engine, moneyline=real_ml,
                                 session_dir=session_dir, refresh=refresh)
    p = pred.probabilities
    print(f"  prediction[{pred.engine}]: "
          f"{fx.home_code} {p[fx.home_code]:.0%} | draw {p['draw']:.0%} | "
          f"{fx.away_code} {p[fx.away_code]:.0%}  (conf {pred.confidence_label})")

    market = _market_for(fx, pred, real_ml, market_mode)
    src = (market or {}).get("market_source", "none")
    mids = {
        s: ((market or {}).get("outcomes", {}).get(s) or {}).get("current_mid_yes")
        for s in ("home", "draw", "away")
    }
    print(f"  market[{src}]: home {mids['home']} | draw {mids['draw']} | away {mids['away']}")

    ledger = broker.load_ledger(session_dir, profiles)
    recs = _agent_recommendations(fx, pred, market, profiles, ledger)
    allocation = allocate_jointly(recs, limits=PortfolioLimits())
    window_trades = []
    accepted_by_agent = {r.agent_name: [] for r in allocation.accepted}
    for rec in allocation.accepted:
        trade = broker.record_recommendation_trade(
            ledger,
            rec.agent_name,
            fx.fixture_code,
            window,
            rec,
            market_source=src,
            home_code=fx.home_code,
            away_code=fx.away_code,
        )
        if trade:
            accepted_by_agent.setdefault(rec.agent_name, []).append(trade)
            window_trades.append(trade)
    rejected = {}
    for row in allocation.rejected:
        rec = row.get("recommendation") or {}
        rejected.setdefault(rec.get("agent_name"), []).append(row.get("reason"))
    for name in profiles:
        trades = accepted_by_agent.get(name) or []
        if trades:
            for t in trades:
                print(f"  [{name}] BET ${t['stake']:.2f} on {t['outcome']} "
                      f"@ {t['entry_price']:.2f}  (edge {t['edge_vs_fair']*100:+.1f}pp, "
                      f"EV {t['ev_per_dollar']*100:+.1f}%)")
        else:
            reason = (rejected.get(name) or ["policy bar not cleared"])[0]
            print(f"  [{name}] no bet ({reason})")
    broker.save_ledger(session_dir, ledger)

    _append_window_log(session_dir, {
        "fixture_code": fx.fixture_code, "window": window, "engine": pred.engine,
        "market_source": src, "n_trades": len(window_trades),
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    return {"prediction": pred.to_dict(), "trades": window_trades}


def _append_window_log(session_dir, row) -> None:
    path = Path(session_dir) / "windows.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def cmd_init(args) -> None:
    fixtures = _load_fixtures_for_args(args)
    profiles = load_profiles(args.profiles)
    session_dir = session_dir_for(args.date, args.session)
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "fixtures.json").write_text(
        json.dumps([f.to_dict() for f in fixtures], indent=2), encoding="utf-8")
    (session_dir / "profiles.json").write_text(
        json.dumps({n: p.to_dict() for n, p in profiles.items()}, indent=2), encoding="utf-8")
    broker.save_ledger(session_dir, broker.load_ledger(session_dir, profiles))
    template = performance.results_template(session_dir, fixtures)
    print(f"Initialized session: {session_dir}")
    print(f"  fixtures: {len(fixtures)}  agents: {', '.join(profiles)}")
    print("  Window schedule (local CDT):")
    for f, w in fx_mod.all_windows(fixtures):
        print(f"    {f.window_trigger(w).astimezone(fx_mod.LOCAL_TZ):%Y-%m-%d %H:%M}  "
              f"{f.fixture_code} {w}")
    print(f"  Fill results after matches in: {template}")


def cmd_now(args) -> None:
    fixtures = _load_fixtures_for_args(args)
    profiles = load_profiles(args.profiles)
    session_dir = session_dir_for(args.date, args.session)
    session_dir.mkdir(parents=True, exist_ok=True)
    fx = _find_fixture(fixtures, args.fixture)
    if not fx:
        print(f"Unknown fixture {args.fixture}. Known: {[f.fixture_code for f in fixtures]}")
        return
    windows = [args.window] if args.window else ["PRE_MATCH", "HT"]
    for w in windows:
        execute_window(fx, w, engine=args.engine, market_mode=args.market,
                       profiles=profiles, session_dir=session_dir, refresh=args.refresh)


def cmd_run(args) -> None:
    fixtures = _load_fixtures_for_args(args)
    profiles = load_profiles(args.profiles)
    session_dir = session_dir_for(args.date, args.session)
    session_dir.mkdir(parents=True, exist_ok=True)
    broker.save_ledger(session_dir, broker.load_ledger(session_dir, profiles))
    performance.results_template(session_dir, fixtures)

    pairs = fx_mod.all_windows(fixtures)
    print(f"Live harness: {len(pairs)} windows across {len(fixtures)} fixtures "
          f"→ session {session_dir}")
    for fx, window in pairs:
        trigger = fx.window_trigger(window).astimezone(timezone.utc)
        now = datetime.now(timezone.utc)
        wait = (trigger - now).total_seconds()
        if wait > 0:
            if args.start_now:
                print(f"[start-now] skipping {wait/60:.1f} min wait for "
                      f"{fx.fixture_code} {window}")
            else:
                print(f"Waiting {wait/60:.1f} min for {fx.fixture_code} {window} "
                      f"(trigger {trigger.astimezone(fx_mod.LOCAL_TZ):%H:%M} CDT)…")
                _sleep_until(trigger, args.poll_seconds)
        else:
            print(f"[catch-up] {fx.fixture_code} {window} trigger already passed "
                  f"({-wait/60:.1f} min ago) — running now")
        try:
            execute_window(fx, window, engine=args.engine, market_mode=args.market,
                           profiles=profiles, session_dir=session_dir, refresh=args.refresh)
        except Exception as exc:
            print(f"  [error running {fx.fixture_code} {window}: {exc!r}] — continuing")
    print("\nAll windows processed. After matches finish, fill results.json and run: "
          "python -m harness settle")


def _sleep_until(trigger_utc, poll_seconds: float) -> None:
    while True:
        remaining = (trigger_utc - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(max(poll_seconds, 1.0), remaining))


def cmd_settle(args) -> None:
    profiles = load_profiles(args.profiles)
    session_dir = session_dir_for(args.date, args.session)
    ledger = broker.load_ledger(session_dir, profiles)
    results = performance.load_results(session_dir)
    if not results:
        print(f"No results yet. Edit {session_dir/'results.json'} "
              "(set result_slot to home|draw|away per fixture), then re-run settle.")
        return
    total = 0
    for code, slot in results.items():
        total += broker.settle(ledger, code, slot)
    broker.save_ledger(session_dir, ledger)
    print(f"Settled {total} open trades across {len(results)} fixtures.")
    cmd_report(args)


def cmd_report(args) -> None:
    session_dir = session_dir_for(args.date, args.session)
    out = performance.write_reports(session_dir)
    s = out["summary"]
    print(f"\nReport → {session_dir}")
    for name, a in s["agents"].items():
        roi = f"{a['roi']*100:+.1f}%" if a["roi"] is not None else "n/a"
        print(f"  {name:13s} bets={a['n_bets']:2d} won={a['n_won']:2d} "
              f"P&L=${a['pnl_total']:+.2f} ROI={roi} end=${a['ending_bankroll']:.2f}")
    if s["predictions"].get("overall"):
        o = s["predictions"]["overall"]
        print(f"  prediction calibration: Brier {o['brier']} acc {o['accuracy']*100:.0f}% (n={o['n']})")
    if out["plots"]:
        print(f"  plots: {', '.join(out['plots'])}")
    print(f"  csv: {out['csv']}")
