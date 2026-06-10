from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import re
from pathlib import Path

from agent.config import Settings, load_settings
from backtesting.runner import BacktestMatch, load_backtest_rows
from harness import paper_broker as broker
from harness.performance import write_reports
from harness.predictor import Prediction
from harness.profiles import load_profiles
from models.archetype import classify_match_archetype
from models.consensus import consensus_triangle
from models.draw_model import apply_draw_model
from models.probability import pre_match_model
from models.probability_blender import DEFAULT_PREMATCH_WEIGHTS
from models.source_reliability import dynamic_source_weights
from reasoning.council import run_council


def run_harness_backtest(
    *,
    dataset: str = "synthetic",
    sample_size: int = 20,
    session: str | None = None,
    profiles_path: str | None = None,
    engine: str = "deterministic",
    settings: Settings | None = None,
) -> dict:
    settings = settings or load_settings(dry_run_override=True)
    rows = load_backtest_rows(settings, dataset=dataset, sample_size=sample_size)
    profiles = load_profiles(profiles_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    session_name = session or f"backtest-{dataset}-{stamp}"
    session_dir = settings.storage_dir / "harness" / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    fixtures_payload = [_fixture_payload(row) for row in rows]
    (session_dir / "fixtures.json").write_text(json.dumps(fixtures_payload, indent=2), encoding="utf-8")
    (session_dir / "profiles.json").write_text(
        json.dumps({name: profile.to_dict() for name, profile in profiles.items()}, indent=2),
        encoding="utf-8",
    )
    (session_dir / "results.json").write_text(
        json.dumps(
            {
                row.fixture_code: {
                    "result_slot": row.result,
                    "home": row.home_team,
                    "away": row.away_team,
                    "source": row.source,
                }
                for row in rows
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ledger = broker.load_ledger(session_dir, profiles)
    match_rows = []
    for row in rows:
        home_code, away_code = _codes(row)
        prediction, model_detail = _predict_row(row, home_code, away_code, engine=engine)
        _write_prediction(session_dir, prediction)
        market = _moneyline(row, home_code, away_code)
        window_trades = []
        if _market_tradable(row):
            for profile in profiles.values():
                window_trades.extend(broker.decide_trades(profile, prediction, market, ledger))
        broker.settle(ledger, row.fixture_code, row.result)
        match_record = {
            "fixture_code": row.fixture_code,
            "window": "PRE_MATCH",
            "teams": f"{row.home_team} vs {row.away_team}",
            "home_team": row.home_team,
            "away_team": row.away_team,
            "home_code": home_code,
            "away_code": away_code,
            "kickoff_utc": row.kickoff_utc,
            "stage": row.stage,
            "match_week": row.match_week,
            "result": row.result,
            "source": row.source,
            "prediction": prediction.to_dict(),
            "model_detail": model_detail,
            "market": market,
            "inputs": {
                "sportmonks": row.sportmonks,
                "bookmaker": row.bookmaker,
                "market": row.market,
                "priors": row.priors,
                "lineup": row.lineup,
                "pre_state": row.pre_state,
                "odds": row.odds,
                "odds_quality": row.odds_quality,
            },
            "trades": window_trades,
        }
        match_rows.append(match_record)
        _append_jsonl(
            session_dir / "windows.jsonl",
            {
                "fixture_code": row.fixture_code,
                "window": "PRE_MATCH",
                "engine": prediction.engine,
                "market_source": market.get("market_source"),
                "n_trades": len(window_trades),
                "result": row.result,
                "ts": datetime.now(timezone.utc).isoformat(),
            },
        )

    broker.save_ledger(session_dir, ledger)
    (session_dir / "matches.json").write_text(json.dumps(match_rows, indent=2), encoding="utf-8")
    _write_match_csv(session_dir / "matches.csv", match_rows)
    report = write_reports(session_dir)
    _augment_summary(session_dir, dataset, sample_size, engine, rows, match_rows)
    return {
        "session_dir": str(session_dir),
        "matches": len(rows),
        "summary": report.get("summary"),
    }


def run_harness_backtest_comparison(
    *,
    dataset: str = "wc2022",
    sample_size: int = 20,
    session: str | None = None,
    profiles_path: str | None = None,
    settings: Settings | None = None,
) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    base = session or f"compare-{dataset}-{stamp}"
    deterministic = run_harness_backtest(
        dataset=dataset,
        sample_size=sample_size,
        session=f"{base}-deterministic",
        profiles_path=profiles_path,
        engine="deterministic",
        settings=settings,
    )
    council = run_harness_backtest(
        dataset=dataset,
        sample_size=sample_size,
        session=f"{base}-council",
        profiles_path=profiles_path,
        engine="council",
        settings=settings,
    )
    return {"deterministic": deterministic, "council": council}


def _predict_row(row: BacktestMatch, home_code: str, away_code: str, *, engine: str) -> tuple[Prediction, dict]:
    completeness = 1.0 if row.market and row.bookmaker and row.sportmonks and row.priors else 0.75
    archetype = classify_match_archetype(
        window="PRE_MATCH",
        sportmonks_probs=row.sportmonks,
        bookmaker_probs=row.bookmaker,
        market_probs=row.market,
        lineup=row.lineup or {},
        data_completeness={"score": completeness},
    )
    reliability = dynamic_source_weights(
        DEFAULT_PREMATCH_WEIGHTS,
        archetype=archetype,
        data_completeness={"score": completeness},
    )
    model = pre_match_model(
        row.sportmonks,
        row.bookmaker,
        row.market,
        row.priors,
        (row.lineup or {}).get("probability_delta"),
        completeness,
        weights=reliability["weights"],
    )
    draw = apply_draw_model(
        model["probabilities"],
        strength_gap=abs(model["probabilities"]["home"] - model["probabilities"]["away"]),
        market_draw=(row.market or {}).get("draw"),
        bookmaker_draw=(row.bookmaker or {}).get("draw"),
    )
    probs_hda = draw["probabilities"]
    code_probs = {
        home_code: round(float(probs_hda["home"]), 4),
        "draw": round(float(probs_hda["draw"]), 4),
        away_code: round(float(probs_hda["away"]), 4),
    }
    confidence = float(model.get("confidence", 0.55) or 0.55)
    if engine == "market":
        code_probs = {
            home_code: round(float(row.market.get("home", 0.4)), 4),
            "draw": round(float(row.market.get("draw", 0.28)), 4),
            away_code: round(float(row.market.get("away", 0.32)), 4),
        }
        confidence = 0.4
    prediction = Prediction(
        fixture_code=row.fixture_code,
        window="PRE_MATCH",
        home_code=home_code,
        away_code=away_code,
        probabilities=code_probs,
        confidence_label=_confidence_label(confidence),
        confidence_num=confidence,
        engine=f"historical_{engine}",
        scout_flags=[],
        note="historical no-leakage row converted into harness prediction",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    cons = consensus_triangle(probs_hda, row.bookmaker, row.market)
    detail = {
        "probabilities_hda": probs_hda,
        "confidence": confidence,
        "uncertainty": model.get("uncertainty"),
        "archetype": archetype,
        "source_reliability": reliability,
        "source_contribution": model.get("source_contribution"),
        "weights": model.get("weights"),
        "steps": model.get("steps"),
        "draw_model": draw,
        "consensus": cons,
    }
    if engine == "council":
        council_prediction, council_detail = _predict_council_row(row, home_code, away_code, fallback=prediction)
        return council_prediction, {**detail, "council": council_detail}
    return prediction, detail


def _predict_council_row(
    row: BacktestMatch,
    home_code: str,
    away_code: str,
    *,
    fallback: Prediction,
) -> tuple[Prediction, dict]:
    sportmonks_digest = {
        "source": row.source,
        "stage": row.stage,
        "match_week": row.match_week,
        "historical_model_probs": row.sportmonks,
        "lineup": row.lineup,
        "pre_state": row.pre_state,
    }
    supabase_digest = {
        "historical_priors": row.priors,
        "note": "No future leakage: priors are from pre-match state only.",
    }
    polymarket_digest = {
        "implied_win_prob": row.market,
        "market_handle": "checkbestodds_archive" if row.odds else "historical_rating_reference",
        "odds": row.odds,
        "odds_quality": row.odds_quality,
        "data_availability": "historical_archive",
    }
    try:
        council = run_council(
            f"{row.home_team} vs {row.away_team}",
            home_code,
            away_code,
            row.home_team,
            row.away_team,
            row.kickoff_utc or f"historical {row.stage}",
            sportmonks_digest,
            supabase_digest,
            polymarket_digest,
            None,
            None,
            None,
        )
        probs = _normalize_code_probs(council.probabilities, home_code, away_code)
        if not probs:
            raise ValueError(f"Council returned invalid probabilities: {council.probabilities}")
        confidence = _confidence_num(council.confidence)
        prediction = Prediction(
            fixture_code=row.fixture_code,
            window="PRE_MATCH",
            home_code=home_code,
            away_code=away_code,
            probabilities=probs,
            confidence_label=str(council.confidence or "low"),
            confidence_num=confidence,
            engine="historical_council",
            scout_flags=list(council.scout_flags or []),
            note=f"market_alignment={council.market_alignment}; {council.council_summary[:240]}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        return prediction, {
            "ok": True,
            "used_fallback": False,
            "outcome": council.outcome,
            "probability": council.probability,
            "confidence": council.confidence,
            "summary": council.council_summary,
            "market_alignment": council.market_alignment,
            "scout_flags": council.scout_flags,
            "social_pulse": council.social_pulse,
            "roles": {
                "pulse": _llm_result_payload(council.pulse),
                "scout": _llm_result_payload(council.scout),
                "analyst": _llm_result_payload(council.analyst),
                "devil": _llm_result_payload(council.devil),
                "judge": _llm_result_payload(council.judge),
            },
        }
    except Exception as exc:
        return fallback, {
            "ok": False,
            "used_fallback": True,
            "reason": repr(exc),
            "fallback_engine": fallback.engine,
        }


def _normalize_code_probs(probs: dict, home_code: str, away_code: str) -> dict | None:
    raw = {
        home_code: probs.get(home_code, probs.get("home")),
        "draw": probs.get("draw"),
        away_code: probs.get(away_code, probs.get("away")),
    }
    vals = {}
    for key, value in raw.items():
        if not isinstance(value, (int, float)):
            return None
        vals[key] = max(0.0, float(value))
    total = sum(vals.values())
    if total <= 0:
        return None
    return {key: round(value / total, 4) for key, value in vals.items()}


def _llm_result_payload(result) -> dict:
    if result is None:
        return {}
    return {
        "provider": getattr(result, "provider", ""),
        "model": getattr(result, "model", ""),
        "tokens_in": getattr(result, "tokens_in", 0),
        "tokens_out": getattr(result, "tokens_out", 0),
        "parsed": getattr(result, "parsed", {}) or {},
        "raw_text_preview": (getattr(result, "raw_text", "") or "")[:2000],
        "thinking_available": bool(getattr(result, "thinking", "")),
    }


def _moneyline(row: BacktestMatch, home_code: str, away_code: str) -> dict:
    if row.odds and _market_tradable(row):
        mids = {
            "home": round(1.0 / max(float(row.odds["home"]), 1.01), 4),
            "draw": round(1.0 / max(float(row.odds["draw"]), 1.01), 4),
            "away": round(1.0 / max(float(row.odds["away"]), 1.01), 4),
        }
        source = "checkbestodds_archive"
    else:
        mids = {slot: round(float(row.market.get(slot, 0.0)), 4) for slot in ("home", "draw", "away")}
        source = "historical_rating_or_synthetic_reference"
    return {
        "fixture": f"{row.home_team} vs {row.away_team}",
        "market_source": source,
        "odds_quality": row.odds_quality,
        "outcomes": {
            "home": {"team_code": home_code, "current_mid_yes": mids["home"]},
            "draw": {"team_code": "draw", "current_mid_yes": mids["draw"]},
            "away": {"team_code": away_code, "current_mid_yes": mids["away"]},
        },
    }


def _market_tradable(row: BacktestMatch) -> bool:
    return bool((row.odds_quality or {"tradable": True}).get("tradable", True))


def _write_prediction(session_dir: Path, prediction: Prediction) -> None:
    path = session_dir / "predictions" / f"{prediction.fixture_code}-{prediction.window}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prediction.to_dict(), indent=2), encoding="utf-8")


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _write_match_csv(path: Path, rows: list[dict]) -> None:
    cols = [
        "fixture_code",
        "teams",
        "stage",
        "result",
        "engine",
        "home_prob",
        "draw_prob",
        "away_prob",
        "market_home",
        "market_draw",
        "market_away",
        "n_trades",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            pred = row["prediction"]
            inputs = row["inputs"]
            writer.writerow(
                {
                    "fixture_code": row["fixture_code"],
                    "teams": row["teams"],
                    "stage": row["stage"],
                    "result": row["result"],
                    "engine": pred["engine"],
                    "home_prob": pred["probabilities"][row["home_code"]],
                    "draw_prob": pred["probabilities"]["draw"],
                    "away_prob": pred["probabilities"][row["away_code"]],
                    "market_home": inputs["market"].get("home"),
                    "market_draw": inputs["market"].get("draw"),
                    "market_away": inputs["market"].get("away"),
                    "n_trades": len(row["trades"]),
                }
            )


def _augment_summary(
    session_dir: Path,
    dataset: str,
    sample_size: int,
    engine: str,
    rows: list[BacktestMatch],
    match_rows: list[dict],
) -> None:
    path = session_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary["backtest"] = {
        "dataset": dataset,
        "sample_size": sample_size,
        "engine": engine,
        "historical_source": "StatsBomb open data + CheckBestOdds archive" if dataset.startswith("wc") else "synthetic generated history",
        "matches": len(rows),
        "real_results": sum(1 for row in rows if row.source != "synthetic"),
        "tradable_markets": sum(1 for row in rows if _market_tradable(row)),
        "lineup_rows": sum(1 for row in rows if row.lineup and "lineup_unconfirmed" not in (row.lineup.get("risk_flags") or [])),
        "match_file": "matches.json",
        "match_csv": "matches.csv",
    }
    if engine == "council":
        council_details = [((row.get("model_detail") or {}).get("council") or {}) for row in match_rows]
        summary["backtest"]["council"] = {
            "calls_ok": sum(1 for detail in council_details if detail.get("ok")),
            "fallbacks": sum(1 for detail in council_details if detail.get("used_fallback")),
            "roles": ["pulse", "scout", "analyst", "devil", "judge"],
            "historical_leakage_policy": "No live web or Reddit inputs are passed during WC2022 replay.",
        }
    summary["match_preview"] = match_rows[:5]
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _fixture_payload(row: BacktestMatch) -> dict:
    home_code, away_code = _codes(row)
    return {
        "fixture_code": row.fixture_code,
        "home": row.home_team,
        "away": row.away_team,
        "home_code": home_code,
        "away_code": away_code,
        "kickoff_utc": row.kickoff_utc,
        "stage": row.stage,
        "match_week": row.match_week,
        "source": row.source,
    }


def _codes(row: BacktestMatch) -> tuple[str, str]:
    return (_team_code(row.home_team), _team_code(row.away_team))


def _team_code(name: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", name.upper())
    if not words:
        return "UNK"
    if len(words) == 1:
        return words[0][:3].ljust(3, "X")
    return "".join(word[0] for word in words)[:3].ljust(3, "X")


def _confidence_label(value: float) -> str:
    if value >= 0.72:
        return "high"
    if value >= 0.55:
        return "medium"
    return "low"


def _confidence_num(value: str | float | None) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    label = str(value or "").lower()
    if label == "high":
        return 0.8
    if label == "medium":
        return 0.6
    return 0.4
