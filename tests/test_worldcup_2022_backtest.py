from pathlib import Path

from backtesting.runner import BacktestMatch, _dataset_audit, load_backtest_rows
from harness.backtest import _component_report
from models.probability_blender import DEFAULT_PREMATCH_WEIGHTS
from models.source_reliability import dynamic_source_weights
from backtesting.worldcup_2022 import implied_probs_from_odds, odds_reference_quality, parse_checkbestodds_html, team_key
from agent.config import load_settings


def test_parse_checkbestodds_rows_from_html_snippet():
    html = """
    <tr> <td class="l2 match"> <span ts="1668963600" class="time hM">18:00</span>
    <a href="/football-odds/world-cup-2022/qatar-ecuador-2022-11-20/1543436804"> Qatar -  Ecuador</a></td>
    <td class="r"> <b class="">100.00</b></td> <td class="r"> <b class="">13.50</b></td>
    <td class="r"> <b class="">2.41</b></td></tr>
    """
    rows = parse_checkbestodds_html(html)
    assert len(rows) == 1
    assert rows[0]["home_team"] == "Qatar"
    assert rows[0]["away_team"] == "Ecuador"
    assert rows[0]["away_odds"] == 2.41


def test_implied_probs_from_odds_normalizes():
    probs = implied_probs_from_odds({"home": 2.0, "draw": 4.0, "away": 4.0})
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["home"] > probs["draw"]


def test_odds_reference_quality_blocks_archive_outliers():
    quality = odds_reference_quality({"home": 100.0, "draw": 13.5, "away": 2.41})
    assert quality["tradable"] is False
    assert "extreme_best_price_outlier" in quality["flags"]


def test_odds_reference_quality_allows_plausible_rows():
    quality = odds_reference_quality({"home": 2.0, "draw": 3.5, "away": 4.0})
    assert quality["tradable"] is True


def test_team_aliases_cover_world_cup_names():
    assert team_key("USA") == "united states"
    assert team_key("South Korea") == "korea republic"
    assert team_key("IR Iran") == "iran"


def test_dataset_audit_marks_synthetic_rows():
    rows = [
        BacktestMatch(
            "F",
            {"home": 0.4, "draw": 0.3, "away": 0.3},
            {"home": 0.4, "draw": 0.3, "away": 0.3},
            {"home": 0.4, "draw": 0.3, "away": 0.3},
            {"home": 0.4, "draw": 0.3, "away": 0.3},
            "home",
            source="synthetic",
        )
    ]
    audit = _dataset_audit(rows)
    assert audit["synthetic_rows"] == 1
    assert audit["real_match_results"] is False


def test_load_synthetic_rows_still_works(tmp_path):
    settings = type(load_settings(True))(dry_run=True, storage_dir=tmp_path)
    rows = load_backtest_rows(settings, dataset="synthetic", sample_size=3)
    assert len(rows) == 3


def test_strong_favorite_consensus_damps_neutral_priors():
    out = dynamic_source_weights(
        DEFAULT_PREMATCH_WEIGHTS,
        archetype={
            "match_archetype": "strong_favorite",
            "market_regime": "market_consensus",
            "tags": ["rich_data", "large_favorite_gap"],
        },
        data_completeness={"score": 1.0},
    )
    assert out["weights"]["bookmaker"] > DEFAULT_PREMATCH_WEIGHTS["bookmaker"]
    assert out["weights"]["sportmonks"] < DEFAULT_PREMATCH_WEIGHTS["sportmonks"]
    assert any("longshot inflation" in reason for reason in out["reasons"])


def test_component_report_summarizes_match_details():
    report = _component_report([
        {
            "home_code": "AAA",
            "away_code": "BBB",
            "result": "home",
            "prediction": {"probabilities": {"AAA": 0.6, "draw": 0.2, "BBB": 0.2}},
            "model_detail": {
                "archetype": {"match_archetype": "strong_favorite"},
                "weights": {"bookmaker": 0.4, "sportmonks": 0.2},
                "source_contribution": {"bookmaker": {"home": 0.3, "draw": 0.05, "away": 0.05}},
                "steps": [
                    {"name": "source_blend", "probabilities": {"home": 0.55, "draw": 0.25, "away": 0.2}},
                    {"name": "calibration", "probabilities": {"home": 0.6, "draw": 0.2, "away": 0.2}},
                ],
                "draw_model": {"delta": {"home": 0.01, "draw": -0.02, "away": 0.01}, "reason": "large favorite gap"},
                "consensus": {"case": "all_agree"},
                "profile_reports": {"aggressive": {"picked_codes": ["AAA"], "skip_reasons": []}},
            },
        }
    ])
    assert report["accuracy_by_archetype"]["strong_favorite"]["accuracy"] == 1.0
    assert report["avg_source_weights"]["bookmaker"] == 0.4
    assert report["draw_model"]["reasons"]["large favorite gap"] == 1
