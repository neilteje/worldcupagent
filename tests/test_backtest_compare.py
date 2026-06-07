from agent.config import load_settings
from backtesting.runner import StrategyResult, _pick_winner, _strategy_summary, settle_bet


def test_strategy_summary_fields():
    result = StrategyResult(
        mode="deterministic",
        ending_bankroll=5.5,
        roi=0.1,
        model_brier=0.6123,
        market_brier=0.625,
        bets=3,
        fallback_count=0,
        blocked_by_llm=0,
        decisions=[],
    )
    summary = _strategy_summary(result)
    assert summary["mode"] == "deterministic"
    assert summary["roi"] == 0.1
    assert summary["bets"] == 3


def test_pick_winner_prefers_better_roi_and_brier():
    det = StrategyResult("deterministic", 5.2, 0.04, 0.61, 0.63, 4, 0, 0, [])
    llm = StrategyResult("llm_central", 5.6, 0.12, 0.59, 0.63, 3, 0, 1, [])
    assert _pick_winner(det, llm) == "llm_central"


def test_settle_bet_uses_decimal_odds_when_available():
    assert round(settle_bet(1.5, "home", "home", {"home": 0.5}, {"home": 2.4}), 2) == 2.1
    assert settle_bet(1.5, "home", "away", {"home": 0.5}, {"home": 2.4}) == -1.5
