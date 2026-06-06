from models.source_reconciliation import reconcile_sources


def test_source_reconciliation_flags_high_divergence():
    out = reconcile_sources(
        {"home": .40, "draw": .30, "away": .30},
        {"home": .61, "draw": .20, "away": .19},
        {"home": .34, "draw": .28, "away": .38},
        {"home": .35, "draw": .28, "away": .37},
    )
    assert "source_divergence_high" in out["flags"]
    assert out["max_gap"] >= .20


def test_source_reconciliation_no_flag_for_small_gaps():
    out = reconcile_sources(
        {"home": .44, "draw": .28, "away": .28},
        {"home": .46, "draw": .27, "away": .27},
        {"home": .45, "draw": .28, "away": .27},
        {"home": .43, "draw": .29, "away": .28},
    )
    assert "source_divergence_high" not in out["flags"]
