"""Scenario comparison: real matches, support gating, empty-group safety."""
from __future__ import annotations

import pandas as pd

from src import config as C
from src import scenarios as S


def test_similar_plays_return_real_matches(tables):
    pe = tables.plays_enriched
    res = S.find_similar_plays(pe, down_bucket="3rd", distance_bucket="long(8+)")
    assert len(res) > 0
    assert (res["down_bucket"] == "3rd").all()
    assert (res["distance_bucket"] == "long(8+)").all()


def test_supported_scenario_reports_sample_size(tables):
    res = S.compare_protection_configurations(tables.plays_enriched, "extra_protection")
    assert res.supported
    assert res.arm_a["n"] >= C.MIN_SUPPORT
    assert res.arm_b["n"] >= C.MIN_SUPPORT
    assert "n=" in res.message


def test_unsupported_scenario_is_flagged(tables):
    pe = tables.plays_enriched
    # Over-constrain to a tiny context to force insufficient support.
    tiny = {
        "down_bucket": "4th",
        "distance_bucket": "short(1-3)",
        "offense_formation": "JUMBO",
    }
    res = S.compare_protection_configurations(pe, "extra_protection", match_filters=tiny)
    assert not res.supported
    assert "Not enough comparable historical evidence" in res.message
    assert res.diff is None


def test_empty_group_returns_no_rate():
    empty = pd.DataFrame({"pressure_event": pd.Series([], dtype="float64")})
    out = S.empirical_pressure_rate(empty)
    assert out["n"] == 0
    assert out["rate"] is None
    assert out["supported"] is False


def test_bootstrap_ci_present_when_supported(tables):
    res = S.compare_protection_configurations(tables.plays_enriched, "play_action")
    if res.supported:
        lo, hi = res.diff_ci
        assert lo is not None and hi is not None
        assert lo <= res.diff <= hi
