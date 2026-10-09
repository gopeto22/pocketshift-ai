"""Tracking-derived energy & fatigue metric tests.

These are skipped automatically if tracking files are absent, so the suite stays
green on a checkout without the optional tracking data.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import config as C
from src import tracking as TR

pytestmark = pytest.mark.skipif(
    not TR.tracking_available(), reason="tracking files not present"
)


@pytest.fixture(scope="module")
def effort():
    # Build only the featured game for a fast, deterministic test.
    pe, fat = TR.build_effort_tables(max_games=1)
    return pe, fat


def test_effort_is_snap_gated_and_positive(effort):
    pe, _ = effort
    assert len(pe) > 0
    # Featured game present.
    assert (pe["gameId"] == C.FEATURED_GAME_ID).any()
    # Energy index is non-negative and finite.
    assert (pe["eei"] >= 0).all()
    assert pe["eei"].notna().all()
    assert (pe["frames"] > 0).all()


def test_energy_index_rewards_distance(effort):
    pe, _ = effort
    # EEI should correlate strongly with distance covered.
    corr = pe["eei"].corr(pe["distance_yd"])
    assert corr > 0.8


def test_fatigue_flag_requires_decline_and_load(effort):
    _, fat = effort
    if fat.empty:
        pytest.skip("no multi-snap players in single-game subset")
    flagged = fat[fat["fatigue_flag"] == 1]
    # Every flagged player really has a >=10% late accel decline and >=15 snaps.
    assert (flagged["accel_decline_pct"] >= 10.0).all()
    assert (flagged["snaps"] >= 15).all()


def test_fatigue_proxy_is_labelled_not_medical():
    # The metric docstring and config make clear this is a kinematic proxy.
    assert "proxy" in TR.__doc__.lower()
    assert "not a physiological" in TR.__doc__.lower() or \
        "no calorie" in TR.__doc__.lower()


def test_player_game_curve_shapes(effort):
    pe, _ = effort
    nid = int(pe[pe["gameId"] == C.FEATURED_GAME_ID]["nflId"].iloc[0])
    curve = TR.player_game_curve(pe, C.FEATURED_GAME_ID, nid)
    assert len(curve.play_order) == len(curve.cum_eei) == len(curve.peak_accel)
    # Cumulative EEI is non-decreasing.
    assert all(b >= a - 1e-9 for a, b in zip(curve.cum_eei, curve.cum_eei[1:]))
