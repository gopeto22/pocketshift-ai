"""Pressure-event labelling + no pre-snap leakage."""
from __future__ import annotations

import pandas as pd
import pytest

from src import config as C
from src.features import build_model_frame


def test_pressure_label_uses_rush_fields(tables, pff_named):
    pe = tables.plays_enriched
    fp = pe[(pe.gameId == C.FEATURED_GAME_ID) & (pe.playId == C.FEATURED_PLAY_ID)].iloc[0]
    # Featured play is a sack → pressure + sack event.
    assert fp.pressure_event == 1
    assert fp.sack_event == 1
    assert fp.n_rushers >= 1


def test_pressure_event_implies_a_rush_outcome(tables, pff_named):
    pe = tables.plays_enriched
    sample = pe[pe.pressure_event == 1].head(50)
    role = pff_named["pff_role"].map(C.normalize_role)
    rush = pff_named[role == C.ROLE_PASS_RUSH]
    for r in sample.itertuples():
        rr = rush[(rush.gameId == r.gameId) & (rush.playId == r.playId)]
        got = (
            pd.to_numeric(rr["pff_hit"], errors="coerce").fillna(0).sum()
            + pd.to_numeric(rr["pff_hurry"], errors="coerce").fillna(0).sum()
            + pd.to_numeric(rr["pff_sack"], errors="coerce").fillna(0).sum()
        )
        assert got >= 1


def test_no_pressure_implies_no_rush_outcome(tables, pff_named):
    pe = tables.plays_enriched
    role = pff_named["pff_role"].map(C.normalize_role)
    rush = pff_named[role == C.ROLE_PASS_RUSH]
    sample = pe[pe.pressure_event == 0].head(50)
    for r in sample.itertuples():
        rr = rush[(rush.gameId == r.gameId) & (rush.playId == r.playId)]
        got = (
            pd.to_numeric(rr["pff_hit"], errors="coerce").fillna(0).sum()
            + pd.to_numeric(rr["pff_hurry"], errors="coerce").fillna(0).sum()
            + pd.to_numeric(rr["pff_sack"], errors="coerce").fillna(0).sum()
        )
        assert got == 0


def test_model_frame_has_no_outcome_leakage(tables):
    frame = build_model_frame(tables.plays_enriched)
    forbidden = C.POST_SNAP_PLAY_COLS | C.POST_SNAP_PFF_COLS | C.RETROSPECTIVE_PLAY_COLS
    leaked = forbidden & set(frame.columns) - {"pressure_event"}
    # pressure_event is the TARGET; everything else must be pre-snap.
    assert not leaked, f"leaked post-snap columns: {leaked}"


def test_feature_list_is_presnap_only():
    bad = (C.POST_SNAP_PLAY_COLS | C.POST_SNAP_PFF_COLS) & set(C.PRESNAP_FEATURES)
    assert not bad
