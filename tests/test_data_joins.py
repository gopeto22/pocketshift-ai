"""Join integrity: games↔plays, plays↔pff, nflId resolution, NA safety."""
from __future__ import annotations

import pandas as pd

from src import config as C
from src import validation


def test_all_validation_checks_pass(raw):
    rep = validation.validate(raw)
    failed = [c.name for c in rep.checks if not c.passed]
    assert rep.ok, f"failed checks: {failed}"


def test_games_join_to_plays(raw):
    missing = (~raw.plays["gameId"].isin(set(raw.games["gameId"]))).sum()
    assert missing == 0


def test_pff_joins_on_composite_key(raw):
    play_keys = set(map(tuple, raw.plays[["gameId", "playId"]].to_numpy()))
    pff_keys = set(map(tuple, raw.pff[["gameId", "playId"]].to_numpy()))
    assert pff_keys <= play_keys


def test_player_ids_resolve(raw):
    player_ids = set(raw.players["nflId"].dropna().astype("int64"))
    unresolved = (~raw.pff["nflId"].dropna().astype("int64").isin(player_ids)).sum()
    assert unresolved == 0


def test_missing_values_do_not_crash_parsing(raw):
    # NA strings became real nulls, not the literal string "NA".
    assert not (raw.pff["pff_hit"].astype("string") == "NA").any()
    assert raw.pff["pff_hit"].isna().sum() > 0  # non-rushers are null, as expected


def test_role_labels_normalise(raw):
    roles = set(raw.pff["pff_role"].dropna().map(C.normalize_role).unique())
    assert roles <= {
        C.ROLE_PASS_RUSH,
        C.ROLE_PASS_BLOCK,
        C.ROLE_PASS_ROUTE,
        C.ROLE_COVERAGE,
        C.ROLE_PASS,
    }


def test_do_not_join_on_playid_alone(raw):
    # playId is not unique across games — proves the composite key is necessary.
    assert raw.plays["playId"].duplicated().any()
