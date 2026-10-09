"""Deterministic data-quality checks over the raw and joined tables."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config as C
from . import ingestion


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(Check(name, passed, detail))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"check": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks]
        )


def validate(raw: ingestion.RawData | None = None) -> ValidationReport:
    raw = raw or ingestion.load_raw()
    rep = ValidationReport()

    # 1. gameId uniqueness in games.
    dup_games = raw.games["gameId"].duplicated().sum()
    rep.add("games.gameId unique", dup_games == 0, f"{dup_games} duplicates")

    # 2. plays composite key uniqueness.
    dup_plays = raw.plays.duplicated(["gameId", "playId"]).sum()
    rep.add("plays (gameId,playId) unique", dup_plays == 0, f"{dup_plays} dups")

    # 3. every play's game exists.
    missing_games = (~raw.plays["gameId"].isin(set(raw.games["gameId"]))).sum()
    rep.add("plays join to games", missing_games == 0, f"{missing_games} orphan plays")

    # 4. PFF rows join to a real play.
    play_keys = set(map(tuple, raw.plays[["gameId", "playId"]].to_numpy()))
    pff_keys = raw.pff[["gameId", "playId"]].to_numpy()
    orphan_pff = sum(1 for k in map(tuple, pff_keys) if k not in play_keys)
    rep.add("pff join to plays", orphan_pff == 0, f"{orphan_pff} orphan pff rows")

    # 5. nflId resolution (non-null nflId must resolve to a player).
    player_ids = set(raw.players["nflId"].dropna().astype("int64"))
    pff_nfl = raw.pff["nflId"].dropna().astype("int64")
    unresolved = (~pff_nfl.isin(player_ids)).sum()
    rep.add("pff.nflId resolves", unresolved == 0, f"{unresolved} unresolved")

    # 6. blocked-player id resolves when populated.
    blk = raw.pff["pff_nflIdBlockedPlayer"].dropna().astype("int64")
    blk_unres = (~blk.isin(player_ids)).sum()
    rep.add(
        "blocked-player id resolves",
        blk_unres == 0,
        f"{blk_unres}/{len(blk)} populated unresolved",
    )

    # 7. role labels normalise to the known set.
    roles = set(raw.pff["pff_role"].dropna().map(C.normalize_role).unique())
    known = {
        C.ROLE_PASS_RUSH,
        C.ROLE_PASS_BLOCK,
        C.ROLE_PASS_ROUTE,
        C.ROLE_COVERAGE,
        C.ROLE_PASS,
    }
    rep.add("pff roles recognised", roles <= known, f"unexpected={roles - known}")

    # 8. featured play present with PFF rows.
    fp = raw.plays[
        (raw.plays["gameId"] == C.FEATURED_GAME_ID)
        & (raw.plays["playId"] == C.FEATURED_PLAY_ID)
    ]
    fp_pff = raw.pff[
        (raw.pff["gameId"] == C.FEATURED_GAME_ID)
        & (raw.pff["playId"] == C.FEATURED_PLAY_ID)
    ]
    rep.add(
        "featured play present",
        len(fp) == 1 and len(fp_pff) > 0,
        f"play rows={len(fp)}, pff rows={len(fp_pff)}",
    )

    return rep


def dataset_summary(raw: ingestion.RawData | None = None) -> dict:
    raw = raw or ingestion.load_raw()
    role_counts = (
        raw.pff["pff_role"].dropna().map(C.normalize_role).value_counts().to_dict()
    )
    return {
        "n_games": int(raw.games["gameId"].nunique()),
        "n_plays": int(len(raw.plays)),
        "n_players": int(raw.players["nflId"].nunique()),
        "n_pff_rows": int(len(raw.pff)),
        "n_teams": int(
            pd.concat([raw.plays["possessionTeam"], raw.plays["defensiveTeam"]])
            .nunique()
        ),
        "role_counts": role_counts,
    }
