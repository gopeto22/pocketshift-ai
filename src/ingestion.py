"""CSV ingestion, NA handling, strict joins, and processed-table builds.

Everything here is deterministic and Streamlit-agnostic so it can be unit
tested and reused by scripts, the app, and the agent tools.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .features import (
    add_pressure_labels,
    build_matchups,
    build_player_profiles,
    build_team_patterns,
)

# PFF binary outcome columns that use "NA" for not-applicable rows.
_PFF_BINARY_COLS = [
    "pff_hit",
    "pff_hurry",
    "pff_sack",
    "pff_beatenByDefender",
    "pff_hitAllowed",
    "pff_hurryAllowed",
    "pff_sackAllowed",
    "pff_backFieldBlock",
]


class DataMissingError(RuntimeError):
    """Raised when a required source CSV is absent."""


def _require(name: str) -> None:
    path = C.RAW_CSVS[name]
    if not path.exists():
        raise DataMissingError(
            f"Required dataset file missing: {path}. "
            "See data/README.md for download/setup instructions. "
            "PocketShift never fabricates missing data."
        )


# ---------------------------------------------------------------------------
# Raw loaders
# ---------------------------------------------------------------------------
def load_games() -> pd.DataFrame:
    _require("games")
    df = pd.read_csv(C.GAMES_CSV, na_values=["NA"])
    df["gameId"] = df["gameId"].astype("int64")
    return df


def load_players() -> pd.DataFrame:
    _require("players")
    df = pd.read_csv(C.PLAYERS_CSV, na_values=["NA"])
    df["nflId"] = pd.to_numeric(df["nflId"], errors="coerce").astype("Int64")
    df["displayName"] = df["displayName"].astype("string")
    df["officialPosition"] = df["officialPosition"].astype("string")
    return df


def load_plays() -> pd.DataFrame:
    _require("plays")
    df = pd.read_csv(C.PLAYS_CSV, na_values=["NA"])
    df["gameId"] = df["gameId"].astype("int64")
    df["playId"] = df["playId"].astype("int64")
    for col in ("down", "quarter", "yardsToGo", "defendersInBox"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Pre-snap score differential from the possession team's perspective is not
    # directly available (home/visitor only), so we expose both and a neutral
    # absolute margin usable as a pre-snap feature.
    df["score_diff"] = (
        pd.to_numeric(df["preSnapHomeScore"], errors="coerce")
        - pd.to_numeric(df["preSnapVisitorScore"], errors="coerce")
    )
    df["is_sack"] = (df["passResult"].astype("string") == "S").astype("int8")
    return df


def load_pff() -> pd.DataFrame:
    _require("pff")
    df = pd.read_csv(C.PFF_CSV, na_values=["NA"])
    df["gameId"] = df["gameId"].astype("int64")
    df["playId"] = df["playId"].astype("int64")
    df["nflId"] = pd.to_numeric(df["nflId"], errors="coerce").astype("Int64")
    df["pff_nflIdBlockedPlayer"] = pd.to_numeric(
        df["pff_nflIdBlockedPlayer"], errors="coerce"
    ).astype("Int64")
    for col in _PFF_BINARY_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int8")
    df["pff_role_norm"] = df["pff_role"].map(C.normalize_role).astype("string")
    df["pff_role"] = df["pff_role"].astype("string")
    df["pff_positionLinedUp"] = df["pff_positionLinedUp"].astype("string")
    df["pff_blockType"] = df["pff_blockType"].astype("string")
    return df


@dataclass
class RawData:
    games: pd.DataFrame
    plays: pd.DataFrame
    players: pd.DataFrame
    pff: pd.DataFrame


def load_raw() -> RawData:
    return RawData(
        games=load_games(),
        plays=load_plays(),
        players=load_players(),
        pff=load_pff(),
    )


# ---------------------------------------------------------------------------
# Joins (strict composite keys)
# ---------------------------------------------------------------------------
def join_plays_games(plays: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """plays.gameId -> games.gameId. Left join keeps every play."""
    keep = ["gameId", "season", "week", "gameDate", "homeTeamAbbr", "visitorTeamAbbr"]
    return plays.merge(games[keep], on="gameId", how="left", validate="many_to_one")


def attach_player_names(pff: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """pff.nflId -> players, plus the blocked-player name when populated."""
    pl = players[["nflId", "displayName", "officialPosition"]].copy()
    out = pff.merge(pl, on="nflId", how="left", validate="many_to_one")
    blocked = pl.rename(
        columns={
            "nflId": "pff_nflIdBlockedPlayer",
            "displayName": "blockedPlayerName",
            "officialPosition": "blockedPlayerPosition",
        }
    )
    out = out.merge(
        blocked, on="pff_nflIdBlockedPlayer", how="left", validate="many_to_one"
    )
    return out


# ---------------------------------------------------------------------------
# Processed-table build
# ---------------------------------------------------------------------------
@dataclass
class ProcessedTables:
    plays_enriched: pd.DataFrame
    matchups: pd.DataFrame
    player_profiles: pd.DataFrame
    team_patterns: pd.DataFrame


def build_processed(raw: RawData | None = None) -> ProcessedTables:
    raw = raw or load_raw()
    pff_named = attach_player_names(raw.pff, raw.players)

    plays_g = join_plays_games(raw.plays, raw.games)
    plays_enriched = add_pressure_labels(plays_g, raw.pff)

    matchups = build_matchups(pff_named, plays_enriched)
    player_profiles = build_player_profiles(pff_named, raw.players, plays_enriched)
    team_patterns = build_team_patterns(plays_enriched)

    return ProcessedTables(
        plays_enriched=plays_enriched,
        matchups=matchups,
        player_profiles=player_profiles,
        team_patterns=team_patterns,
    )


def write_processed(tables: ProcessedTables) -> dict[str, str]:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "plays_enriched": C.PLAYS_ENRICHED_PARQUET,
        "matchups": C.MATCHUPS_PARQUET,
        "player_profiles": C.PLAYER_PROFILES_PARQUET,
        "team_patterns": C.TEAM_PATTERNS_PARQUET,
    }
    tables.plays_enriched.to_parquet(paths["plays_enriched"], index=False)
    tables.matchups.to_parquet(paths["matchups"], index=False)
    tables.player_profiles.to_parquet(paths["player_profiles"], index=False)
    tables.team_patterns.to_parquet(paths["team_patterns"], index=False)
    return {k: str(v) for k, v in paths.items()}


def processed_exists() -> bool:
    return all(
        p.exists()
        for p in (
            C.PLAYS_ENRICHED_PARQUET,
            C.MATCHUPS_PARQUET,
            C.PLAYER_PROFILES_PARQUET,
            C.TEAM_PATTERNS_PARQUET,
        )
    )


def load_processed(rebuild: bool = False) -> ProcessedTables:
    """Load processed Parquet tables, building them once if needed."""
    if rebuild or not processed_exists():
        tables = build_processed()
        write_processed(tables)
        return tables
    return ProcessedTables(
        plays_enriched=pd.read_parquet(C.PLAYS_ENRICHED_PARQUET),
        matchups=pd.read_parquet(C.MATCHUPS_PARQUET),
        player_profiles=pd.read_parquet(C.PLAYER_PROFILES_PARQUET),
        team_patterns=pd.read_parquet(C.TEAM_PATTERNS_PARQUET),
    )
