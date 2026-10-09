"""Feature engineering: pressure labels, matchup edges, profiles, team patterns,
and the leakage-guarded pre-snap model frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Pressure labelling
# ---------------------------------------------------------------------------
def _play_pressure_from_pff(pff: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pass-rush outcomes to one row per (gameId, playId).

    pressure_event = any Pass Rush row with hit|hurry|sack == 1.
    This is a scouting-derived proxy for recorded pressure involvement.
    """
    role = pff["pff_role"].map(C.normalize_role)
    rush = pff[role == C.ROLE_PASS_RUSH].copy()
    for col in ("pff_hit", "pff_hurry", "pff_sack"):
        rush[col] = pd.to_numeric(rush[col], errors="coerce").fillna(0)

    grp = rush.groupby(["gameId", "playId"], as_index=False).agg(
        n_rushers=("nflId", "nunique"),
        hits=("pff_hit", "sum"),
        hurries=("pff_hurry", "sum"),
        sacks=("pff_sack", "sum"),
    )
    grp["pressure_event"] = (
        (grp["hits"] > 0) | (grp["hurries"] > 0) | (grp["sacks"] > 0)
    ).astype("int8")
    grp["sack_event"] = (grp["sacks"] > 0).astype("int8")

    blocks = pff[role == C.ROLE_PASS_BLOCK]
    nblk = blocks.groupby(["gameId", "playId"], as_index=False).agg(
        n_blockers=("nflId", "nunique")
    )
    out = grp.merge(nblk, on=["gameId", "playId"], how="outer")
    out["n_rushers"] = out["n_rushers"].fillna(0).astype("int16")
    out["n_blockers"] = out["n_blockers"].fillna(0).astype("int16")
    for col in ("hits", "hurries", "sacks", "pressure_event", "sack_event"):
        out[col] = out[col].fillna(0).astype("int16")
    out["extra_protection"] = (
        out["n_blockers"] >= C.EXTRA_PROTECTION_BLOCKER_THRESHOLD
    ).astype("int8")
    return out


def _passer_name(pff: pd.DataFrame, players: pd.DataFrame | None) -> pd.DataFrame:
    role = pff["pff_role"].map(C.normalize_role)
    passers = pff[role == C.ROLE_PASS][["gameId", "playId", "nflId"]].copy()
    passers = passers.rename(columns={"nflId": "passerNflId"})
    passers = passers.drop_duplicates(["gameId", "playId"])
    return passers


def add_pressure_labels(plays: pd.DataFrame, pff: pd.DataFrame) -> pd.DataFrame:
    """Return plays with pressure columns + distance/down buckets for matching."""
    agg = _play_pressure_from_pff(pff)
    passers = _passer_name(pff, None)
    out = plays.merge(agg, on=["gameId", "playId"], how="left")
    out = out.merge(passers, on=["gameId", "playId"], how="left")

    for col in (
        "n_rushers",
        "n_blockers",
        "hits",
        "hurries",
        "sacks",
        "pressure_event",
        "sack_event",
        "extra_protection",
    ):
        out[col] = out[col].fillna(0).astype("int16")

    # Matching buckets (controllable context).
    out["down_bucket"] = out["down"].map(
        {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    ).astype("string")
    ytg = pd.to_numeric(out["yardsToGo"], errors="coerce")
    out["distance_bucket"] = pd.cut(
        ytg,
        bins=[-0.1, 3, 7, 100],
        labels=["short(1-3)", "medium(4-7)", "long(8+)"],
    ).astype("string")
    return out


# ---------------------------------------------------------------------------
# Blocker -> rusher interaction edges
# ---------------------------------------------------------------------------
def build_matchups(pff_named: pd.DataFrame, plays_enriched: pd.DataFrame) -> pd.DataFrame:
    """One row per recorded blocking assignment (blocker -> blocked defender).

    An interaction network edge, NOT a spatial reconstruction.
    """
    role = pff_named["pff_role"].map(C.normalize_role)
    blockers = pff_named[role == C.ROLE_PASS_BLOCK].copy()
    cols = [
        "gameId",
        "playId",
        "nflId",
        "displayName",
        "officialPosition",
        "pff_positionLinedUp",
        "pff_blockType",
        "pff_backFieldBlock",
        "pff_hitAllowed",
        "pff_hurryAllowed",
        "pff_sackAllowed",
        "pff_beatenByDefender",
        "pff_nflIdBlockedPlayer",
        "blockedPlayerName",
        "blockedPlayerPosition",
    ]
    cols = [c for c in cols if c in blockers.columns]
    m = blockers[cols].rename(
        columns={
            "nflId": "blockerNflId",
            "displayName": "blockerName",
            "officialPosition": "blockerPosition",
            "pff_positionLinedUp": "blockerLinedUp",
            "pff_nflIdBlockedPlayer": "defenderNflId",
            "blockedPlayerName": "defenderName",
            "blockedPlayerPosition": "defenderPosition",
        }
    )
    for col in (
        "pff_hitAllowed",
        "pff_hurryAllowed",
        "pff_sackAllowed",
        "pff_beatenByDefender",
    ):
        if col in m:
            m[col] = pd.to_numeric(m[col], errors="coerce").fillna(0).astype("int8")
    m["pressure_allowed"] = (
        (m.get("pff_hitAllowed", 0) == 1)
        | (m.get("pff_hurryAllowed", 0) == 1)
        | (m.get("pff_sackAllowed", 0) == 1)
    ).astype("int8")
    return m


# ---------------------------------------------------------------------------
# Player profiles (exposure counts + outcome rates)
# ---------------------------------------------------------------------------
def build_player_profiles(
    pff_named: pd.DataFrame, players: pd.DataFrame, plays_enriched: pd.DataFrame
) -> pd.DataFrame:
    df = pff_named.copy()
    df["role_norm"] = df["pff_role"].map(C.normalize_role)

    def _agg(role_token: str, prefix: str) -> pd.DataFrame:
        sub = df[df["role_norm"] == role_token]
        if sub.empty:
            return pd.DataFrame(columns=["nflId"])
        if role_token == C.ROLE_PASS_BLOCK:
            g = sub.groupby("nflId").agg(
                **{
                    f"{prefix}_plays": ("playId", "count"),
                    f"{prefix}_hits_allowed": ("pff_hitAllowed", "sum"),
                    f"{prefix}_hurries_allowed": ("pff_hurryAllowed", "sum"),
                    f"{prefix}_sacks_allowed": ("pff_sackAllowed", "sum"),
                    f"{prefix}_beaten": ("pff_beatenByDefender", "sum"),
                }
            )
        else:  # pass_rush
            sub = sub.copy()
            for c in ("pff_hit", "pff_hurry", "pff_sack"):
                sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0)
            g = sub.groupby("nflId").agg(
                **{
                    f"{prefix}_plays": ("playId", "count"),
                    f"{prefix}_hits": ("pff_hit", "sum"),
                    f"{prefix}_hurries": ("pff_hurry", "sum"),
                    f"{prefix}_sacks": ("pff_sack", "sum"),
                }
            )
        return g.reset_index()

    block = _agg(C.ROLE_PASS_BLOCK, "block")
    rush = _agg(C.ROLE_PASS_RUSH, "rush")

    base = players[["nflId", "displayName", "officialPosition", "collegeName"]].copy()
    prof = base.merge(block, on="nflId", how="left").merge(rush, on="nflId", how="left")

    # Total observed exposure (proxy for workload — label as proxy in UI).
    prof["block_plays"] = prof.get("block_plays", 0)
    prof["rush_plays"] = prof.get("rush_plays", 0)
    prof[["block_plays", "rush_plays"]] = prof[["block_plays", "rush_plays"]].fillna(0)
    prof["observed_pass_pro_plays"] = (
        prof["block_plays"] + prof["rush_plays"]
    ).astype("int32")
    prof = prof[prof["observed_pass_pro_plays"] > 0].reset_index(drop=True)
    return prof


# ---------------------------------------------------------------------------
# Team pressure patterns
# ---------------------------------------------------------------------------
def build_team_patterns(plays_enriched: pd.DataFrame) -> pd.DataFrame:
    df = plays_enriched.copy()
    off = df.groupby("possessionTeam").agg(
        off_plays=("playId", "count"),
        off_pressure_rate=("pressure_event", "mean"),
        off_sack_rate=("sack_event", "mean"),
    )
    deff = df.groupby("defensiveTeam").agg(
        def_plays=("playId", "count"),
        def_pressure_rate=("pressure_event", "mean"),
        def_sack_rate=("sack_event", "mean"),
    )
    out = off.join(deff, how="outer").reset_index(names="team")
    return out


# ---------------------------------------------------------------------------
# Leakage-guarded pre-snap model frame
# ---------------------------------------------------------------------------
def build_model_frame(plays_enriched: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with pre-snap features + target, asserting no leakage."""
    needed = set(C.PRESNAP_FEATURES) | {"pressure_event", "gameId", "playId"}
    frame = plays_enriched[[c for c in plays_enriched.columns if c in needed]].copy()

    forbidden = (C.POST_SNAP_PLAY_COLS | C.POST_SNAP_PFF_COLS) & set(C.PRESNAP_FEATURES)
    if forbidden:
        raise ValueError(f"Leakage: post-snap columns in features: {sorted(forbidden)}")

    frame = frame.dropna(subset=["pressure_event"])
    frame = frame[frame["down"].notna()]  # require a real down
    return frame.reset_index(drop=True)
