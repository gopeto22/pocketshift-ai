"""Tracking-derived energy-expenditure & fatigue metric (side-quest).

Implements the NFL practitioner's recommendation: track energy expenditure and
fatigue (loss of high-speed acceleration/deceleration over the course of a game)
from frame-by-frame player tracking, gated at the snap via the `event` column.

Metric definitions (all computed from real tracking columns s, a, dis):

- Snap-gated: for each (game, play, player) we take frames FROM ball_snap onward
  (falling back to all frames if snap is unlabelled), so we measure effort during
  the live play, not pre-snap milling.

- Energy Expenditure Index (EEI): a metabolic-power-style proxy per play =
  sum over post-snap frames of (dis_i)  +  w_a * sum(|a_i| * dis_i) * dt.
  It rewards distance covered and penalises explosive accel/decel, the two drivers
  of metabolic cost in intermittent sport. Reported in arbitrary "effort units",
  explicitly a proxy — NOT calories and NOT a physiological measurement.

- High-speed efforts: count of post-snap frames with s >= HSR_THRESHOLD (yd/s).
- Peak accel / decel: max positive a and max |negative-trend a| in the play.

- Fatigue signal (per player, within a game, ordered by play sequence):
  rolling cumulative EEI (the "load" curve) and the trend in peak high-speed
  acceleration across the game. A decline in late-game peak acceleration while
  cumulative load is high is the fatigue flag the practitioner described.

Everything is a workload/effort PROXY from kinematics. No calorie, injury, or
physiological claim is made.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

# 10 Hz tracking → 0.1s per frame.
DT = 0.1
# Acceleration weighting in the EEI proxy (dimensionless).
ACCEL_WEIGHT = 0.17
# High-speed-running threshold (yards/second). ~15 mph ≈ 6.7 yd/s is a common
# HSR cut; we use 6.0 yd/s to capture meaningful bursts in short pass plays.
HSR_THRESHOLD = 6.0
# Acceleration burst threshold (yd/s^2).
ACCEL_BURST = 3.5

FATIGUE_PARQUET = C.PROCESSED_DIR / "player_fatigue.parquet"
PLAY_EFFORT_PARQUET = C.PROCESSED_DIR / "play_effort.parquet"

_SNAP_EVENTS = {"ball_snap", "autoevent_ballsnap"}


def _tracking_files() -> list[str]:
    return sorted(glob.glob(str(C.TRACKING_DIR / "tracking_*.csv")))


def tracking_available() -> bool:
    return len(_tracking_files()) > 0


def _post_snap(df_play: pd.DataFrame) -> pd.DataFrame:
    """Return frames at/after the snap; fall back to all frames if unlabelled."""
    snap = df_play[df_play["event"].isin(_SNAP_EVENTS)]
    if snap.empty:
        return df_play
    snap_frame = int(snap["frameId"].min())
    return df_play[df_play["frameId"] >= snap_frame]


def compute_play_effort(df_game: pd.DataFrame) -> pd.DataFrame:
    """Per (gameId, playId, nflId) effort metrics for one game's tracking frames."""
    df = df_game[df_game["nflId"].notna()].copy()
    df["nflId"] = df["nflId"].astype("int64")
    for col in ("s", "a", "dis"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    rows: list[dict] = []
    for (gid, pid), g in df.groupby(["gameId", "playId"], sort=False):
        g = _post_snap(g.sort_values(["nflId", "frameId"]))
        for nid, pg in g.groupby("nflId", sort=False):
            dist = float(pg["dis"].sum())
            accel_load = float((pg["a"].abs() * pg["dis"]).sum()) * DT
            eei = dist + ACCEL_WEIGHT * accel_load
            rows.append(
                {
                    "gameId": int(gid),
                    "playId": int(pid),
                    "nflId": int(nid),
                    "frames": int(len(pg)),
                    "distance_yd": dist,
                    "eei": eei,
                    "hsr_frames": int((pg["s"] >= HSR_THRESHOLD).sum()),
                    "peak_speed": float(pg["s"].max()) if len(pg) else 0.0,
                    "peak_accel": float(pg["a"].max()) if len(pg) else 0.0,
                    "accel_bursts": int((pg["a"] >= ACCEL_BURST).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_effort_tables(
    max_games: int | None = None, progress=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process tracking files → (play_effort, player_fatigue). Cached to Parquet.

    max_games bounds the work for a fast demo build. The featured game is always
    included first so the demo pathway is complete.
    """
    files = _tracking_files()
    if not files:
        raise FileNotFoundError("No tracking files found under data/tracking/.")

    featured = str(C.TRACKING_DIR / f"tracking_{C.FEATURED_GAME_ID}.csv")
    ordered = ([featured] if featured in files else []) + [f for f in files if f != featured]
    if max_games is not None:
        ordered = ordered[:max_games]

    parts: list[pd.DataFrame] = []
    for i, f in enumerate(ordered):
        if progress:
            progress(i + 1, len(ordered))
        dfg = pd.read_csv(
            f,
            na_values=["NA"],
            usecols=["gameId", "playId", "nflId", "frameId", "s", "a", "dis", "event"],
        )
        parts.append(compute_play_effort(dfg))
    play_effort = pd.concat(parts, ignore_index=True)

    fatigue = build_fatigue_timeline(play_effort)
    return play_effort, fatigue


def build_fatigue_timeline(play_effort: pd.DataFrame) -> pd.DataFrame:
    """Per (gameId, nflId): cumulative load + late-game acceleration decline.

    Fatigue flag = the player's peak-acceleration trend declines over the game
    while cumulative EEI (load) is high.
    """
    pe = play_effort.sort_values(["gameId", "nflId", "playId"]).copy()
    pe["cum_eei"] = pe.groupby(["gameId", "nflId"])["eei"].cumsum()
    pe["play_order"] = pe.groupby(["gameId", "nflId"]).cumcount() + 1

    out: list[dict] = []
    for (gid, nid), g in pe.groupby(["gameId", "nflId"], sort=False):
        n = len(g)
        if n < 6:
            continue
        total_load = float(g["eei"].sum())
        # Split game into first vs. last third of this player's snaps.
        k = max(2, n // 3)
        first = g.head(k)
        last = g.tail(k)
        early_peak_accel = float(first["peak_accel"].mean())
        late_peak_accel = float(last["peak_accel"].mean())
        early_hsr = float(first["hsr_frames"].mean())
        late_hsr = float(last["hsr_frames"].mean())
        accel_decline_pct = (
            (early_peak_accel - late_peak_accel) / early_peak_accel * 100.0
            if early_peak_accel > 0
            else 0.0
        )
        out.append(
            {
                "gameId": int(gid),
                "nflId": int(nid),
                "snaps": n,
                "total_eei": total_load,
                "mean_eei": float(g["eei"].mean()),
                "total_distance_yd": float(g["distance_yd"].sum()),
                "early_peak_accel": early_peak_accel,
                "late_peak_accel": late_peak_accel,
                "accel_decline_pct": accel_decline_pct,
                "early_hsr_per_play": early_hsr,
                "late_hsr_per_play": late_hsr,
                # Fatigue flag: meaningful late-game accel drop under real load.
                "fatigue_flag": int(accel_decline_pct >= 10.0 and n >= 15),
            }
        )
    return pd.DataFrame(out)


def effort_tables_exist() -> bool:
    return PLAY_EFFORT_PARQUET.exists() and FATIGUE_PARQUET.exists()


def write_effort_tables(play_effort: pd.DataFrame, fatigue: pd.DataFrame) -> None:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    play_effort.to_parquet(PLAY_EFFORT_PARQUET, index=False)
    fatigue.to_parquet(FATIGUE_PARQUET, index=False)


def load_effort_tables(
    max_games: int | None = None, rebuild: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rebuild or not effort_tables_exist():
        pe, fat = build_effort_tables(max_games=max_games)
        write_effort_tables(pe, fat)
        return pe, fat
    return pd.read_parquet(PLAY_EFFORT_PARQUET), pd.read_parquet(FATIGUE_PARQUET)


# ---------------------------------------------------------------------------
# Player-game fatigue curve (for charting a single player in a game)
# ---------------------------------------------------------------------------
@dataclass
class FatigueCurve:
    game_id: int
    nfl_id: int
    play_order: list[int]
    cum_eei: list[float]
    peak_accel: list[float]
    peak_speed: list[float]
    hsr_frames: list[int]


def player_game_curve(play_effort: pd.DataFrame, game_id: int, nfl_id: int) -> FatigueCurve:
    g = play_effort[
        (play_effort["gameId"] == int(game_id)) & (play_effort["nflId"] == int(nfl_id))
    ].sort_values("playId")
    order = list(range(1, len(g) + 1))
    return FatigueCurve(
        game_id=int(game_id),
        nfl_id=int(nfl_id),
        play_order=order,
        cum_eei=g["eei"].cumsum().tolist(),
        peak_accel=g["peak_accel"].tolist(),
        peak_speed=g["peak_speed"].tolist(),
        hsr_frames=g["hsr_frames"].tolist(),
    )
