# Data setup — PocketShift AI

PocketShift uses the **NFL Big Data Bowl 2023** dataset (2021 season, Weeks 1-8),
originally provided by the NFL Next Gen Stats team and Pro Football Focus (PFF).

**The raw dataset is NOT redistributed in this repository** (it is gitignored to
respect the competition's licensing/redistribution conditions). Download it from
the official competition source and place the files here.

## Required files (place in this `data/` directory)

```
data/
  games.csv              # 122 games
  plays.csv              # 8,557 plays
  players.csv            # 1,679 players
  pffScoutingData.csv    # 188,254 scouting rows
  tracking/              # OPTIONAL: tracking_<gameId>.csv (122 files)
```

The four CSVs are required. The `tracking/` files are optional — PocketShift's
primary workflow does not use player x/y coordinates and never fabricates them.

## Download

1. Get the dataset from the official NFL Big Data Bowl 2023 competition page
   (Kaggle). You must accept the competition rules to download.
2. Unzip the CSVs into this `data/` directory, matching the layout above.

## Build the processed tables

From the project root:

```bash
python profile_data.py
```

This validates the joins, writes `outputs/data_profile.md`, and builds:

```
data/processed/plays_enriched.parquet
data/processed/matchups.parquet
data/processed/player_profiles.parquet
data/processed/team_patterns.parquet
```

The app also builds these automatically on first launch if they are missing.

## Key join variables

- `games.gameId` → `plays.gameId`
- `plays.(gameId, playId)` → `pffScoutingData.(gameId, playId)`
- `pffScoutingData.nflId` → `players.nflId`
- `pffScoutingData.pff_nflIdBlockedPlayer` → `players.nflId` (when populated)

Never join on `playId` alone — it is not unique across games.
