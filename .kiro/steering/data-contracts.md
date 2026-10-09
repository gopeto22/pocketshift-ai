# Data Contracts — PocketShift AI

## Source files (data/)
- `games.csv` — 122 games (header + 122 rows). Key: `gameId`.
- `plays.csv` — 8,557 plays. Keys: `gameId`, `playId`.
- `players.csv` — 1,679 players. Key: `nflId`.
- `pffScoutingData.csv` — 188,254 scouting rows. Keys: `gameId`, `playId`, `nflId`.
- `tracking/tracking_[gameId].csv` — present (122 files). OPTIONAL. 10 Hz frames
  with s/a/dis/x/y/o/dir and an `event` column (ball_snap, pass_forward, …). Used
  by the Energy & Fatigue feature only; the core app never requires it. Energy =
  kinematic proxy (effort units), never calories or a physiological measurement.

Row counts are measured, not assumed. Re-profile if the workspace data differs.

## Join rules (NEVER join on playId alone)
- `games.gameId` → `plays.gameId`
- `plays.(gameId, playId)` → `pffScoutingData.(gameId, playId)`
- `pffScoutingData.nflId` → `players.nflId`
- `pffScoutingData.pff_nflIdBlockedPlayer` → `players.nflId` (when populated)

## Role labels (observed casing — title case with a space)
`Pass Rush`, `Pass Block`, `Pass Route`, `Coverage`, `Pass`.
Normalise to lowercase snake tokens internally: pass_rush, pass_block,
pass_route, coverage, pass. Always compare normalised.

## Missing values
Documented `NA` strings must be read as nulls. Blocking outcome fields
(pff_hitAllowed, etc.) are NA for non-blockers; rush fields (pff_hit, etc.) are
NA for non-rushers. Never treat NA as 0 unless the role makes 0 meaningful.

## Pre-snap vs post-snap (critical)
Post-snap / outcome fields MUST NOT be used as pre-snap model predictors:
- plays: passResult, playResult, prePenaltyPlayResult, penalty fields
- pff: pff_hit, pff_hurry, pff_sack, pff_*Allowed, pff_beatenByDefender,
  pff_blockType, pff_nflIdBlockedPlayer, pff_backFieldBlock
- pff_passCoverage / pff_passCoverageType are PFF-charted; treat as retrospective
  (oracle) features only, never as known-pre-snap unless explicitly labelled.

## Pressure-event label (play level)
A play is a pressure event when ANY `Pass Rush` row has pff_hit OR pff_hurry OR
pff_sack == 1. This is a scouting-derived proxy for recorded pressure
involvement, NOT an official Next Gen Stats pressure probability.
