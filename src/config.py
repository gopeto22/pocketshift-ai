"""Central configuration: paths, constants, thresholds, evidence typing.

Pure constants and small helpers. No heavy imports so every other module and
test can import this cheaply.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRACKING_DIR = DATA_DIR / "tracking"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"

GAMES_CSV = DATA_DIR / "games.csv"
PLAYS_CSV = DATA_DIR / "plays.csv"
PLAYERS_CSV = DATA_DIR / "players.csv"
PFF_CSV = DATA_DIR / "pffScoutingData.csv"

RAW_CSVS = {
    "games": GAMES_CSV,
    "plays": PLAYS_CSV,
    "players": PLAYERS_CSV,
    "pff": PFF_CSV,
}

PLAYS_ENRICHED_PARQUET = PROCESSED_DIR / "plays_enriched.parquet"
MATCHUPS_PARQUET = PROCESSED_DIR / "matchups.parquet"
PLAYER_PROFILES_PARQUET = PROCESSED_DIR / "player_profiles.parquet"
TEAM_PATTERNS_PARQUET = PROCESSED_DIR / "team_patterns.parquet"

# ---------------------------------------------------------------------------
# Featured demo case (verified present in the dataset)
# ---------------------------------------------------------------------------
FEATURED_GAME_ID = 2021090900
FEATURED_PLAY_ID = 2279
FEATURED_SUMMARY = "Dak Prescott sacked by Shaquil Barrett (DAL vs TB)"

# ---------------------------------------------------------------------------
# Role normalisation (dataset uses title-case with a space)
# ---------------------------------------------------------------------------
ROLE_PASS_RUSH = "pass_rush"
ROLE_PASS_BLOCK = "pass_block"
ROLE_PASS_ROUTE = "pass_route"
ROLE_COVERAGE = "coverage"
ROLE_PASS = "pass"


def normalize_role(raw: str | None) -> str | None:
    """Map any observed PFF role label to a normalised snake token."""
    if raw is None:
        return None
    key = str(raw).strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    return {
        "pass rush": ROLE_PASS_RUSH,
        "pass block": ROLE_PASS_BLOCK,
        "pass route": ROLE_PASS_ROUTE,
        "coverage": ROLE_COVERAGE,
        "pass": ROLE_PASS,
    }.get(key, key.replace(" ", "_"))


# ---------------------------------------------------------------------------
# Leakage control: columns that describe during/after-snap outcomes.
# These MUST NOT be used as pre-snap model predictors.
# ---------------------------------------------------------------------------
POST_SNAP_PLAY_COLS = frozenset(
    {
        "passResult",
        "playResult",
        "prePenaltyPlayResult",
        "penaltyYards",
        "foulName1",
        "foulName2",
        "foulName3",
        "foulNFLId1",
        "foulNFLId2",
        "foulNFLId3",
    }
)
POST_SNAP_PFF_COLS = frozenset(
    {
        "pff_hit",
        "pff_hurry",
        "pff_sack",
        "pff_beatenByDefender",
        "pff_hitAllowed",
        "pff_hurryAllowed",
        "pff_sackAllowed",
        "pff_blockType",
        "pff_nflIdBlockedPlayer",
        "pff_backFieldBlock",
    }
)
# PFF-charted coverage is retrospective; usable only as an oracle feature.
RETROSPECTIVE_PLAY_COLS = frozenset({"pff_passCoverage", "pff_passCoverageType"})

# Pre-snap features the pressure model is allowed to use.
PRESNAP_FEATURES = [
    "down",
    "yardsToGo",
    "quarter",
    "defendersInBox",
    "absoluteYardlineNumber",
    "pff_playAction",
    "score_diff",
    "offenseFormation",
    "personnelO",
    "personnelD",
]

# ---------------------------------------------------------------------------
# Evidence typing
# ---------------------------------------------------------------------------
EV_OBSERVED = "observed"
EV_HISTORICAL = "historical"
EV_MODEL = "model"
EV_HYPOTHETICAL = "hypothetical"
EVIDENCE_TYPES = frozenset({EV_OBSERVED, EV_HISTORICAL, EV_MODEL, EV_HYPOTHETICAL})

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Minimum matched plays per arm before an empirical rate may be reported.
MIN_SUPPORT = 30
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 7
EXTRA_PROTECTION_BLOCKER_THRESHOLD = 6  # >= this many blockers = "extra help"

# ---------------------------------------------------------------------------
# Bedrock
# ---------------------------------------------------------------------------
# Region: prefer explicit env, else the account's default (eu-central-1 here).
AWS_REGION = (
    os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "eu-central-1"
)
BEDROCK_MODEL_ID = os.environ.get("POCKETSHIFT_BEDROCK_MODEL_ID", "").strip()
BEDROCK_DISABLED = os.environ.get("POCKETSHIFT_DISABLE_BEDROCK", "0").strip() == "1"
# Candidate models to auto-try (first that answers wins). Verified ACTIVE in the
# hackathon eu-central-1 account. INFERENCE_PROFILE models are addressed with a
# region-prefixed inference-profile id ("eu."); ON_DEMAND models by bare id.
BEDROCK_MODEL_CANDIDATES = [
    # Anthropic Claude via EU inference profiles (strong tool use).
    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "eu.anthropic.claude-opus-4-5-20251101-v1:0",
    "eu.anthropic.claude-haiku-5-5",
    "eu.anthropic.claude-sonnet-4-6",
    # Bare ids (in case profiles are addressed directly in-region).
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-opus-4-5-20251101-v1:0",
    # ON_DEMAND fallbacks that need no inference profile.
    "openai.gpt-oss-120b-1:0",
    "mistral.devstral-2-123b",
]

# ---------------------------------------------------------------------------
# UI theme
# ---------------------------------------------------------------------------
THEME = {
    "bg": "#0b1220",
    "panel": "#111a2b",
    "panel2": "#15213a",
    "text": "#e6edf6",
    "muted": "#8aa0b6",
    "accent": "#22d3ee",   # cyan
    "accent2": "#34d399",  # green
    "caution": "#f59e0b",  # amber
    "breakdown": "#ef4444",  # red
    "grid": "#1e2b45",
}
