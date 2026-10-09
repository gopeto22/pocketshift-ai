"""Shared fixtures. Processed tables are built once per test session."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure Bedrock stays off during tests (deterministic paths only).
os.environ.setdefault("POCKETSHIFT_DISABLE_BEDROCK", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import ingestion  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return ingestion.load_raw()


@pytest.fixture(scope="session")
def tables():
    return ingestion.load_processed(rebuild=True)


@pytest.fixture(scope="session")
def pff_named(raw):
    return ingestion.attach_player_names(raw.pff, raw.players)
