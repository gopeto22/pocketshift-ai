"""Evidence objects and the deterministic quality reviewer (Agent 5).

The reviewer is authoritative: an LLM answer can never override a deterministic
validation failure. It checks that cited sources exist, that no medical/injury
claims appear, that empirical claims carry a sample size, and that every tool
call in a trace actually ran.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import config as C


@dataclass
class Source:
    game_id: int
    play_id: int
    source_table: str = "pffScoutingData"

    def to_dict(self) -> dict:
        return {
            "game_id": str(self.game_id),
            "play_id": str(self.play_id),
            "source_table": self.source_table,
        }

    def label(self) -> str:
        return f"Game {self.game_id} · Play {self.play_id}"


@dataclass
class Evidence:
    question: str
    observed_facts: list[str] = field(default_factory=list)
    historical_comparisons: list[str] = field(default_factory=list)
    model_estimates: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    recommendation: str = ""
    sources: list[Source] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "observed_facts": self.observed_facts,
            "historical_comparisons": self.historical_comparisons,
            "model_estimates": self.model_estimates,
            "limitations": self.limitations,
            "recommendation": self.recommendation,
            "sources": [s.to_dict() for s in self.sources],
        }


# ---------------------------------------------------------------------------
# Deterministic reviewer
# ---------------------------------------------------------------------------
# Terms that would overstate the product (injury/medical/physiological claims).
# These are flagged only as AFFIRMATIVE claims — a negated/disclaimer mention
# ("no medical inference is made") is allowed and even encouraged.
_FORBIDDEN_TERMS = [
    r"injur(?:y|ies|ed)",
    r"will get hurt",
    r"diagnos(?:e|is|ed)",
    r"prescrib(?:e|ed|ing)",
    r"medical(?:ly)?",
    r"fatigue level",
    r"physiological(?:ly)?",
    r"proven cause",
    r"causes the sack",
    r"official nfl endorsement",
]
# Negation words that neutralise a forbidden term when they appear just before it.
_NEGATORS = r"(?:no|not|never|without|isn't|aren't|don't|doesn't|cannot|can't|won't|free from|rather than)"

_CAUSAL_OVERCLAIM = [
    r"\bwould have prevented\b",
    r"\bwould have stopped\b",
    r"\bdefinitely\b",
    r"\bguarantee(?:d|s)?\b",
]


def _affirmative_hit(text: str, term: str) -> bool:
    """True if `term` appears NOT immediately preceded (within ~5 words) by a negator."""
    for m in re.finditer(r"\b" + term + r"\b", text):
        window = text[max(0, m.start() - 40):m.start()]
        if re.search(_NEGATORS + r"[\w\s,'-]{0,40}$", window):
            continue  # negated / disclaimer context
        return True
    return False


@dataclass
class ReviewResult:
    ok: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class EvidenceReviewer:
    def __init__(self, plays_enriched: pd.DataFrame, known_tools: set[str]):
        self._keys = set(
            map(tuple, plays_enriched[["gameId", "playId"]].to_numpy().tolist())
        )
        self._known_tools = set(known_tools)

    def valid_source(self, game_id: int, play_id: int) -> bool:
        return (int(game_id), int(play_id)) in self._keys

    def review(
        self,
        *,
        answer_text: str,
        evidence: Evidence | None,
        tool_trace: list[dict] | None,
        empirical_claimed: bool = False,
    ) -> ReviewResult:
        failures: list[str] = []
        warnings: list[str] = []
        text = (answer_text or "").lower()

        # 1. No affirmative medical / injury / physiological claims.
        for term in _FORBIDDEN_TERMS:
            if _affirmative_hit(text, term):
                failures.append(f"affirmative forbidden claim: /{term}/")
        for pat in _CAUSAL_OVERCLAIM:
            if re.search(pat, text):
                warnings.append(f"causal overclaim matched /{pat}/")

        # 2. Cited sources must exist.
        if evidence is not None:
            for s in evidence.sources:
                if not self.valid_source(s.game_id, s.play_id):
                    failures.append(
                        f"invalid source Game {s.game_id} Play {s.play_id}"
                    )

        # 3. Tool trace entries must map to real tools and claim execution.
        if tool_trace:
            for entry in tool_trace:
                name = entry.get("tool")
                if name not in self._known_tools:
                    failures.append(f"unknown tool in trace: {name}")
                if not entry.get("executed", False):
                    failures.append(f"tool not actually executed: {name}")

        # 4. Empirical claims must carry a sample size somewhere.
        if empirical_claimed and evidence is not None:
            joined = " ".join(evidence.historical_comparisons)
            if "n=" not in joined and "n =" not in joined:
                warnings.append("empirical claim without visible sample size")

        return ReviewResult(ok=len(failures) == 0, failures=failures, warnings=warnings)
