"""PerformanceContextAdapter + preparation brief builder (Agent 4).

This turns tactical analysis into a performance-staff decision-support brief,
strictly gated behind human review. It demonstrates how play intelligence COULD
feed a preparation workflow without replacing a coach, nutritionist, or clinician.

Hard rule: the NFL scouting dataset has no physiological, injury, or medical
information. We never infer those. Workload here = observed play involvement,
explicitly labelled a proxy.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Adapter contract (newly written; a future authorised integration target)
# ---------------------------------------------------------------------------
@dataclass
class ReadinessContext:
    sleep_hours: float | None = None
    self_reported_fatigue: int | None = None  # 1-10 if provided, else None
    hydration_checkin: str | None = None

    @property
    def is_fictional(self) -> bool:
        """All readiness context is user-supplied/fictional, never dataset-derived."""
        return True


@dataclass
class UpcomingSession:
    type: str = "practice"
    duration_minutes: int | None = None


@dataclass
class PerformanceContext:
    athlete_id: str
    position: str
    upcoming_session: UpcomingSession = field(default_factory=UpcomingSession)
    readiness_context: ReadinessContext = field(default_factory=ReadinessContext)
    observed_play_involvement: int = 0
    preparation_focus: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "athlete_id": self.athlete_id,
            "position": self.position,
            "upcoming_session": asdict(self.upcoming_session),
            "readiness_context": asdict(self.readiness_context),
            "tactical_context": {
                "observed_play_involvement": self.observed_play_involvement,
                "preparation_focus": self.preparation_focus,
            },
            "_meta": {
                "readiness_is_fictional": True,
                "source": "PocketShift PerformanceContextAdapter (prototype)",
            },
        }


class PerformanceContextAdapter:
    """Builds a PerformanceContext from dataset-derived facts + optional input."""

    POSITION_DEMANDS = {
        "T": ["edge pass-set footwork", "anchor vs. power rush", "hand reset timing"],
        "G": ["interior anchor", "combo-to-second-level timing", "stunt pass-off"],
        "C": ["protection ID & calls", "snap-to-set quickness", "stunt communication"],
        "TE": ["chip-and-release timing", "edge help technique", "leverage on bigger ends"],
        "RB": ["blitz pickup ID", "cut vs. square protection", "scan progression"],
    }

    def get_position_demands(self, position: str) -> list[str]:
        key = (position or "").strip().upper()
        return self.POSITION_DEMANDS.get(key, ["general pass-protection technique"])

    def build_context(
        self,
        *,
        athlete_id: str,
        position: str,
        observed_play_involvement: int,
        preparation_focus: list[str] | None = None,
        session_type: str = "practice",
        session_minutes: int | None = None,
        sleep_hours: float | None = None,
        fatigue: int | None = None,
        hydration: str | None = None,
    ) -> PerformanceContext:
        return PerformanceContext(
            athlete_id=athlete_id,
            position=position,
            upcoming_session=UpcomingSession(session_type, session_minutes),
            readiness_context=ReadinessContext(sleep_hours, fatigue, hydration),
            observed_play_involvement=observed_play_involvement,
            preparation_focus=preparation_focus or self.get_position_demands(position),
        )


# ---------------------------------------------------------------------------
# Preparation brief
# ---------------------------------------------------------------------------
@dataclass
class PreparationBrief:
    athlete_label: str
    position: str
    game_finding: str
    technical_priority: list[str]
    workload_questions: list[str]
    preparation_checklist: list[str]
    recovery_questions: list[str]
    evidence_limitations: list[str]
    human_review_required: bool
    sources: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# Preparation Brief — {self.athlete_label} ({self.position})",
            "",
            "> PROTOTYPE decision-support bridge. Requires coaching / practitioner "
            "approval before any use. Not medical advice; no injury or physiological "
            "prediction is made.",
            "",
            "## Game finding",
            self.game_finding,
            "",
            "## Technical priority",
        ]
        lines += [f"- {x}" for x in self.technical_priority]
        lines += ["", "## Workload questions for the coach (workload = observed-play-involvement proxy)"]
        lines += [f"- {x}" for x in self.workload_questions]
        lines += ["", "## Preparation & hydration checklist"]
        lines += [f"- [ ] {x}" for x in self.preparation_checklist]
        lines += ["", "## Recovery questions for performance staff"]
        lines += [f"- {x}" for x in self.recovery_questions]
        lines += ["", "## Evidence limitations"]
        lines += [f"- {x}" for x in self.evidence_limitations]
        lines += ["", "## Human review"]
        lines += ["- [ ] Reviewed and approved by a coach / qualified practitioner before use."]
        if self.sources:
            lines += ["", "## Sources"]
            lines += [f"- Game {s['game_id']} · Play {s['play_id']} ({s['source_table']})"
                      for s in self.sources]
        return "\n".join(lines)


def build_preparation_brief(
    *,
    context: PerformanceContext,
    player_name: str,
    matchup_note: str,
    sources: list[dict] | None = None,
) -> PreparationBrief:
    pos = context.position
    demands = context.preparation_focus
    involvement = context.observed_play_involvement

    workload_questions = [
        f"Observed pass-pro involvement in this dataset: {involvement} plays "
        "(a proxy for exposure, NOT a measured training load).",
        "Does the planned session volume align with the week's actual rep count?",
        "Is the rep distribution weighted toward the technical priority above?",
    ]
    checklist = [
        f"Film review of the flagged matchup and {min(3, 3)} comparable situations.",
        f"Technical drill block targeting: {demands[0] if demands else 'pass-pro technique'}.",
        "Pre-session hydration check-in (self-reported; practitioner-reviewable).",
    ]
    if context.readiness_context.sleep_hours is not None:
        checklist.append(
            f"User-supplied sleep: {context.readiness_context.sleep_hours}h "
            "(fictional demo input; discuss with staff, do not treat as clinical)."
        )
    recovery_questions = [
        "How did the athlete self-report recovery after the last comparable workload?",
        "Any schedule constraints affecting the next preparation window?",
    ]
    limitations = [
        "Workload is observed play involvement only — a proxy, not a measured load.",
        "No physiological, injury, fatigue or medical data exists in the NFL scouting "
        "dataset; none is inferred here.",
        "Readiness/sleep/hydration fields are fictional demo inputs for prototype flow.",
        "Blocking choices are not randomised; comparisons are observational, not causal.",
    ]
    return PreparationBrief(
        athlete_label=player_name,
        position=pos,
        game_finding=matchup_note,
        technical_priority=demands,
        workload_questions=workload_questions,
        preparation_checklist=checklist,
        recovery_questions=recovery_questions,
        evidence_limitations=limitations,
        human_review_required=True,
        sources=sources or [],
    )
