"""Agent orchestration: a real tool registry over the analytical modules, with
Bedrock tool-use when available and a deterministic fallback otherwise.

Every tool is a real function that touches the data. Tool traces reflect actual
execution. The deterministic reviewer (evidence.EvidenceReviewer) can veto any
answer; the LLM never overrides it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from . import config as C
from . import scenarios as S
from .bedrock_client import BedrockClient
from .evidence import Evidence, EvidenceReviewer, Source
from .ingestion import ProcessedTables
from .matchup_graph import build_play_graph, graph_summary
from .performance_adapter import PerformanceContextAdapter, build_preparation_brief
from .retrieval import RetrievalIndex


# ---------------------------------------------------------------------------
# Tool registry — each entry wraps a real analytical function.
# ---------------------------------------------------------------------------
class ToolRegistry:
    def __init__(
        self,
        tables: ProcessedTables,
        pff_named: pd.DataFrame,
        retrieval: RetrievalIndex,
    ):
        self.tables = tables
        self.plays = tables.plays_enriched
        self.matchups = tables.matchups
        self.profiles = tables.player_profiles
        self.pff_named = pff_named
        self.retrieval = retrieval
        self._adapter = PerformanceContextAdapter()
        self._tools: dict[str, Callable[..., dict]] = {
            "lookup_play": self.lookup_play,
            "get_play_participants": self.get_play_participants,
            "get_pass_protection_graph": self.get_pass_protection_graph,
            "get_player_history": self.get_player_history,
            "get_pressure_events": self.get_pressure_events,
            "find_similar_plays": self.find_similar_plays,
            "calculate_empirical_pressure_rate": self.calculate_empirical_pressure_rate,
            "compare_protection_configurations": self.compare_protection_configurations,
            "estimate_historical_alternative": self.estimate_historical_alternative,
            "get_position_demands": self.get_position_demands,
            "summarize_observed_exposure": self.summarize_observed_exposure,
            "get_player_fatigue": self.get_player_fatigue,
            "get_play_energy": self.get_play_energy,
        }
        # Optional tracking-derived effort/fatigue tables (loaded lazily).
        self._play_effort = None
        self._fatigue = None

    @property
    def tool_names(self) -> set[str]:
        return set(self._tools)

    def execute(self, name: str, args: dict) -> dict:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        result = self._tools[name](**args)
        return _jsonable(result)

    # --- Agent 1: Play Forensics tools ---
    def lookup_play(self, game_id: int, play_id: int) -> dict:
        row = self._play_row(game_id, play_id)
        if row is None:
            return {"found": False, "game_id": game_id, "play_id": play_id}
        return {
            "found": True,
            "game_id": int(game_id),
            "play_id": int(play_id),
            "description": _s(row.get("playDescription")),
            "quarter": _i(row.get("quarter")),
            "down": _i(row.get("down")),
            "yardsToGo": _i(row.get("yardsToGo")),
            "possessionTeam": _s(row.get("possessionTeam")),
            "defensiveTeam": _s(row.get("defensiveTeam")),
            "offenseFormation": _s(row.get("offenseFormation")),
            "personnelO": _s(row.get("personnelO")),
            "personnelD": _s(row.get("personnelD")),
            "defendersInBox": _i(row.get("defendersInBox")),
            "pff_passCoverage": _s(row.get("pff_passCoverage")),
            "passResult": _s(row.get("passResult")),
            "pressure_event": _i(row.get("pressure_event")),
            "sack_event": _i(row.get("sack_event")),
            "n_rushers": _i(row.get("n_rushers")),
            "n_blockers": _i(row.get("n_blockers")),
        }

    def get_play_participants(self, game_id: int, play_id: int) -> dict:
        sub = self._pff_play(game_id, play_id)
        parts = []
        for _, r in sub.iterrows():
            parts.append(
                {
                    "nflId": _i(r.get("nflId")),
                    "name": _s(r.get("displayName")),
                    "role": C.normalize_role(r.get("pff_role")),
                    "linedUp": _s(r.get("pff_positionLinedUp")),
                }
            )
        return {"game_id": int(game_id), "play_id": int(play_id), "participants": parts}

    def get_pass_protection_graph(self, game_id: int, play_id: int) -> dict:
        sub = self._pff_play(game_id, play_id)
        g = build_play_graph(sub)
        edges = []
        for u, v, d in g.edges(data=True):
            edges.append(
                {
                    "blocker": g.nodes[u].get("name"),
                    "blocker_pos": g.nodes[u].get("position"),
                    "defender": g.nodes[v].get("name"),
                    "block_type": d.get("block_type"),
                    "pressure_allowed": d.get("pressure_allowed"),
                    "sack_allowed": d.get("sack_allowed"),
                }
            )
        return {
            "game_id": int(game_id),
            "play_id": int(play_id),
            "summary": graph_summary(g),
            "assignments": edges,
        }

    def get_pressure_events(self, game_id: int, play_id: int) -> dict:
        sub = self._pff_play(game_id, play_id)
        role = sub["pff_role"].map(C.normalize_role)
        rush = sub[role == C.ROLE_PASS_RUSH]
        events = []
        for _, r in rush.iterrows():
            if _i(r.get("pff_hit")) or _i(r.get("pff_hurry")) or _i(r.get("pff_sack")):
                events.append(
                    {
                        "defender": _s(r.get("displayName")),
                        "position": _s(r.get("pff_positionLinedUp")),
                        "hit": _i(r.get("pff_hit")),
                        "hurry": _i(r.get("pff_hurry")),
                        "sack": _i(r.get("pff_sack")),
                    }
                )
        return {"game_id": int(game_id), "play_id": int(play_id), "pressure_events": events}

    def get_player_history(self, nfl_id: int) -> dict:
        prof = self.profiles[self.profiles["nflId"] == int(nfl_id)]
        if prof.empty:
            return {"nfl_id": int(nfl_id), "found": False}
        r = prof.iloc[0]
        return {
            "nfl_id": int(nfl_id),
            "found": True,
            "name": _s(r.get("displayName")),
            "position": _s(r.get("officialPosition")),
            "observed_pass_pro_plays": _i(r.get("observed_pass_pro_plays")),
            "block_plays": _i(r.get("block_plays")),
            "rush_plays": _i(r.get("rush_plays")),
            "hits_allowed": _i(r.get("block_hits_allowed")),
            "hurries_allowed": _i(r.get("block_hurries_allowed")),
            "sacks_allowed": _i(r.get("block_sacks_allowed")),
            "rush_sacks": _i(r.get("rush_sacks")),
            "rush_hurries": _i(r.get("rush_hurries")),
        }

    # --- Agent 2: Historical Comparison tools ---
    def find_similar_plays(
        self,
        down_bucket: str | None = None,
        distance_bucket: str | None = None,
        offense_formation: str | None = None,
        limit: int = 10,
    ) -> dict:
        res = S.find_similar_plays(
            self.plays,
            down_bucket=down_bucket,
            distance_bucket=distance_bucket,
            offense_formation=offense_formation,
            limit=limit,
        )
        return {
            "n_matched": int(len(res)),
            "sample": [
                {"game_id": int(r.gameId), "play_id": int(r.playId),
                 "pressure_event": int(r.pressure_event)}
                for r in res.head(limit).itertuples()
            ],
        }

    def calculate_empirical_pressure_rate(
        self,
        down_bucket: str | None = None,
        distance_bucket: str | None = None,
        offense_formation: str | None = None,
        outcome: str = "pressure_event",
    ) -> dict:
        pool = S.find_similar_plays(
            self.plays,
            down_bucket=down_bucket,
            distance_bucket=distance_bucket,
            offense_formation=offense_formation,
        )
        return S.empirical_pressure_rate(pool, outcome)

    def compare_protection_configurations(
        self,
        scenario: str,
        down_bucket: str | None = None,
        distance_bucket: str | None = None,
        offense_formation: str | None = None,
        outcome: str = "pressure_event",
    ) -> dict:
        filters = {
            k: v
            for k, v in {
                "down_bucket": down_bucket,
                "distance_bucket": distance_bucket,
                "offense_formation": offense_formation,
            }.items()
            if v
        }
        res = S.compare_protection_configurations(
            self.plays, scenario, outcome=outcome, match_filters=filters
        )
        return res.to_dict()

    # --- Agent 3: Counterfactual Investigator tool ---
    def estimate_historical_alternative(
        self,
        game_id: int,
        play_id: int,
        scenario: str,
        outcome: str = "pressure_event",
    ) -> dict:
        row = self._play_row(game_id, play_id)
        if row is None:
            return {"error": "play not found"}
        filters = S.context_from_play(row)
        res = S.compare_protection_configurations(
            self.plays, scenario, outcome=outcome, match_filters=filters
        )
        d = res.to_dict()
        d["observed_outcome"] = _i(row.get(outcome))
        return d

    # --- Agent 4: Performance Preparation tools ---
    def get_position_demands(self, position: str) -> dict:
        return {"position": position, "demands": self._adapter.get_position_demands(position)}

    def summarize_observed_exposure(self, nfl_id: int) -> dict:
        return self.get_player_history(nfl_id)

    # --- Tracking-derived energy & fatigue tools (side-quest) ---
    def _ensure_tracking(self) -> bool:
        if self._fatigue is not None:
            return not self._fatigue.empty
        try:
            from . import tracking as TR

            if not TR.effort_tables_exist():
                self._fatigue = pd.DataFrame()
                return False
            self._play_effort, self._fatigue = TR.load_effort_tables()
            return True
        except Exception:  # noqa: BLE001
            self._fatigue = pd.DataFrame()
            return False

    def get_player_fatigue(self, nfl_id: int, game_id: int | None = None) -> dict:
        """Tracking-derived energy expenditure and late-game acceleration decline."""
        if not self._ensure_tracking():
            return {"available": False,
                    "note": "Tracking effort tables not built; run profile_data.py --tracking."}
        f = self._fatigue[self._fatigue["nflId"] == int(nfl_id)]
        if game_id is not None:
            f = f[f["gameId"] == int(game_id)]
        if f.empty:
            return {"available": True, "found": False, "nfl_id": int(nfl_id)}
        rows = []
        for _, r in f.iterrows():
            rows.append(
                {
                    "game_id": int(r["gameId"]),
                    "snaps": int(r["snaps"]),
                    "total_energy_index": round(float(r["total_eei"]), 1),
                    "total_distance_yd": round(float(r["total_distance_yd"]), 1),
                    "early_peak_accel": round(float(r["early_peak_accel"]), 2),
                    "late_peak_accel": round(float(r["late_peak_accel"]), 2),
                    "accel_decline_pct": round(float(r["accel_decline_pct"]), 1),
                    "fatigue_flag": int(r["fatigue_flag"]),
                }
            )
        return {"available": True, "found": True, "nfl_id": int(nfl_id), "games": rows}

    def get_play_energy(self, game_id: int, play_id: int) -> dict:
        """Per-player post-snap energy expenditure for one play."""
        if not self._ensure_tracking():
            return {"available": False}
        pe = self._play_effort[
            (self._play_effort["gameId"] == int(game_id))
            & (self._play_effort["playId"] == int(play_id))
        ]
        if pe.empty:
            return {"available": True, "found": False}
        names = (
            self.pff_named[["nflId", "displayName", "officialPosition"]]
            .dropna(subset=["nflId"])
            .drop_duplicates("nflId")
        )
        pe = pe.merge(names, on="nflId", how="left").sort_values("eei", ascending=False)
        players = [
            {
                "name": _s(r.displayName),
                "position": _s(r.officialPosition),
                "energy_index": round(float(r.eei), 1),
                "distance_yd": round(float(r.distance_yd), 1),
                "peak_speed": round(float(r.peak_speed), 1),
                "hsr_frames": int(r.hsr_frames),
            }
            for r in pe.head(12).itertuples()
        ]
        return {"available": True, "found": True, "game_id": int(game_id),
                "play_id": int(play_id), "players": players}

    # --- helpers ---
    def _play_row(self, game_id: int, play_id: int):
        sub = self.plays[
            (self.plays["gameId"] == int(game_id))
            & (self.plays["playId"] == int(play_id))
        ]
        return None if sub.empty else sub.iloc[0]

    def _pff_play(self, game_id: int, play_id: int) -> pd.DataFrame:
        return self.pff_named[
            (self.pff_named["gameId"] == int(game_id))
            & (self.pff_named["playId"] == int(play_id))
        ]


# ---------------------------------------------------------------------------
# Bedrock tool specs
# ---------------------------------------------------------------------------
def bedrock_tool_specs() -> list[dict]:
    def spec(name, desc, props, required):
        return {
            "toolSpec": {
                "name": name,
                "description": desc,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    }
                },
            }
        }

    gp = {"game_id": {"type": "integer"}, "play_id": {"type": "integer"}}
    ctx = {
        "down_bucket": {"type": "string", "enum": ["1st", "2nd", "3rd", "4th"]},
        "distance_bucket": {
            "type": "string",
            "enum": ["short(1-3)", "medium(4-7)", "long(8+)"],
        },
        "offense_formation": {"type": "string"},
    }
    scen = {"scenario": {"type": "string", "enum": list(S.SCENARIOS.keys())}}
    return [
        spec("lookup_play", "Retrieve the observed facts of a specific play.", gp, ["game_id", "play_id"]),
        spec("get_play_participants", "List players and roles on a play.", gp, ["game_id", "play_id"]),
        spec("get_pass_protection_graph", "Reconstruct blocker->rusher assignments and breakdowns.", gp, ["game_id", "play_id"]),
        spec("get_pressure_events", "List defenders credited with hit/hurry/sack on a play.", gp, ["game_id", "play_id"]),
        spec("get_player_history", "Observed pass-pro exposure and outcomes for a player.", {"nfl_id": {"type": "integer"}}, ["nfl_id"]),
        spec("find_similar_plays", "Find real plays matching controllable context.", {**ctx, "limit": {"type": "integer"}}, []),
        spec("calculate_empirical_pressure_rate", "Empirical pressure/sack rate over matched plays with bootstrap CI and sample size.", {**ctx, "outcome": {"type": "string", "enum": ["pressure_event", "sack_event"]}}, []),
        spec("compare_protection_configurations", "Matched A/B comparison of a protection modification.", {**scen, **ctx, "outcome": {"type": "string"}}, ["scenario"]),
        spec("estimate_historical_alternative", "For a specific play, estimate a protection alternative from matched history.", {**gp, **scen, "outcome": {"type": "string"}}, ["game_id", "play_id", "scenario"]),
        spec("get_position_demands", "Position-specific preparation focus areas.", {"position": {"type": "string"}}, ["position"]),
        spec("summarize_observed_exposure", "Observed play-involvement exposure for a player (proxy).", {"nfl_id": {"type": "integer"}}, ["nfl_id"]),
        spec("get_player_fatigue", "Tracking-derived energy expenditure and late-game high-speed-acceleration decline (fatigue proxy) for a player.", {"nfl_id": {"type": "integer"}, "game_id": {"type": "integer"}}, ["nfl_id"]),
        spec("get_play_energy", "Per-player post-snap energy expenditure (effort proxy) from tracking for one play.", gp, ["game_id", "play_id"]),
    ]


SYSTEM_PROMPT = """You are PocketShift, an NFL pass-protection analytics agent.
You answer a coach grounded ONLY in data returned by your tools. Rules:
- Call tools to get facts; never invent numbers, players, or play identifiers.
- Distinguish observed facts, historical comparisons (always cite sample size n),
  model estimates, and purely hypothetical statements.
- Never claim an alternative WOULD have changed the outcome; historical blocking
  choices are not randomised, so comparisons are observational, not causal.
- Never make medical, injury, or physiological claims. Energy expenditure and
  fatigue are kinematic PROXIES from tracking (speed/accel/distance), never
  calories or a measured physiological state — say so when you use them.
- When you cite a play, use the exact "Game <id> · Play <id>" format.
Keep the final answer concise and coach-usable."""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@dataclass
class AgentResponse:
    text: str
    tool_trace: list[dict]
    used_bedrock: bool
    model_id: str | None
    evidence: Evidence
    review: dict
    ai_available: bool
    unavailable_reason: str = ""


class Orchestrator:
    def __init__(self, registry: ToolRegistry, reviewer: EvidenceReviewer):
        self.registry = registry
        self.reviewer = reviewer
        self.bedrock = BedrockClient()

    def ai_status(self) -> dict:
        st = self.bedrock.status()
        return {"available": st.available, "model_id": st.model_id, "reason": st.reason}

    def ask(self, question: str, *, featured_hint: tuple[int, int] | None = None) -> AgentResponse:
        st = self.bedrock.status()
        if st.available:
            try:
                turn = self.bedrock.converse_with_tools(
                    system=SYSTEM_PROMPT,
                    user_message=self._augment(question, featured_hint),
                    tool_specs=bedrock_tool_specs(),
                    executor=self.registry.execute,
                )
                evidence = self._evidence_from_trace(question, turn.tool_trace, turn.text)
                review = self.reviewer.review(
                    answer_text=turn.text,
                    evidence=evidence,
                    tool_trace=turn.tool_trace,
                    empirical_claimed=any(
                        t["tool"].startswith(("calculate_", "compare_", "estimate_"))
                        for t in turn.tool_trace
                    ),
                )
                text = turn.text
                if not review.ok:
                    text = (
                        "The AI draft failed deterministic evidence review and was "
                        "blocked. Showing a verified deterministic answer instead.\n\n"
                        + self._deterministic_answer(question, featured_hint)[0]
                    )
                return AgentResponse(
                    text=text,
                    tool_trace=turn.tool_trace,
                    used_bedrock=True,
                    model_id=turn.model_id,
                    evidence=evidence,
                    review=review.to_dict(),
                    ai_available=True,
                )
            except Exception as exc:  # noqa: BLE001 — fall back on any runtime error
                import os as _os

                if _os.environ.get("POCKETSHIFT_DEBUG_AGENT") == "1":
                    import traceback

                    traceback.print_exc()
                return self._fallback(question, featured_hint, reason=f"{type(exc).__name__}: {exc}")
        return self._fallback(question, featured_hint, reason=st.reason)

    # --- deterministic fallback ---
    def _fallback(self, question, featured_hint, reason: str) -> AgentResponse:
        text, trace, evidence = self._deterministic_answer(question, featured_hint, return_all=True)
        review = self.reviewer.review(
            answer_text=text, evidence=evidence, tool_trace=trace, empirical_claimed=True
        )
        return AgentResponse(
            text=text,
            tool_trace=trace,
            used_bedrock=False,
            model_id=None,
            evidence=evidence,
            review=review.to_dict(),
            ai_available=False,
            unavailable_reason=reason,
        )

    def _deterministic_answer(self, question, featured_hint, return_all: bool = False):
        """A grounded, tool-driven answer without an LLM. Runs real tools."""
        g, p = featured_hint or (C.FEATURED_GAME_ID, C.FEATURED_PLAY_ID)
        trace: list[dict] = []

        def run(tool, **kw):
            res = self.registry.execute(tool, kw)
            trace.append({"tool": tool, "args": kw, "executed": True,
                          "result_keys": list(res.keys())})
            return res

        play = run("lookup_play", game_id=g, play_id=p)
        graph = run("get_pass_protection_graph", game_id=g, play_id=p)
        pevents = run("get_pressure_events", game_id=g, play_id=p)
        alt = run("estimate_historical_alternative", game_id=g, play_id=p,
                  scenario="extra_protection")

        lines = []
        if play.get("found"):
            lines.append(
                f"Observed: {play['description']} "
                f"({play['possessionTeam']} vs {play['defensiveTeam']}, "
                f"Q{play['quarter']} {play['down']}&{play['yardsToGo']}). "
                f"Pressure event={play['pressure_event']}, sack={play['sack_event']}, "
                f"{play['n_rushers']} rushers vs {play['n_blockers']} blockers."
            )
        if pevents.get("pressure_events"):
            who = ", ".join(
                f"{e['defender']} ({'sack' if e['sack'] else 'hit' if e['hit'] else 'hurry'})"
                for e in pevents["pressure_events"]
            )
            lines.append(f"Charted pressure credited to: {who}.")
        br = graph["summary"]["n_breakdowns"]
        lines.append(
            f"Interaction graph: {graph['summary']['n_assignments']} assignments, "
            f"{br} charted breakdown(s)."
        )
        if alt.get("supported"):
            lines.append(alt["message"])
        else:
            lines.append(alt.get("message", "Alternative not estimable with current data."))
        lines.append(
            "Note: comparisons are observational, not causal. No medical or injury "
            "inference is made."
        )
        text = " ".join(lines)

        evidence = Evidence(
            question=question,
            observed_facts=[l for l in lines[:3]],
            historical_comparisons=[alt.get("message", "")] if alt.get("supported") else [],
            limitations=[
                "Observational comparison; blocking choices not randomised.",
                "Pressure label is a scouting-derived proxy.",
            ],
            recommendation=(
                "Review the flagged matchup on film and compare the matched "
                "historical subset before changing the protection call."
            ),
            sources=[Source(g, p, "pffScoutingData")],
        )
        if return_all:
            return text, trace, evidence
        return text, evidence

    def _augment(self, question: str, featured_hint) -> str:
        if featured_hint:
            g, p = featured_hint
            return (
                f"{question}\n\n(Context: the coach is currently looking at "
                f"Game {g} · Play {p}. Use the tools to ground your answer.)"
            )
        return question

    def _evidence_from_trace(self, question, trace, text) -> Evidence:
        sources: list[Source] = []
        for t in trace:
            a = t.get("args", {})
            if "game_id" in a and "play_id" in a:
                try:
                    sources.append(Source(int(a["game_id"]), int(a["play_id"])))
                except (TypeError, ValueError):
                    pass
        # Deduplicate.
        seen = set()
        uniq = []
        for s in sources:
            k = (s.game_id, s.play_id)
            if k not in seen:
                seen.add(k)
                uniq.append(s)
        return Evidence(
            question=question,
            observed_facts=[],
            recommendation="",
            sources=uniq,
        )


def _jsonable(obj):
    """Coerce numpy/pandas scalars and containers to plain JSON-safe Python.

    Bedrock's Converse toolResult requires native types; numpy int64/float64 and
    pandas NA break the serialization and silently drop the tool loop.
    """
    import numpy as _np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        f = float(obj)
        return None if (f != f) else f  # NaN -> None
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if obj is None:
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def _s(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v)


def _i(v) -> int:
    try:
        if pd.isna(v):
            return 0
        return int(v)
    except (TypeError, ValueError):
        return 0
