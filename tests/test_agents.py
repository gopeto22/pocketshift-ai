"""Agent orchestration: real tool execution, graceful degradation, prep brief."""
from __future__ import annotations

from src import agents, config as C, evidence, retrieval
from src.performance_adapter import PerformanceContextAdapter, build_preparation_brief


def _orch(tables, pff_named):
    idx = retrieval.RetrievalIndex(tables.plays_enriched)
    reg = agents.ToolRegistry(tables, pff_named, idx)
    rev = evidence.EvidenceReviewer(tables.plays_enriched, reg.tool_names)
    return agents.Orchestrator(reg, rev), reg


def test_tools_execute_real_functions(tables, pff_named):
    _, reg = _orch(tables, pff_named)
    play = reg.execute("lookup_play", {"game_id": C.FEATURED_GAME_ID, "play_id": C.FEATURED_PLAY_ID})
    assert play["found"] is True
    assert play["sack_event"] == 1
    graph = reg.execute("get_pass_protection_graph",
                        {"game_id": C.FEATURED_GAME_ID, "play_id": C.FEATURED_PLAY_ID})
    assert graph["summary"]["n_assignments"] > 0


def test_pressure_events_tool_lists_barrett(tables, pff_named):
    _, reg = _orch(tables, pff_named)
    ev = reg.execute("get_pressure_events",
                     {"game_id": C.FEATURED_GAME_ID, "play_id": C.FEATURED_PLAY_ID})
    names = {e["defender"] for e in ev["pressure_events"]}
    assert any("Barrett" in n for n in names)


def test_graceful_degradation_without_bedrock(tables, pff_named):
    orch, _ = _orch(tables, pff_named)
    # Bedrock disabled via env in conftest.
    assert orch.ai_status()["available"] is False
    resp = orch.ask("Why did this protection fail?",
                    featured_hint=(C.FEATURED_GAME_ID, C.FEATURED_PLAY_ID))
    assert resp.used_bedrock is False
    assert resp.ai_available is False
    # Deterministic path still runs real tools and passes review.
    assert len(resp.tool_trace) > 0
    assert all(t["executed"] for t in resp.tool_trace)
    assert resp.review["ok"] is True
    assert any(s.game_id == C.FEATURED_GAME_ID for s in resp.evidence.sources)


def test_evidence_sources_are_valid(tables, pff_named):
    orch, reg = _orch(tables, pff_named)
    resp = orch.ask("What should we investigate next?",
                    featured_hint=(C.FEATURED_GAME_ID, C.FEATURED_PLAY_ID))
    rev = evidence.EvidenceReviewer(tables.plays_enriched, reg.tool_names)
    for s in resp.evidence.sources:
        assert rev.valid_source(s.game_id, s.play_id)


def test_preparation_brief_is_fictional_and_gated(tables, pff_named):
    adapter = PerformanceContextAdapter()
    ctx = adapter.build_context(
        athlete_id="demo-001", position="T", observed_play_involvement=42,
        sleep_hours=7.5,
    )
    assert ctx.readiness_context.is_fictional is True
    brief = build_preparation_brief(
        context=ctx, player_name="Demo Player",
        matchup_note="Charted on a demo play.",
        sources=[{"game_id": C.FEATURED_GAME_ID, "play_id": C.FEATURED_PLAY_ID,
                  "source_table": "pffScoutingData"}],
    )
    assert brief.human_review_required is True
    md = brief.to_markdown()
    assert "Requires coaching / practitioner approval" in md
    assert "proxy" in md.lower()
    # No injury/medical prediction language in the brief body.
    assert "will get injured" not in md.lower()


def test_live_bedrock_tool_loop_when_available(tables, pff_named):
    """If Bedrock is reachable, the live tool loop runs real tools and passes review.

    Skipped automatically when no model is accessible (CI / offline), so the suite
    stays green without AWS. Requires POCKETSHIFT_ENABLE_LIVE_BEDROCK=1 to opt in,
    since conftest disables Bedrock by default.
    """
    import os

    import pytest

    if os.environ.get("POCKETSHIFT_ENABLE_LIVE_BEDROCK") != "1":
        pytest.skip("live Bedrock test not opted in")
    os.environ["POCKETSHIFT_DISABLE_BEDROCK"] = "0"

    from src import agents, config as _C, evidence, retrieval
    from src.bedrock_client import BedrockClient

    _C.BEDROCK_DISABLED = False  # conftest froze this True at import

    if not BedrockClient().status().available:
        pytest.skip("no accessible Bedrock model")

    idx = retrieval.RetrievalIndex(tables.plays_enriched)
    reg = agents.ToolRegistry(tables, pff_named, idx)
    rev = evidence.EvidenceReviewer(tables.plays_enriched, reg.tool_names)
    orch = agents.Orchestrator(reg, rev)
    resp = orch.ask("Why did this protection fail?",
                    featured_hint=(C.FEATURED_GAME_ID, C.FEATURED_PLAY_ID))
    assert resp.used_bedrock is True
    assert resp.model_id
    assert len(resp.tool_trace) >= 1
    assert all(t["executed"] for t in resp.tool_trace)
    assert resp.review["ok"] is True


def test_no_sensitive_data_in_export(tables, pff_named):
    adapter = PerformanceContextAdapter()
    ctx = adapter.build_context(athlete_id="demo-001", position="G",
                                observed_play_involvement=10)
    payload = ctx.to_dict()
    # Readiness context is explicitly fictional / null unless user supplies it.
    assert payload["_meta"]["readiness_is_fictional"] is True
    assert payload["readiness_context"]["sleep_hours"] is None
