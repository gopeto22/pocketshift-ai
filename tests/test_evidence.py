"""Evidence reviewer: valid sources, no overclaims, no fabricated tools."""
from __future__ import annotations

from src import config as C
from src.evidence import Evidence, EvidenceReviewer, Source


def _reviewer(tables):
    return EvidenceReviewer(tables.plays_enriched, {"lookup_play", "get_pressure_events"})


def test_valid_source_recognised(tables):
    rev = _reviewer(tables)
    assert rev.valid_source(C.FEATURED_GAME_ID, C.FEATURED_PLAY_ID)
    assert not rev.valid_source(123, 456)


def test_invalid_source_blocks_answer(tables):
    rev = _reviewer(tables)
    ev = Evidence(question="q", sources=[Source(123, 456)])
    res = rev.review(answer_text="ok", evidence=ev, tool_trace=None)
    assert not res.ok


def test_affirmative_medical_claim_blocked(tables):
    rev = _reviewer(tables)
    res = rev.review(
        answer_text="He will suffer an injury; recommend medical treatment.",
        evidence=None,
        tool_trace=None,
    )
    assert not res.ok


def test_negated_disclaimer_allowed(tables):
    rev = _reviewer(tables)
    res = rev.review(
        answer_text="No medical or injury inference is made from this data.",
        evidence=None,
        tool_trace=None,
    )
    assert res.ok


def test_fabricated_tool_blocked(tables):
    rev = _reviewer(tables)
    trace = [{"tool": "lookup_play", "executed": True},
             {"tool": "make_up_numbers", "executed": True}]
    res = rev.review(answer_text="ok", evidence=None, tool_trace=trace)
    assert not res.ok


def test_unexecuted_tool_blocked(tables):
    rev = _reviewer(tables)
    trace = [{"tool": "lookup_play", "executed": False}]
    res = rev.review(answer_text="ok", evidence=None, tool_trace=trace)
    assert not res.ok


def test_valid_sources_and_tools_pass(tables):
    rev = _reviewer(tables)
    ev = Evidence(
        question="q",
        historical_comparisons=["matched rate 40% (n=120)"],
        sources=[Source(C.FEATURED_GAME_ID, C.FEATURED_PLAY_ID)],
    )
    trace = [{"tool": "lookup_play", "executed": True}]
    res = rev.review(answer_text="Grounded answer.", evidence=ev, tool_trace=trace,
                     empirical_claimed=True)
    assert res.ok
