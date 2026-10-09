"""End-to-end smoke test of the full featured-demo pathway."""
from __future__ import annotations

from src import agents, config as C, evidence, modelling, retrieval, scenarios as S
from src.matchup_graph import build_play_graph, graph_summary


def test_full_featured_demo_pathway(tables, pff_named):
    pe = tables.plays_enriched

    # 1. Featured play present & correct.
    fp = pe[(pe.gameId == C.FEATURED_GAME_ID) & (pe.playId == C.FEATURED_PLAY_ID)]
    assert len(fp) == 1
    fp = fp.iloc[0]
    assert fp.sack_event == 1

    # 2. Forensics graph builds with a real breakdown structure.
    sub = pff_named[(pff_named.gameId == C.FEATURED_GAME_ID) & (pff_named.playId == C.FEATURED_PLAY_ID)]
    g = build_play_graph(sub)
    gs = graph_summary(g)
    assert gs["n_assignments"] > 0
    assert gs["n_blockers"] > 0

    # 3. Counterfactual produces a supported, uncertainty-qualified result.
    res = S.compare_protection_configurations(
        pe, "extra_protection", match_filters=S.context_from_play(fp)
    )
    assert res.supported
    assert res.diff_ci[0] is not None

    # 4. Agent answers with real tools and passes review.
    idx = retrieval.RetrievalIndex(pe)
    reg = agents.ToolRegistry(tables, pff_named, idx)
    rev = evidence.EvidenceReviewer(pe, reg.tool_names)
    orch = agents.Orchestrator(reg, rev)
    resp = orch.ask("Investigate this sack and a protection alternative.",
                    featured_hint=(C.FEATURED_GAME_ID, C.FEATURED_PLAY_ID))
    assert resp.review["ok"]
    assert len(resp.tool_trace) >= 3

    # 5. Model trains and reports honest metrics.
    rep = modelling.train_pressure_model(pe, kind="logreg", n_splits=5)
    assert 0.0 <= rep.roc_auc <= 1.0
    assert rep.n_total > 1000
    assert rep.brier <= rep.baseline_brier + 0.02  # at least not far worse than base rate
