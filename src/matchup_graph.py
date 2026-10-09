"""Blocker -> rusher interaction network for a single play.

This is an interaction graph derived from PFF assignments, NOT a spatial
reconstruction. Nodes are blockers and defenders; edges are recorded
blocking assignments with their charted outcomes.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

from . import config as C


def build_play_graph(
    pff_named_play: pd.DataFrame,
) -> nx.DiGraph:
    """Build a directed graph for one play's PFF rows (already name-joined).

    Expects columns from ingestion.attach_player_names filtered to one play.
    """
    g = nx.DiGraph()
    role = pff_named_play["pff_role"].map(C.normalize_role)

    # Offensive blockers.
    for _, r in pff_named_play[role == C.ROLE_PASS_BLOCK].iterrows():
        nid = _safe_int(r["nflId"])
        g.add_node(
            f"O{nid}",
            kind="blocker",
            side="offense",
            nflId=nid,
            name=_name(r.get("displayName"), nid),
            position=_s(r.get("pff_positionLinedUp")),
            block_type=_s(r.get("pff_blockType")),
            hit_allowed=_zero(r.get("pff_hitAllowed")),
            hurry_allowed=_zero(r.get("pff_hurryAllowed")),
            sack_allowed=_zero(r.get("pff_sackAllowed")),
            beaten=_zero(r.get("pff_beatenByDefender")),
        )

    # Defensive rushers.
    for _, r in pff_named_play[role == C.ROLE_PASS_RUSH].iterrows():
        nid = _safe_int(r["nflId"])
        g.add_node(
            f"D{nid}",
            kind="rusher",
            side="defense",
            nflId=nid,
            name=_name(r.get("displayName"), nid),
            position=_s(r.get("pff_positionLinedUp")),
            hit=_zero(r.get("pff_hit")),
            hurry=_zero(r.get("pff_hurry")),
            sack=_zero(r.get("pff_sack")),
        )

    # Edges: blocker -> blocked defender.
    for _, r in pff_named_play[role == C.ROLE_PASS_BLOCK].iterrows():
        blocked = r.get("pff_nflIdBlockedPlayer")
        if pd.isna(blocked):
            continue
        o = f"O{_safe_int(r['nflId'])}"
        d = f"D{_safe_int(blocked)}"
        if d not in g:  # blocked a player not charted as a rusher
            g.add_node(
                d,
                kind="rusher",
                side="defense",
                nflId=_safe_int(blocked),
                name=_name(r.get("blockedPlayerName"), _safe_int(blocked)),
                position=_s(r.get("blockedPlayerPosition")),
                hit=0,
                hurry=0,
                sack=0,
            )
        allowed = (
            _zero(r.get("pff_hitAllowed"))
            or _zero(r.get("pff_hurryAllowed"))
            or _zero(r.get("pff_sackAllowed"))
        )
        g.add_edge(
            o,
            d,
            block_type=_s(r.get("pff_blockType")),
            pressure_allowed=int(bool(allowed)),
            sack_allowed=_zero(r.get("pff_sackAllowed")),
        )
    return g


def graph_summary(g: nx.DiGraph) -> dict:
    blockers = [n for n, d in g.nodes(data=True) if d.get("kind") == "blocker"]
    rushers = [n for n, d in g.nodes(data=True) if d.get("kind") == "rusher"]
    breakdown_edges = [
        (u, v) for u, v, d in g.edges(data=True) if d.get("pressure_allowed") == 1
    ]
    return {
        "n_blockers": len(blockers),
        "n_rushers": len(rushers),
        "n_assignments": g.number_of_edges(),
        "n_breakdowns": len(breakdown_edges),
    }


def _safe_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def _zero(v) -> int:
    try:
        if pd.isna(v):
            return 0
        return int(v)
    except (TypeError, ValueError):
        return 0


def _s(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v)


def _name(v, nid: int) -> str:
    s = _s(v)
    return s if s else f"#{nid}"
