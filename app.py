"""PocketShift AI — Streamlit application.

From pressure to preparation. An AI coaching agent that investigates NFL pass
protection, explores tactical alternatives, and turns game evidence into a
coach-reviewable preparation brief.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config as C
from src import agents, evidence, ingestion, modelling, retrieval, scenarios as S
from src import visualizations as viz
from src.matchup_graph import build_play_graph, graph_summary
from src.performance_adapter import PerformanceContextAdapter, build_preparation_brief
from src.validation import dataset_summary, validate

st.set_page_config(
    page_title="PocketShift AI",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

T = C.THEME

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {T['bg']}; color: {T['text']}; }}
    section[data-testid="stSidebar"] {{ background-color: {T['panel']}; }}
    .ps-title {{ font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em;
        color: {T['text']}; margin-bottom: 0; }}
    .ps-accent {{ color: {T['accent']}; }}
    .ps-sub {{ color: {T['muted']}; font-size: 1.02rem; margin-top: .2rem; }}
    .ps-card {{ background: {T['panel']}; border: 1px solid {T['grid']};
        border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: .8rem; }}
    .ps-kpi {{ font-size: 1.8rem; font-weight: 700; color: {T['accent']}; }}
    .ps-kpi-label {{ color: {T['muted']}; font-size: .8rem; text-transform: uppercase;
        letter-spacing: .06em; }}
    .ps-tag {{ display:inline-block; padding:.12rem .5rem; border-radius:6px;
        font-size:.72rem; font-weight:700; letter-spacing:.04em; }}
    .ps-observed {{ background:#0e2a3a; color:{T['accent']}; }}
    .ps-historical {{ background:#0e3326; color:{T['accent2']}; }}
    .ps-model {{ background:#3a2f0e; color:{T['caution']}; }}
    .ps-hypo {{ background:#3a0e14; color:{T['breakdown']}; }}
    .ps-break {{ color:{T['breakdown']}; font-weight:700; }}
    .stButton>button {{ background:{T['accent']}; color:{T['bg']}; border:0;
        font-weight:700; border-radius:8px; }}
    .trace-ok {{ color:{T['accent2']}; font-family:monospace; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached loaders (expensive joins happen once)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading & joining NFL data…")
def load_everything():
    tables = ingestion.load_processed()
    pff_named = ingestion.attach_player_names(ingestion.load_pff(), ingestion.load_players())
    index = retrieval.RetrievalIndex(tables.plays_enriched)
    registry = agents.ToolRegistry(tables, pff_named, index)
    reviewer = evidence.EvidenceReviewer(tables.plays_enriched, registry.tool_names)
    orch = agents.Orchestrator(registry, reviewer)
    summary = dataset_summary()
    return tables, pff_named, index, orch, summary


@st.cache_data(show_spinner="Validating data…")
def cached_validation():
    return validate().to_frame()


@st.cache_data(show_spinner="Training pre-snap pressure model…")
def cached_model(kind: str):
    tables = ingestion.load_processed()
    return modelling.train_pressure_model(tables.plays_enriched, kind=kind).to_dict()


@st.cache_resource(show_spinner="Loading tracking energy/fatigue tables…")
def cached_effort():
    from src import tracking as TR

    if not TR.tracking_available():
        return None, None, False
    if not TR.effort_tables_exist():
        # Build a demo-sized subset (featured game always included) on first use.
        pe, fat = TR.build_effort_tables(max_games=12)
        TR.write_effort_tables(pe, fat)
    else:
        pe, fat = TR.load_effort_tables()
    return pe, fat, True


TABLES, PFF_NAMED, INDEX, ORCH, SUMMARY = load_everything()
PLAYS = TABLES.plays_enriched


def ev_tag(kind: str) -> str:
    cls = {
        C.EV_OBSERVED: "ps-observed",
        C.EV_HISTORICAL: "ps-historical",
        C.EV_MODEL: "ps-model",
        C.EV_HYPOTHETICAL: "ps-hypo",
    }[kind]
    return f'<span class="ps-tag {cls}">{kind.upper()}</span>'


def play_label(row) -> str:
    return (
        f"{int(row.gameId)} · {int(row.playId)} — "
        f"{row.possessionTeam} vs {row.defensiveTeam}: "
        f"{str(row.playDescription)[:70]}"
    )


# --------------------------------------------------------------------------
# Sidebar — navigation + AI status
# --------------------------------------------------------------------------
st.sidebar.markdown(
    f'<div class="ps-title">POCKET<span class="ps-accent">SHIFT</span></div>'
    f'<div class="ps-sub">From pressure to preparation.</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
screen = st.sidebar.radio(
    "Navigate",
    [
        "🏟  Command Centre",
        "🔎  Play Forensics",
        "🧪  Counterfactual Lab",
        "💬  Ask PocketShift",
        "⚡  Energy & Fatigue",
        "🧑‍⚕  Player Preparation",
    ],
    label_visibility="collapsed",
)

ai = ORCH.ai_status()
if ai["available"]:
    st.sidebar.success(f"AI agent online\n\n`{ai['model_id']}`")
else:
    st.sidebar.warning("AI agent unavailable\n\nDeterministic mode active.")
    with st.sidebar.expander("Why?"):
        st.caption(ai["reason"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "Prototype decision-support. Observational, not causal. No injury or medical "
    "prediction. Dataset: NFL Big Data Bowl 2023 (2021 season)."
)


# Session state for the currently selected play (shared across screens).
if "sel_game" not in st.session_state:
    st.session_state.sel_game = C.FEATURED_GAME_ID
    st.session_state.sel_play = C.FEATURED_PLAY_ID


def set_play(g: int, p: int) -> None:
    st.session_state.sel_game = int(g)
    st.session_state.sel_play = int(p)


def current_play_row():
    sub = PLAYS[
        (PLAYS.gameId == st.session_state.sel_game)
        & (PLAYS.playId == st.session_state.sel_play)
    ]
    return sub.iloc[0] if not sub.empty else None


# ==========================================================================
# SCREEN A — Command Centre
# ==========================================================================
def screen_command_centre():
    st.markdown('<div class="ps-title">Command Centre</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ps-sub">An AI coaching agent that investigates NFL pass '
        "protection, explores tactical alternatives and turns game evidence into "
        "actionable preparation.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    cols = st.columns(5)
    kpis = [
        ("Games", SUMMARY["n_games"]),
        ("Plays", f"{SUMMARY['n_plays']:,}"),
        ("Players", f"{SUMMARY['n_players']:,}"),
        ("PFF rows", f"{SUMMARY['n_pff_rows']:,}"),
        ("Pressure-event rate", f"{PLAYS['pressure_event'].mean():.1%}"),
    ]
    for col, (label, val) in zip(cols, kpis):
        col.markdown(
            f'<div class="ps-card"><div class="ps-kpi">{val}</div>'
            f'<div class="ps-kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Featured demonstration case")
        fp = PLAYS[(PLAYS.gameId == C.FEATURED_GAME_ID) & (PLAYS.playId == C.FEATURED_PLAY_ID)].iloc[0]
        st.markdown(
            f'<div class="ps-card">{ev_tag(C.EV_OBSERVED)} '
            f"<b>Game {C.FEATURED_GAME_ID} · Play {C.FEATURED_PLAY_ID}</b><br>"
            f"{fp.playDescription}<br><span style='color:{T['muted']}'>"
            f"{fp.possessionTeam} offense vs {fp.defensiveTeam} defense · "
            f"Q{int(fp.quarter)} {int(fp.down)} & {int(fp.yardsToGo)} · "
            f"{fp.offenseFormation}</span></div>",
            unsafe_allow_html=True,
        )
        if st.button("▶  Investigate this play", type="primary"):
            set_play(C.FEATURED_GAME_ID, C.FEATURED_PLAY_ID)
            st.success("Loaded. Open **Play Forensics** from the sidebar.")

        st.markdown("#### Filters")
        teams = sorted(set(PLAYS.possessionTeam.dropna()) | set(PLAYS.defensiveTeam.dropna()))
        f1, f2 = st.columns(2)
        off = f1.selectbox("Offense", ["(any)"] + teams, key="cc_off")
        deff = f2.selectbox("Defense", ["(any)"] + teams, key="cc_def")
        f3, f4 = st.columns(2)
        downs = ["(any)", "1st", "2nd", "3rd", "4th"]
        dsel = f3.selectbox("Down", downs, key="cc_down")
        forms = ["(any)"] + sorted(PLAYS.offenseFormation.dropna().unique().tolist())
        fsel = f4.selectbox("Formation", forms, key="cc_form")

        filt = PLAYS
        if off != "(any)":
            filt = filt[filt.possessionTeam == off]
        if deff != "(any)":
            filt = filt[filt.defensiveTeam == deff]
        if dsel != "(any)":
            filt = filt[filt.down_bucket == dsel]
        if fsel != "(any)":
            filt = filt[filt.offenseFormation == fsel]

        st.caption(
            f"{len(filt):,} plays match · pressure rate "
            f"{filt['pressure_event'].mean():.1%} · sack rate "
            f"{filt['sack_event'].mean():.1%}"
        )
        high = filt[filt.sack_event == 1].copy()
        high["pressure_score"] = high.hits + high.hurries + high.sacks
        high = high.sort_values("pressure_score", ascending=False).head(12)
        st.markdown("#### High-pressure plays (recorded sacks)")
        for r in high.itertuples():
            c1, c2 = st.columns([5, 1])
            c1.markdown(
                f"**{int(r.gameId)} · {int(r.playId)}** {r.possessionTeam} vs "
                f"{r.defensiveTeam} — {str(r.playDescription)[:66]}"
            )
            if c2.button("Open", key=f"open_{r.gameId}_{r.playId}"):
                set_play(r.gameId, r.playId)
                st.success(f"Loaded Game {int(r.gameId)} · Play {int(r.playId)}.")

    with right:
        st.markdown("#### Pressure-event rate by offensive formation")
        by_form = INDEX.pressure_rate_by("offenseFormation")
        st.plotly_chart(
            viz.pressure_rate_bar(by_form, "Offensive formation", ""),
            use_container_width=True,
        )
        st.markdown("#### Most common block types")
        bt = (
            PFF_NAMED["pff_blockType"].dropna().value_counts().head(8).rename_axis("block").reset_index(name="n")
        )
        st.dataframe(bt, use_container_width=True, hide_index=True)


# ==========================================================================
# SCREEN B — Play Forensics
# ==========================================================================
def screen_forensics():
    st.markdown('<div class="ps-title">Play Forensics</div>', unsafe_allow_html=True)
    _play_picker()
    row = current_play_row()
    if row is None:
        st.error("Selected play not found.")
        return
    g, p = int(row.gameId), int(row.playId)

    st.markdown(
        f'<div class="ps-card">{ev_tag(C.EV_OBSERVED)} <b>{row.playDescription}</b><br>'
        f"<span style='color:{T['muted']}'>Game {g} · Play {p}</span></div>",
        unsafe_allow_html=True,
    )

    c = st.columns(6)
    facts = [
        ("Quarter", f"Q{_safe(row.quarter)}"),
        ("Down & dist", f"{_safe(row.down)} & {_safe(row.yardsToGo)}"),
        ("Offense", str(row.possessionTeam)),
        ("Defense", str(row.defensiveTeam)),
        ("Formation", str(row.offenseFormation)),
        ("Coverage", str(row.pff_passCoverage)),
    ]
    for col, (k, v) in zip(c, facts):
        col.markdown(
            f'<div class="ps-card"><div class="ps-kpi-label">{k}</div>'
            f'<div style="font-weight:700">{v}</div></div>',
            unsafe_allow_html=True,
        )

    c2 = st.columns(5)
    outcome_facts = [
        ("Pass result", str(row.passResult)),
        ("Pressure event", "YES" if row.pressure_event else "no"),
        ("Sack", "YES" if row.sack_event else "no"),
        ("Rushers", int(row.n_rushers)),
        ("Blockers", int(row.n_blockers)),
    ]
    for col, (k, v) in zip(c2, outcome_facts):
        col.markdown(
            f'<div class="ps-card"><div class="ps-kpi-label">{k}</div>'
            f'<div style="font-weight:700">{v}</div></div>',
            unsafe_allow_html=True,
        )

    sub = PFF_NAMED[(PFF_NAMED.gameId == g) & (PFF_NAMED.playId == p)]
    graph = build_play_graph(sub)
    gsum = graph_summary(graph)

    gcol, icol = st.columns([1.4, 1])
    with gcol:
        st.plotly_chart(viz.matchup_graph_figure(graph), use_container_width=True)
        st.caption(
            f"{gsum['n_blockers']} blockers · {gsum['n_rushers']} rushers · "
            f"{gsum['n_assignments']} assignments · "
            f"**{gsum['n_breakdowns']} charted breakdown(s)** (red). "
            "Interaction graph from PFF assignments — not a spatial reconstruction."
        )
    with icol:
        st.markdown("#### Pass rushers")
        role = sub["pff_role"].map(C.normalize_role)
        rush = sub[role == C.ROLE_PASS_RUSH][
            ["displayName", "pff_positionLinedUp", "pff_hit", "pff_hurry", "pff_sack"]
        ].rename(columns={"displayName": "player", "pff_positionLinedUp": "pos"})
        st.dataframe(rush.fillna(0), use_container_width=True, hide_index=True)

        st.markdown("#### Pass blockers & assignments")
        blk = sub[role == C.ROLE_PASS_BLOCK][
            ["displayName", "pff_positionLinedUp", "pff_blockType",
             "blockedPlayerName", "pff_hitAllowed", "pff_hurryAllowed", "pff_sackAllowed"]
        ].rename(
            columns={
                "displayName": "blocker",
                "pff_positionLinedUp": "pos",
                "pff_blockType": "block",
                "blockedPlayerName": "assigned to",
            }
        )
        st.dataframe(blk.fillna(0), use_container_width=True, hide_index=True)

    st.info(
        "Jump to the **Counterfactual Lab** to explore protection alternatives, "
        "or **Player Preparation** to build a brief from this matchup."
    )


# ==========================================================================
# SCREEN C — Counterfactual Lab
# ==========================================================================
def screen_counterfactual():
    st.markdown('<div class="ps-title">Counterfactual Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ps-sub">Observed vs. a plausible alternative — grounded in '
        "matched historical evidence, with honest uncertainty.</div>",
        unsafe_allow_html=True,
    )
    _play_picker()
    row = current_play_row()
    if row is None:
        st.error("Selected play not found.")
        return

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Observed play")
        st.markdown(
            f'<div class="ps-card">{ev_tag(C.EV_OBSERVED)} {row.playDescription}<br>'
            f"<span style='color:{T['muted']}'>Pressure event: "
            f"<b>{'YES' if row.pressure_event else 'no'}</b> · Sack: "
            f"<b>{'YES' if row.sack_event else 'no'}</b></span></div>",
            unsafe_allow_html=True,
        )
        scenario = st.selectbox(
            "Protection-related alternative to investigate",
            options=list(S.SCENARIOS.keys()),
            format_func=lambda k: S.SCENARIOS[k]["label"],
        )
        outcome = st.radio("Outcome metric", ["pressure_event", "sack_event"], horizontal=True)
        st.markdown("#### Matching context")
        keys = []
        if st.checkbox("Match on down", value=True):
            keys.append("down_bucket")
        if st.checkbox("Match on distance", value=True):
            keys.append("distance_bucket")
        if st.checkbox("Match on formation", value=True):
            keys.append("offense_formation")
        run = st.button("Estimate from matched history", type="primary")

    with right:
        if run:
            filters = S.context_from_play(row, keys)
            res = S.compare_protection_configurations(
                PLAYS, scenario, outcome=outcome, match_filters=filters
            )
            st.markdown("#### Hypothetical alternative vs. baseline")
            st.caption(f"Matched on: {filters or '(no filters — whole dataset)'}")
            if not res.supported:
                st.markdown(
                    f'<div class="ps-card">{ev_tag(C.EV_HYPOTHETICAL)} '
                    f'<span class="ps-break">{res.message}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.plotly_chart(viz.scenario_compare_bar(res.to_dict()), use_container_width=True)
                st.markdown(
                    f'<div class="ps-card">{ev_tag(C.EV_HISTORICAL)} {res.message}</div>',
                    unsafe_allow_html=True,
                )
                a, b = res.arm_a, res.arm_b
                m1, m2, m3 = st.columns(3)
                m1.metric(res.arm_a_label, f"{a['rate']:.1%}", f"n={a['n']}")
                m2.metric(res.arm_b_label, f"{b['rate']:.1%}", f"n={b['n']}")
                m3.metric("Difference", f"{res.diff:+.1%}",
                          f"95% CI [{res.diff_ci[0]:.1%}, {res.diff_ci[1]:.1%}]")
                if res.diff is not None and res.diff > 0:
                    st.warning(
                        "The modified arm shows a HIGHER pressure rate. This is "
                        "likely selection bias: teams add protection precisely when "
                        "they already expect pressure. A difference here is not a "
                        "causal effect of the modification."
                    )
            with st.expander("Why this is not a causal claim"):
                st.write(
                    "Historical blocking choices are not randomised. We compare "
                    "real plays that happened to use each configuration in a matched "
                    "context. Confounders (opponent, game script, pre-snap read) are "
                    "not controlled, so treat the difference as an association with "
                    "uncertainty, not a guaranteed outcome change."
                )
        else:
            st.info("Choose a scenario and matching context, then estimate.")


# ==========================================================================
# SCREEN D — Ask PocketShift
# ==========================================================================
def screen_ask():
    st.markdown('<div class="ps-title">Ask PocketShift</div>', unsafe_allow_html=True)
    _play_picker()
    row = current_play_row()
    hint = (int(row.gameId), int(row.playId)) if row is not None else None

    examples = [
        "Why did this protection fail?",
        "Which defender created the most pressure?",
        "What happened historically when extra blocking help was used here?",
        "What should we investigate in our next film session?",
    ]
    st.caption("Example questions:")
    ecols = st.columns(len(examples))
    picked = None
    for col, ex in zip(ecols, examples):
        if col.button(ex, key=f"ex_{hash(ex)}"):
            picked = ex

    q = st.text_input("Your question", value=picked or "", placeholder="Ask the AI coach…")
    go = st.button("Ask", type="primary") or bool(picked)

    if go and q.strip():
        with st.spinner("Agent reasoning over retrieved evidence…"):
            resp = ORCH.ask(q, featured_hint=hint)

        mode = (
            f"🟢 Bedrock agent · `{resp.model_id}`"
            if resp.used_bedrock
            else "🟡 Deterministic mode (AI agent unavailable)"
        )
        st.markdown(f"**Mode:** {mode}")

        st.markdown("#### Execution trace (actual tool calls)")
        for t in resp.tool_trace:
            status = "✓" if t.get("executed") else "✗"
            st.markdown(
                f'<span class="trace-ok">{status} {t["tool"]}'
                f'({_fmt_args(t.get("args", {}))})</span>',
                unsafe_allow_html=True,
            )
        if not resp.tool_trace:
            st.caption("No tools were called.")

        st.markdown("#### Answer")
        st.markdown(f'<div class="ps-card">{resp.text}</div>', unsafe_allow_html=True)

        ev = resp.evidence
        st.markdown("#### Evidence panel")
        if ev.sources:
            st.markdown("**Sources**")
            for s in ev.sources:
                st.markdown(f"- `{s.label()}` ({s.source_table})")
        if ev.observed_facts:
            st.markdown("**Observed facts**")
            for f in ev.observed_facts:
                st.markdown(f"- {f}")
        if ev.historical_comparisons:
            st.markdown("**Historical comparisons**")
            for h in ev.historical_comparisons:
                st.markdown(f"- {h}")
        if ev.limitations:
            st.markdown("**Limitations**")
            for l in ev.limitations:
                st.markdown(f"- {l}")
        if ev.recommendation:
            st.markdown(f"**Recommendation:** {ev.recommendation}")

        rev = resp.review
        if rev["ok"]:
            st.success("Evidence reviewer: PASS (sources valid, no overclaims).")
        else:
            st.error(f"Evidence reviewer: BLOCKED — {rev['failures']}")
        if rev["warnings"]:
            st.caption(f"Reviewer warnings: {rev['warnings']}")


# ==========================================================================
# SCREEN E — Player Preparation
# ==========================================================================
def screen_preparation():
    st.markdown('<div class="ps-title">Player Preparation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ps-sub">A prototype bridge from tactical analysis to a '
        "coach-reviewable preparation brief. Not a medical or injury-prevention "
        "system.</div>",
        unsafe_allow_html=True,
    )
    _play_picker()
    row = current_play_row()
    if row is None:
        st.error("Selected play not found.")
        return
    g, p = int(row.gameId), int(row.playId)

    sub = PFF_NAMED[(PFF_NAMED.gameId == g) & (PFF_NAMED.playId == p)]
    role = sub["pff_role"].map(C.normalize_role)
    blockers = sub[role == C.ROLE_PASS_BLOCK]
    options = {
        f"{r.displayName} ({r.pff_positionLinedUp})": (int(r.nflId), str(r.displayName), str(r.officialPosition))
        for r in blockers.itertuples()
        if pd.notna(r.nflId)
    }
    if not options:
        st.warning("No charted blockers on this play to build a brief for.")
        return

    left, right = st.columns([1, 1.2])
    with left:
        choice = st.selectbox("Player (blocker on this play)", list(options.keys()))
        nfl_id, name, position = options[choice]
        hist = ORCH.registry.get_player_history(nfl_id)
        involvement = hist.get("observed_pass_pro_plays", 0)
        st.markdown(
            f'<div class="ps-card">{ev_tag(C.EV_OBSERVED)} <b>{name}</b> · {position}<br>'
            f"Observed pass-pro involvement: <b>{involvement}</b> plays "
            f"<span style='color:{T['muted']}'>(proxy for exposure, not a measured "
            f"load)</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### Optional fictional readiness context")
        st.caption("Demo-only inputs. Never dataset-derived; never clinical.")
        session_type = st.selectbox("Session type", ["practice", "walkthrough", "lift"])
        minutes = st.slider("Session minutes", 30, 150, 90, 15)
        sleep = st.number_input("Self-reported sleep (h)", 0.0, 12.0, 0.0, 0.5)
        fatigue = st.slider("Self-reported fatigue (1-10, 0=skip)", 0, 10, 0)
        hydration = st.selectbox("Hydration check-in", ["(skip)", "green", "amber", "red"])
        make = st.button("Create Preparation Brief", type="primary")

    with right:
        if make:
            adapter = PerformanceContextAdapter()
            # Matchup note from the actual play.
            brow = blockers[blockers.nflId == nfl_id].iloc[0]
            allowed = any(
                _safe(brow.get(c)) == 1
                for c in ("pff_hitAllowed", "pff_hurryAllowed", "pff_sackAllowed")
            )
            matchup_note = (
                f"On Game {g} · Play {p}, {name} ({brow.pff_positionLinedUp}) was "
                f"charted blocking {brow.blockedPlayerName or 'an assigned defender'} "
                f"with a {brow.pff_blockType} block. "
                + ("This blocker was charted as allowing pressure on the play."
                   if allowed else "No pressure was charted against this blocker on the play.")
            )
            ctx = adapter.build_context(
                athlete_id=f"demo-{nfl_id}",
                position=position,
                observed_play_involvement=involvement,
                session_type=session_type,
                session_minutes=minutes,
                sleep_hours=sleep or None,
                fatigue=fatigue or None,
                hydration=None if hydration == "(skip)" else hydration,
            )
            brief = build_preparation_brief(
                context=ctx,
                player_name=name,
                matchup_note=matchup_note,
                sources=[{"game_id": g, "play_id": p, "source_table": "pffScoutingData"}],
            )
            st.markdown("#### Preparation brief")
            st.markdown(f'<div class="ps-card">{ev_tag(C.EV_OBSERVED)} '
                        f"<b>Game finding</b><br>{brief.game_finding}</div>",
                        unsafe_allow_html=True)
            st.markdown("**Technical priority**")
            for x in brief.technical_priority:
                st.markdown(f"- {x}")
            st.markdown("**Workload questions** (workload = observed-involvement proxy)")
            for x in brief.workload_questions:
                st.markdown(f"- {x}")
            st.markdown("**Preparation & hydration checklist**")
            for x in brief.preparation_checklist:
                st.markdown(f"- ☐ {x}")
            st.markdown("**Recovery questions for performance staff**")
            for x in brief.recovery_questions:
                st.markdown(f"- {x}")
            st.error(
                "**Human review required** — a coach or qualified practitioner must "
                "approve this brief before any use. No injury, fatigue or medical "
                "state is inferred from the dataset."
            )
            with st.expander("Evidence limitations"):
                for x in brief.evidence_limitations:
                    st.markdown(f"- {x}")
            st.download_button(
                "⬇ Download brief (Markdown)",
                data=brief.to_markdown(),
                file_name=f"preparation_brief_{nfl_id}_{g}_{p}.md",
                mime="text/markdown",
            )
            with st.expander("PerformanceContextAdapter payload (NutriElite-inspired, newly written)"):
                st.json(ctx.to_dict())
        else:
            st.info("Pick a player and optional context, then create the brief.")


# ==========================================================================
# SCREEN F — Energy & Fatigue (tracking-derived; side-quest)
# ==========================================================================
def screen_energy_fatigue():
    from src import tracking as TR

    st.markdown('<div class="ps-title">Energy &amp; Fatigue</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ps-sub">Frame-by-frame tracking turned into an Energy '
        "Expenditure Index and a late-game acceleration-decline fatigue signal. "
        "Kinematic proxies — not calories, not a medical measurement.</div>",
        unsafe_allow_html=True,
    )
    pe, fat, ok = cached_effort()
    if not ok:
        st.warning(
            "Tracking files not found under `data/tracking/`. This screen is "
            "optional; the rest of PocketShift runs without it."
        )
        return

    st.caption(
        f"Energy/fatigue computed over {pe['gameId'].nunique()} game(s), "
        f"{pe['nflId'].nunique():,} players, {len(pe):,} player-plays. "
        "Snap-gated via the tracking `event` column."
    )

    tab1, tab2, tab3 = st.tabs(
        ["Player fatigue curve", "Energy on a play", "Fatigue ↔ outcomes"]
    )

    # --- Tab 1: player fatigue curve across a game ---
    with tab1:
        pl = PFF_NAMED[["nflId", "displayName", "officialPosition"]].drop_duplicates()
        games = sorted(pe["gameId"].unique().tolist())
        gi = st.selectbox("Game", games, key="fat_game")
        gfat = fat[fat.gameId == gi].merge(pl, on="nflId", how="left")
        gfat = gfat.sort_values("total_eei", ascending=False)
        gfat["label"] = gfat["displayName"].fillna("#" + gfat["nflId"].astype(str)) + \
            " (" + gfat["officialPosition"].fillna("?") + ")"
        if gfat.empty:
            st.info("No fatigue rows for this game.")
        else:
            lab = st.selectbox("Player (ranked by total energy)", gfat["label"].tolist())
            nid = int(gfat[gfat.label == lab]["nflId"].iloc[0])
            curve = TR.player_game_curve(pe, gi, nid)
            st.plotly_chart(viz.fatigue_curve_figure(curve), use_container_width=True)
            row = gfat[gfat.nflId == nid].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Snaps", int(row.snaps))
            c2.metric("Total energy index", f"{row.total_eei:,.0f}")
            c3.metric("Total distance", f"{row.total_distance_yd:,.0f} yd")
            c4.metric("Late-game accel decline", f"{row.accel_decline_pct:+.1f}%")
            if int(row.fatigue_flag) == 1:
                st.warning(
                    f"**Fatigue flag:** {lab}'s peak acceleration fell "
                    f"{row.accel_decline_pct:.1f}% from the first to the last third "
                    "of their snaps while carrying a high energy load. This is the "
                    "'loss of high-speed acceleration late in the game' pattern — a "
                    "kinematic proxy a performance coach would investigate for "
                    "refuelling/rotation, not a medical conclusion."
                )
            else:
                st.success(
                    "No late-game acceleration decline flag for this player in this "
                    "game (explosive output held up under load)."
                )

    # --- Tab 2: per-player energy on a single play ---
    with tab2:
        _play_picker()
        row = current_play_row()
        if row is not None:
            res = ORCH.registry.get_play_energy(int(row.gameId), int(row.playId))
            if res.get("available") and res.get("found"):
                st.plotly_chart(viz.play_energy_bar(res["players"]), use_container_width=True)
                st.caption(
                    "Post-snap distance and acceleration load per player. The ball "
                    "carrier, blockers who pulled, and rushers who looped typically "
                    "top this list — the 'assist' players the pro flagged, not just "
                    "the passer/receiver."
                )
            else:
                st.info("No tracking rows for this play in the built subset.")

    # --- Tab 3: fatigue/energy linked to outcomes ---
    with tab3:
        plays = PLAYS[["gameId", "playId", "pressure_event"]]
        m = pe.merge(plays, on=["gameId", "playId"], how="inner")
        agg = m.groupby(["gameId", "playId"]).agg(
            mean_eei=("eei", "mean"), pressure=("pressure_event", "first")
        ).reset_index()
        agg = agg.dropna(subset=["mean_eei"])
        try:
            agg["load_tertile"] = pd.qcut(agg.mean_eei, 3, labels=["low", "mid", "high"])
            tert = agg.groupby("load_tertile", observed=True).agg(
                n=("pressure", "size"), pressure_rate=("pressure", "mean")
            ).reset_index()
            st.plotly_chart(viz.effort_pressure_bar(tert), use_container_width=True)
            st.markdown(
                "Higher post-snap energy load (longer dropbacks, scrambles, extended "
                "rushes) coincides with a markedly higher pressure-event rate. This is "
                "the energy ↔ outcome link the practitioner asked for: effort and "
                "result move together. **Association, not causation** — longer plays "
                "both demand more energy and expose the pocket longer."
            )
        except ValueError:
            st.info("Not enough plays with tracking to tertile reliably.")


# ==========================================================================
# Shared widgets
# ==========================================================================
def _play_picker():
    with st.expander("Select a play", expanded=False):
        c1, c2, c3 = st.columns([1, 1, 2])
        games = sorted(PLAYS.gameId.unique().tolist())
        gi = c1.selectbox(
            "Game", games, index=games.index(st.session_state.sel_game)
            if st.session_state.sel_game in games else 0,
        )
        gplays = PLAYS[PLAYS.gameId == gi].sort_values("playId")
        pids = gplays.playId.tolist()
        default_p = (
            st.session_state.sel_play if st.session_state.sel_play in pids else pids[0]
        )
        pi = c2.selectbox("Play", pids, index=pids.index(default_p))
        rr = gplays[gplays.playId == pi].iloc[0]
        c3.caption(str(rr.playDescription)[:120])
        if c1.button("Load play"):
            set_play(gi, pi)
    st.caption(
        f"Current: **Game {st.session_state.sel_game} · Play {st.session_state.sel_play}**"
    )


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())


def _safe(v):
    try:
        if pd.isna(v):
            return "—"
        return int(v)
    except (TypeError, ValueError):
        return v


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
if "Command Centre" in screen:
    screen_command_centre()
elif "Play Forensics" in screen:
    screen_forensics()
elif "Counterfactual Lab" in screen:
    screen_counterfactual()
elif "Ask PocketShift" in screen:
    screen_ask()
elif "Energy & Fatigue" in screen:
    screen_energy_fatigue()
else:
    screen_preparation()
