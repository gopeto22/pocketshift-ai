"""Plotly visuals + theme. All charts carry labelled axes and units."""
from __future__ import annotations

import math

import networkx as nx
import plotly.graph_objects as go

from . import config as C

T = C.THEME


def _base_layout(fig: go.Figure, title: str = "", height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=T["panel"],
        plot_bgcolor=T["panel"],
        font=dict(color=T["text"], size=13),
        title=dict(text=title, font=dict(size=16, color=T["text"])),
        margin=dict(l=50, r=20, t=50 if title else 20, b=50),
        height=height,
        colorway=[T["accent"], T["accent2"], T["caution"], T["breakdown"]],
    )
    fig.update_xaxes(gridcolor=T["grid"], zerolinecolor=T["grid"])
    fig.update_yaxes(gridcolor=T["grid"], zerolinecolor=T["grid"])
    return fig


def pressure_rate_bar(df, group_col: str, title: str) -> go.Figure:
    d = df.sort_values("pressure_rate", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=(d["pressure_rate"] * 100),
            y=d["group_value"].astype(str),
            orientation="h",
            marker_color=T["accent"],
            customdata=d["n_plays"],
            hovertemplate="%{y}<br>Pressure rate: %{x:.1f}%<br>n=%{customdata} plays<extra></extra>",
        )
    )
    fig = _base_layout(fig, title, height=max(320, 26 * len(d)))
    fig.update_xaxes(title_text="Pressure-event rate (%)")
    fig.update_yaxes(title_text=group_col)
    return fig


def scenario_compare_bar(result: dict) -> go.Figure:
    a, b = result["arm_a"], result["arm_b"]
    labels = [result["arm_a_label"], result["arm_b_label"]]
    rates = [(a["rate"] or 0) * 100, (b["rate"] or 0) * 100]
    ns = [a["n"], b["n"]]
    err = None
    if a.get("ci") and a["ci"][0] is not None:
        err = dict(
            type="data",
            symmetric=False,
            array=[(a["ci"][1] - a["rate"]) * 100, (b["ci"][1] - b["rate"]) * 100],
            arrayminus=[(a["rate"] - a["ci"][0]) * 100, (b["rate"] - b["ci"][0]) * 100],
            color=T["muted"],
        )
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=rates,
            marker_color=[T["breakdown"], T["accent2"]],
            error_y=err,
            customdata=ns,
            hovertemplate="%{x}<br>%{y:.1f}%<br>n=%{customdata} plays<extra></extra>",
        )
    )
    fig = _base_layout(fig, f"{result['outcome']} rate by configuration", height=340)
    fig.update_yaxes(title_text=f"{result['outcome']} rate (%)")
    fig.update_xaxes(title_text="Protection configuration (matched historical plays)")
    return fig


def calibration_plot(calibration: list[dict]) -> go.Figure:
    if not calibration:
        return _base_layout(go.Figure(), "Calibration (no data)")
    xs = [c["pred_mean"] for c in calibration]
    ys = [c["obs_rate"] for c in calibration]
    ns = [c["n"] for c in calibration]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=T["muted"]),
                   name="perfect")
    )
    fig.add_trace(
        go.Scatter(x=xs, y=ys, mode="markers+lines", marker=dict(size=10, color=T["accent"]),
                   customdata=ns, name="model",
                   hovertemplate="pred %{x:.2f}<br>obs %{y:.2f}<br>n=%{customdata}<extra></extra>")
    )
    fig = _base_layout(fig, "Calibration: predicted vs observed pressure rate", height=360)
    fig.update_xaxes(title_text="Predicted probability", range=[0, 1])
    fig.update_yaxes(title_text="Observed rate", range=[0, 1])
    return fig


def matchup_graph_figure(g: nx.DiGraph) -> go.Figure:
    """Interaction graph: blockers (left) vs rushers (right). Not spatial."""
    blockers = [n for n, d in g.nodes(data=True) if d.get("kind") == "blocker"]
    rushers = [n for n, d in g.nodes(data=True) if d.get("kind") == "rusher"]

    pos = {}
    for i, n in enumerate(blockers):
        pos[n] = (0.0, _spread(i, len(blockers)))
    for i, n in enumerate(rushers):
        pos[n] = (1.0, _spread(i, len(rushers)))

    edge_x, edge_y, edge_colors = [], [], []
    fig = go.Figure()
    for u, v, d in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        color = T["breakdown"] if d.get("pressure_allowed") else T["muted"]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(width=3 if d.get("pressure_allowed") else 1.5, color=color),
                hoverinfo="text",
                text=f"{g.nodes[u].get('name')} → {g.nodes[v].get('name')}<br>"
                     f"block: {d.get('block_type')}<br>"
                     f"pressure allowed: {'yes' if d.get('pressure_allowed') else 'no'}",
                showlegend=False,
            )
        )

    def _add_nodes(nodes, color, symbol, side):
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        texts, hovers = [], []
        for n in nodes:
            nd = g.nodes[n]
            texts.append(f"{nd.get('name')}<br>{nd.get('position')}")
            if nd.get("kind") == "rusher":
                tag = "SACK" if nd.get("sack") else "HIT" if nd.get("hit") else "HURRY" if nd.get("hurry") else ""
            else:
                tag = "beaten" if nd.get("beaten") else ""
            hovers.append(f"{nd.get('name')} ({nd.get('position')}) {tag}")
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="markers+text", text=texts, textposition="middle " + side,
                marker=dict(size=26, color=color, symbol=symbol, line=dict(width=1, color=T["bg"])),
                hovertext=hovers, hoverinfo="text", showlegend=True, name=side == "right" and "Blockers" or "Rushers",
            )
        )

    _add_nodes(blockers, T["accent2"], "square", "right")
    _add_nodes(rushers, T["breakdown"], "circle", "left")

    fig = _base_layout(fig, "Blocker → rusher interaction graph (PFF assignments, not spatial)", height=460)
    fig.update_xaxes(visible=False, range=[-0.4, 1.4])
    fig.update_yaxes(visible=False)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        annotations=[
            dict(x=0.0, y=1.06, xref="x", yref="paper", text="OFFENSE (blockers)",
                 showarrow=False, font=dict(color=T["accent2"])),
            dict(x=1.0, y=1.06, xref="x", yref="paper", text="DEFENSE (rushers)",
                 showarrow=False, font=dict(color=T["breakdown"])),
        ],
    )
    return fig


def _spread(i: int, n: int) -> float:
    if n <= 1:
        return 0.0
    return 1.0 - (2.0 * i / (n - 1))


# ---------------------------------------------------------------------------
# Tracking-derived fatigue / energy visuals (side-quest)
# ---------------------------------------------------------------------------
def fatigue_curve_figure(curve) -> go.Figure:
    """Dual-axis: cumulative energy load (bars) + per-play peak acceleration (line).

    A rising load with a falling late-game peak-acceleration line is the fatigue
    signature a performance coach looks for.
    """
    from plotly.subplots import make_subplots

    x = curve.play_order
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=x, y=curve.cum_eei, name="Cumulative energy index",
               marker_color=T["accent"], opacity=0.55,
               hovertemplate="Snap #%{x}<br>Cumulative EEI: %{y:.0f}<extra></extra>"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=x, y=curve.peak_accel, name="Peak acceleration (per play)",
                   mode="lines+markers", line=dict(color=T["caution"], width=2),
                   hovertemplate="Snap #%{x}<br>Peak accel: %{y:.1f} yd/s²<extra></extra>"),
        secondary_y=True,
    )
    # Trend line on peak acceleration to show late-game decline.
    if len(x) >= 3:
        import numpy as np

        coef = np.polyfit(x, curve.peak_accel, 1)
        trend = np.polyval(coef, x)
        fig.add_trace(
            go.Scatter(x=x, y=trend, name="Accel trend", mode="lines",
                       line=dict(color=T["breakdown"], width=2, dash="dash"),
                       hoverinfo="skip"),
            secondary_y=True,
        )
    fig = _base_layout(fig, "Energy load vs. explosive output over the game", height=380)
    fig.update_xaxes(title_text="Player snap number (chronological)")
    fig.update_yaxes(title_text="Cumulative energy index (proxy units)", secondary_y=False)
    fig.update_yaxes(title_text="Peak acceleration (yd/s²)", secondary_y=True)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      barmode="overlay")
    return fig


def play_energy_bar(players: list[dict]) -> go.Figure:
    d = sorted(players, key=lambda r: r["energy_index"])
    fig = go.Figure(
        go.Bar(
            x=[r["energy_index"] for r in d],
            y=[f"{r['name']} ({r['position']})" for r in d],
            orientation="h", marker_color=T["accent2"],
            customdata=[[r["distance_yd"], r["peak_speed"]] for r in d],
            hovertemplate="%{y}<br>Energy index: %{x:.1f}<br>"
                          "Distance: %{customdata[0]:.1f} yd<br>"
                          "Peak speed: %{customdata[1]:.1f} yd/s<extra></extra>",
        )
    )
    fig = _base_layout(fig, "Post-snap energy expenditure by player (this play)",
                       height=max(300, 28 * len(d)))
    fig.update_xaxes(title_text="Energy Expenditure Index (proxy units)")
    fig.update_yaxes(title_text="Player")
    return fig


def effort_pressure_bar(tertile_df) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=tertile_df["load_tertile"].astype(str),
            y=(tertile_df["pressure_rate"] * 100),
            marker_color=[T["accent2"], T["caution"], T["breakdown"]],
            customdata=tertile_df["n"],
            hovertemplate="%{x} effort<br>Pressure rate: %{y:.1f}%<br>"
                          "n=%{customdata} plays<extra></extra>",
        )
    )
    fig = _base_layout(fig, "Pressure rate by play energy-load tertile", height=320)
    fig.update_xaxes(title_text="Play energy-load tertile (post-snap effort)")
    fig.update_yaxes(title_text="Pressure-event rate (%)")
    return fig
