"""Historical comparison and counterfactual estimation.

All numbers here are deterministic and empirical. Nothing is causal: these are
matched observational comparisons over real plays, always reported with sample
size and uncertainty, and gated by a minimum-support threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------
MATCH_KEYS_DEFAULT = ["down_bucket", "distance_bucket", "offenseFormation"]


def find_similar_plays(
    plays_enriched: pd.DataFrame,
    *,
    down_bucket: str | None = None,
    distance_bucket: str | None = None,
    offense_formation: str | None = None,
    possession_team: str | None = None,
    defensive_team: str | None = None,
    personnel_o: str | None = None,
    coverage: str | None = None,
    exclude_key: tuple[int, int] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Return real plays matching the supplied controllable context filters."""
    df = plays_enriched
    mask = pd.Series(True, index=df.index)
    if down_bucket:
        mask &= df["down_bucket"] == down_bucket
    if distance_bucket:
        mask &= df["distance_bucket"] == distance_bucket
    if offense_formation:
        mask &= df["offenseFormation"] == offense_formation
    if possession_team:
        mask &= df["possessionTeam"] == possession_team
    if defensive_team:
        mask &= df["defensiveTeam"] == defensive_team
    if personnel_o:
        mask &= df["personnelO"] == personnel_o
    if coverage:
        mask &= df["pff_passCoverage"] == coverage
    out = df[mask]
    if exclude_key is not None:
        g, p = exclude_key
        out = out[~((out["gameId"] == g) & (out["playId"] == p))]
    if limit:
        out = out.head(limit)
    return out


# ---------------------------------------------------------------------------
# Empirical rates with bootstrap uncertainty
# ---------------------------------------------------------------------------
def _bootstrap_ci(
    values: np.ndarray, iters: int = C.BOOTSTRAP_ITERS, seed: int = C.BOOTSTRAP_SEED
) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(iters)
    for i in range(iters):
        sample = values[rng.integers(0, n, n)]
        means[i] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def empirical_pressure_rate(
    plays: pd.DataFrame, outcome: str = "pressure_event"
) -> dict[str, Any]:
    """Rate of `outcome` over the supplied plays with a bootstrap 95% CI."""
    vals = pd.to_numeric(plays[outcome], errors="coerce").dropna().to_numpy()
    n = int(len(vals))
    if n == 0:
        return {"n": 0, "rate": None, "ci": (None, None), "supported": False}
    rate = float(vals.mean())
    lo, hi = _bootstrap_ci(vals)
    return {
        "n": n,
        "rate": rate,
        "ci": (lo, hi),
        "supported": n >= C.MIN_SUPPORT,
        "outcome": outcome,
    }


# ---------------------------------------------------------------------------
# Scenario definitions (controllable, protection-related modifications)
# ---------------------------------------------------------------------------
def _split_extra_protection(plays: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    with_help = plays[plays["extra_protection"] == 1]
    without = plays[plays["extra_protection"] == 0]
    return with_help, without


def _split_play_action(plays: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pa = pd.to_numeric(plays["pff_playAction"], errors="coerce").fillna(0)
    return plays[pa == 1], plays[pa == 0]


SCENARIOS = {
    "extra_protection": {
        "label": "Keep an extra blocker in (6+ pass blockers)",
        "split": _split_extra_protection,
        "arm_a": "with extra protection",
        "arm_b": "standard protection",
    },
    "play_action": {
        "label": "Use play-action vs. straight dropback",
        "split": _split_play_action,
        "arm_a": "with play-action",
        "arm_b": "without play-action",
    },
}


@dataclass
class ScenarioResult:
    scenario: str
    label: str
    outcome: str
    arm_a_label: str
    arm_b_label: str
    arm_a: dict
    arm_b: dict
    diff: float | None
    diff_ci: tuple[float | None, float | None]
    supported: bool
    message: str
    match_filters: dict = field(default_factory=dict)
    evidence_type: str = C.EV_HISTORICAL

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def _diff_bootstrap(
    a_vals: np.ndarray, b_vals: np.ndarray, seed: int = C.BOOTSTRAP_SEED
) -> tuple[float, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    iters = C.BOOTSTRAP_ITERS
    na, nb = len(a_vals), len(b_vals)
    diffs = np.empty(iters)
    for i in range(iters):
        da = a_vals[rng.integers(0, na, na)].mean()
        db = b_vals[rng.integers(0, nb, nb)].mean()
        diffs[i] = da - db
    return float(a_vals.mean() - b_vals.mean()), (
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def compare_protection_configurations(
    plays_enriched: pd.DataFrame,
    scenario: str,
    *,
    outcome: str = "pressure_event",
    match_filters: dict | None = None,
) -> ScenarioResult:
    """Matched A/B comparison of a protection modification on real plays.

    match_filters narrows to a comparable context (e.g. the featured play's
    down/distance/formation). The split function then partitions that context
    into the modified arm (A) and the baseline arm (B).
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    spec = SCENARIOS[scenario]
    match_filters = match_filters or {}

    pool = find_similar_plays(plays_enriched, **match_filters)
    arm_a_df, arm_b_df = spec["split"](pool)

    a = empirical_pressure_rate(arm_a_df, outcome)
    b = empirical_pressure_rate(arm_b_df, outcome)

    supported = a["supported"] and b["supported"]
    if not supported:
        return ScenarioResult(
            scenario=scenario,
            label=spec["label"],
            outcome=outcome,
            arm_a_label=spec["arm_a"],
            arm_b_label=spec["arm_b"],
            arm_a=a,
            arm_b=b,
            diff=None,
            diff_ci=(None, None),
            supported=False,
            message=(
                "Not enough comparable historical evidence to estimate this "
                f"scenario reliably (need >= {C.MIN_SUPPORT} plays per arm; "
                f"have {a['n']} and {b['n']})."
            ),
            match_filters=match_filters,
        )

    a_vals = pd.to_numeric(arm_a_df[outcome], errors="coerce").dropna().to_numpy()
    b_vals = pd.to_numeric(arm_b_df[outcome], errors="coerce").dropna().to_numpy()
    diff, diff_ci = _diff_bootstrap(a_vals, b_vals)

    direction = "lower" if diff < 0 else "higher"
    msg = (
        f"In matched historical plays, {spec['arm_a']} shows a {outcome} rate of "
        f"{a['rate']:.1%} (n={a['n']}) vs {b['rate']:.1%} (n={b['n']}) for "
        f"{spec['arm_b']} — a {abs(diff):.1%} {direction} rate "
        f"(95% CI {diff_ci[0]:.1%} to {diff_ci[1]:.1%}). Observational, not causal."
    )
    return ScenarioResult(
        scenario=scenario,
        label=spec["label"],
        outcome=outcome,
        arm_a_label=spec["arm_a"],
        arm_b_label=spec["arm_b"],
        arm_a=a,
        arm_b=b,
        diff=diff,
        diff_ci=diff_ci,
        supported=True,
        message=msg,
        match_filters=match_filters,
    )


def context_from_play(play_row: pd.Series, keys: list[str] | None = None) -> dict:
    """Extract match filters from a play row for the Counterfactual Lab."""
    keys = keys or ["down_bucket", "distance_bucket", "offense_formation"]
    out: dict[str, Any] = {}
    if "down_bucket" in keys and pd.notna(play_row.get("down_bucket")):
        out["down_bucket"] = play_row["down_bucket"]
    if "distance_bucket" in keys and pd.notna(play_row.get("distance_bucket")):
        out["distance_bucket"] = play_row["distance_bucket"]
    if "offense_formation" in keys and pd.notna(play_row.get("offenseFormation")):
        out["offense_formation"] = play_row["offenseFormation"]
    return out
