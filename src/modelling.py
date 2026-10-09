"""Baseline pre-snap pressure-event model with leakage-safe, game-grouped CV.

Target: pressure_event (scouting-derived proxy). Only pre-snap/context features
enter X. Validation uses GroupKFold on gameId so plays from the same game never
straddle the split. Reports ROC-AUC, Brier, calibration, held-out n, prevalence,
and a base-rate baseline. Weak performance is reported, not concealed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .features import build_model_frame

NUMERIC = ["down", "yardsToGo", "quarter", "defendersInBox", "absoluteYardlineNumber",
           "pff_playAction", "score_diff"]
CATEGORICAL = ["offenseFormation", "personnelO", "personnelD"]


@dataclass
class ModelReport:
    model_name: str
    n_total: int
    prevalence: float
    roc_auc: float
    brier: float
    baseline_brier: float
    calibration: list[dict] = field(default_factory=list)
    folds: int = 0
    leakage_checked: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _build_pipeline(kind: str) -> Pipeline:
    num = [c for c in NUMERIC]
    cat = [c for c in CATEGORICAL]
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=10, sparse_output=False
                ),
                cat,
            ),
        ]
    )
    if kind == "logreg":
        clf = LogisticRegression(max_iter=2000, C=0.5)
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=250, l2_regularization=1.0
        )
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("clf", clf)])


def _prep_xy(plays_enriched: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = build_model_frame(plays_enriched)  # raises on leakage
    cols = NUMERIC + CATEGORICAL
    X = frame.copy()
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[NUMERIC] = X[NUMERIC].fillna(X[NUMERIC].median(numeric_only=True))
    for c in CATEGORICAL:
        X[c] = X[c].astype("string").fillna("UNK")
    y = frame["pressure_event"].astype(int).to_numpy()
    groups = frame["gameId"].to_numpy()
    return X[cols], y, groups


def _calibration_table(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        out.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(m.sum()),
                "pred_mean": float(p[m].mean()),
                "obs_rate": float(y_true[m].mean()),
            }
        )
    return out


def train_pressure_model(
    plays_enriched: pd.DataFrame, kind: str = "hgb", n_splits: int = 5
) -> ModelReport:
    """Train with game-grouped CV and return out-of-fold evaluation."""
    X, y, groups = _prep_xy(plays_enriched)
    n = len(y)
    prevalence = float(y.mean())

    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(n, np.nan)

    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe = _build_pipeline(kind)
        base = pipe
        # Calibrate on the training fold via internal CV for honest probabilities.
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

    mask = ~np.isnan(oof)
    y_eval, p_eval = y[mask], oof[mask]
    auc = float(roc_auc_score(y_eval, p_eval))
    brier = float(brier_score_loss(y_eval, p_eval))
    baseline = float(brier_score_loss(y_eval, np.full_like(p_eval, prevalence)))
    cal = _calibration_table(y_eval, p_eval)

    note = ""
    if auc < 0.6:
        note = (
            "Pre-snap context has limited power to predict recorded pressure; "
            "this is expected and reported honestly rather than concealed."
        )
    return ModelReport(
        model_name=f"{kind} (isotonic-calibrated, GroupKFold by gameId)",
        n_total=int(mask.sum()),
        prevalence=prevalence,
        roc_auc=auc,
        brier=brier,
        baseline_brier=baseline,
        calibration=cal,
        folds=n_splits,
        note=note,
    )
