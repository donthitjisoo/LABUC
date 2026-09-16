"""Patient-cluster bootstrap: the primary uncertainty estimate for this
project, because images from the same patient are not independent samples.

Procedure per bootstrap iteration:
  1. sample patient ids WITH replacement (same count as the original set)
  2. take ALL images belonging to each sampled patient (a resampled patient
     drawn twice contributes its images twice)
  3. compute the metric on that pooled set

This is deliberately not a naive per-image bootstrap, which would understate
uncertainty by treating correlated same-patient images as independent draws.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

REQUIRED_COLUMNS = ("patient_id", "y_true", "y_pred")


def _qwk(df: pd.DataFrame) -> float:
    return float(cohen_kappa_score(df["y_true"], df["y_pred"], weights="quadratic"))


def _macro_f1(df: pd.DataFrame) -> float:
    return float(f1_score(df["y_true"], df["y_pred"], average="macro", zero_division=0))


def _mae(df: pd.DataFrame) -> float:
    return float((df["y_true"] - df["y_pred"]).abs().mean())


def _accuracy(df: pd.DataFrame) -> float:
    return float(accuracy_score(df["y_true"], df["y_pred"]))


def _balanced_accuracy(df: pd.DataFrame) -> float:
    return float(balanced_accuracy_score(df["y_true"], df["y_pred"]))


DEFAULT_METRICS: Dict[str, Callable[[pd.DataFrame], float]] = {
    "qwk": _qwk,
    "macro_f1": _macro_f1,
    "mae": _mae,
    "accuracy": _accuracy,
    "balanced_accuracy": _balanced_accuracy,
}


def patient_cluster_bootstrap(
    predictions: pd.DataFrame,
    metrics: Dict[str, Callable[[pd.DataFrame], float]] = None,
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, Tuple[float, float, float]]:
    """predictions must have columns patient_id, y_true, y_pred.

    Returns {metric_name: (point_estimate, ci_low, ci_high)}.
    """
    for col in REQUIRED_COLUMNS:
        if col not in predictions.columns:
            raise ValueError(f"predictions is missing required column '{col}'")

    metrics = metrics or DEFAULT_METRICS
    rng = np.random.default_rng(seed)
    patient_ids = predictions["patient_id"].unique()
    n_patients = len(patient_ids)

    by_patient = {pid: sub for pid, sub in predictions.groupby("patient_id")}

    boot_values: Dict[str, list] = {name: [] for name in metrics}
    for _ in range(n_boot):
        sampled = rng.choice(patient_ids, size=n_patients, replace=True)
        pooled = pd.concat([by_patient[pid] for pid in sampled], ignore_index=True)
        for name, fn in metrics.items():
            try:
                boot_values[name].append(fn(pooled))
            except ValueError:
                continue  # degenerate resample (e.g. single class present)

    alpha = (1 - ci) / 2
    results: Dict[str, Tuple[float, float, float]] = {}
    for name, fn in metrics.items():
        point = fn(predictions)
        vals = np.array(boot_values[name])
        lo, hi = (float("nan"), float("nan")) if len(vals) == 0 else (
            float(np.quantile(vals, alpha)), float(np.quantile(vals, 1 - alpha))
        )
        results[name] = (point, lo, hi)
    return results
