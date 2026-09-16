"""Full metrics suite for ordinal MES 0-3 evaluation. Accuracy alone is
explicitly not the target metric -- QWK is primary; everything else here is
reported alongside it.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

NUM_CLASSES = 4


def per_class_sensitivity_specificity_precision(cm: np.ndarray) -> Dict[str, np.ndarray]:
    n = cm.shape[0]
    sens, spec, prec = np.zeros(n), np.zeros(n), np.zeros(n)
    total = cm.sum()
    for c in range(n):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        sens[c] = tp / max(tp + fn, 1)
        spec[c] = tn / max(tn + fp, 1)
        prec[c] = tp / max(tp + fp, 1)
    return {"sensitivity": sens, "specificity": spec, "precision": prec}


def adjacent_boundary_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """probs: [N,3] = P(Y>0), P(Y>1), P(Y>2). For each adjacent boundary
    (k, k+1), restrict to samples with true label in {k, k+1} and score with
    P(Y>k) as the binary classifier for "is it the higher grade"."""
    out: Dict[str, float] = {}
    for k in range(3):
        mask = (y_true == k) | (y_true == k + 1)
        if mask.sum() < 2 or len(set(y_true[mask].tolist())) < 2:
            out[f"mes{k}_vs_{k+1}_auroc"] = float("nan")
            out[f"mes{k}_vs_{k+1}_f1"] = float("nan")
            continue
        y_bin = (y_true[mask] == k + 1).astype(int)
        score = probs[mask, k]
        try:
            out[f"mes{k}_vs_{k+1}_auroc"] = float(roc_auc_score(y_bin, score))
        except ValueError:
            out[f"mes{k}_vs_{k+1}_auroc"] = float("nan")
        pred_bin = (score > 0.5).astype(int)
        out[f"mes{k}_vs_{k+1}_f1"] = float(f1_score(y_bin, pred_bin, zero_division=0))
    return out


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    z: Optional[np.ndarray] = None,
    probs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    diff = np.abs(y_true - y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    class_stats = per_class_sensitivity_specificity_precision(cm)

    metrics: Dict[str, float] = {
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mae": float(diff.mean()),
        "adjacent_error_rate": float((diff == 1).mean()),
        "severe_error_rate": float((diff >= 2).mean()),
        "within_one_accuracy": float((diff <= 1).mean()),
    }

    if z is not None:
        rho, _ = spearmanr(np.asarray(z).reshape(-1), y_true)
        metrics["spearman_z_vs_true"] = float(rho)
    rho_pred, _ = spearmanr(y_pred, y_true)
    metrics["spearman_pred_vs_true"] = float(rho_pred)

    for c in range(NUM_CLASSES):
        metrics[f"mes{c}_sensitivity"] = float(class_stats["sensitivity"][c])
        metrics[f"mes{c}_specificity"] = float(class_stats["specificity"][c])
        metrics[f"mes{c}_precision"] = float(class_stats["precision"][c])

    if probs is not None:
        metrics.update(adjacent_boundary_metrics(y_true, np.asarray(probs)))

    metrics["_confusion_matrix"] = cm.tolist()
    metrics["_confusion_matrix_normalized"] = cm_norm.tolist()
    return metrics


def tune_thresholds_for_qwk(
    z_val: np.ndarray, y_val: np.ndarray, grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Grid-search 3 cutpoints on z to maximize validation QWK. Validation
    data ONLY -- callers must never pass test-set z/y here."""
    z_val = np.asarray(z_val).reshape(-1)
    if grid is None:
        lo, hi = float(z_val.min()), float(z_val.max())
        grid = np.linspace(lo, hi, 25)  # 25^3 combos is enough resolution without being slow

    best_qwk, best_taus = -2.0, np.array([np.median(grid)] * 3)
    for t0 in grid:
        for t1 in grid[grid > t0]:
            for t2 in grid[grid > t1]:
                pred = (z_val[:, None] > np.array([t0, t1, t2])[None, :]).sum(axis=1)
                qwk = cohen_kappa_score(y_val, pred, weights="quadratic")
                if qwk > best_qwk:
                    best_qwk, best_taus = qwk, np.array([t0, t1, t2])
    return best_taus
