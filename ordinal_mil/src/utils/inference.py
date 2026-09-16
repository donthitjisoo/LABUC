"""Shared inference + classification-rule helpers used by both train.py
(validation, for early stopping + threshold fitting) and evaluate.py (the
one-shot held-out test pass).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


@torch.no_grad()
def run_inference(model, loader, device) -> Dict[str, np.ndarray]:
    model.eval()
    all_y, all_patient, all_path = [], [], []
    all_z, all_probs, all_ce_logits = [], [], []

    for images, y, patient_ids, paths in loader:
        out = model(images.to(device))
        all_y.append(y.numpy())
        all_patient.extend(patient_ids)
        all_path.extend(paths)
        if "z" in out:
            all_z.append(out["z"].cpu().numpy())
            all_probs.append(torch.sigmoid(out["ordinal_logits"]).cpu().numpy())
        if "ce_logits" in out:
            all_ce_logits.append(out["ce_logits"].cpu().numpy())

    result: Dict[str, np.ndarray] = {
        "y_true": np.concatenate(all_y),
        "patient_id": np.array(all_patient),
        "path": np.array(all_path),
    }
    if all_z:
        result["z"] = np.concatenate(all_z).reshape(-1)
        result["probs"] = np.concatenate(all_probs)  # [N,3] = P(Y>0), P(Y>1), P(Y>2)
    if all_ce_logits:
        result["ce_logits"] = np.concatenate(all_ce_logits)
    return result


def classify_default(result: Dict[str, np.ndarray], head_type: str) -> np.ndarray:
    """Each head's own native decision rule at probability 0.5. For the
    ordinal head this is mathematically identical to classifying by z against
    that head's own trained cutpoints (sigmoid(scale*(z-tau))>0.5 <=> z>tau),
    NOT an arbitrary fixed number -- it's "whatever training converged to."
    """
    if head_type == "ce":
        return result["ce_logits"].argmax(axis=1)
    return (result["probs"] > 0.5).sum(axis=1)


def classify_tuned(result: Dict[str, np.ndarray], taus: np.ndarray) -> np.ndarray:
    """Classify by z against externally-fit cutpoints (from
    metrics.tune_thresholds_for_qwk on validation data only)."""
    z = result["z"].reshape(-1, 1)
    return (z > taus.reshape(1, -1)).sum(axis=1)
