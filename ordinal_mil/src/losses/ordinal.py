"""Ordinal (CORAL-style cumulative) loss, plain CE for the baseline, and
class-weighting utilities.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 4
NUM_THRESHOLDS = NUM_CLASSES - 1


def inverse_frequency_weights(class_counts: np.ndarray) -> torch.Tensor:
    counts = np.maximum(class_counts.astype(np.float64), 1)
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32)


def effective_number_weights(class_counts: np.ndarray, beta: float = 0.999) -> torch.Tensor:
    """Class-Balanced Loss weighting (Cui et al., 2019): weight ~ (1-beta) / (1-beta^n)."""
    counts = np.maximum(class_counts.astype(np.float64), 1)
    effective_num = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / effective_num
    w = w / w.sum() * len(counts)
    return torch.tensor(w, dtype=torch.float32)


def build_class_weights(class_counts: np.ndarray, method: str, beta: float = 0.999) -> Optional[torch.Tensor]:
    if method == "none":
        return None
    if method == "inverse_freq":
        return inverse_frequency_weights(class_counts)
    if method == "effective_number":
        return effective_number_weights(class_counts, beta=beta)
    raise ValueError(f"Unknown class_weighting method: {method}")


def ordinal_targets(y: torch.Tensor, num_thresholds: int = NUM_THRESHOLDS) -> torch.Tensor:
    """y: [B] int labels in {0..3} -> targets: [B, num_thresholds], target_k = 1[y>k]."""
    thresholds = torch.arange(num_thresholds, device=y.device).unsqueeze(0)  # [1,K]
    return (y.unsqueeze(1) > thresholds).float()


class OrdinalCoralLoss(nn.Module):
    """Mean-over-thresholds BCE, with an optional per-sample class weight
    (from build_class_weights, indexed by the sample's true class)."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("class_weights", class_weights, persistent=False)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        targets = ordinal_targets(y, logits.shape[1])
        per_threshold = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [B,K]
        per_sample = per_threshold.mean(dim=1)  # [B]
        if self.class_weights is not None:
            per_sample = per_sample * self.class_weights[y]
        return per_sample.mean()


class WeightedCELoss(nn.Module):
    """Plain 4-way cross-entropy for Baseline A, with the same class-weighting
    machinery as the ordinal loss for a fair comparison."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, y)


class RegressionLoss(nn.Module):
    """Auxiliary SmoothL1(z, y) -- deliberately weak (small lambda in config)
    so it nudges z toward the label scale without dominating ordinal learning."""

    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss(z.squeeze(-1), y.float())
