"""Ordinal classification heads for MES 0-3 (3 cumulative thresholds).

Two variants, selectable via config `model.ordinal_head_type`:

- "coral" (default): thresholds directly on the continuous severity scalar z.
  logits_k = scale * (z - tau_k), with tau_0 < tau_1 < tau_2 enforced via
  softplus-parameterized gaps. Monotonicity of P(Y>0)>=P(Y>1)>=P(Y>2) is
  structurally guaranteed for any z, not just encouraged by the loss.

- "corn": three independent linear heads off h_fused (not z). Monotonicity
  isn't structural here -- it's approximated by CORN's conditional training
  scheme (see losses/ordinal.py) -- kept as a configurable point of
  comparison against "coral", not the default.

Either way, z (continuous severity) is always computed and always used by the
ranking/regression losses and for visualization -- "corn" just means the
*classification* head doesn't read from it directly.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralCutpointHead(nn.Module):
    def __init__(self, num_thresholds: int = 3):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.tau0 = nn.Parameter(torch.zeros(1))
        self.log_gaps = nn.Parameter(torch.zeros(num_thresholds - 1))
        self.log_scale = nn.Parameter(torch.zeros(1))

    def thresholds(self) -> torch.Tensor:
        gaps = F.softplus(self.log_gaps)
        taus = [self.tau0.squeeze(0)]
        for i in range(self.num_thresholds - 1):
            taus.append(taus[-1] + gaps[i])
        return torch.stack(taus)  # [num_thresholds], strictly increasing

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B,1] -> logits: [B, num_thresholds]"""
        taus = self.thresholds()
        scale = F.softplus(self.log_scale) + 1e-3
        return scale * (z - taus.unsqueeze(0))

    @torch.no_grad()
    def predict_class(self, z: torch.Tensor, thresholds: Optional[torch.Tensor] = None) -> torch.Tensor:
        taus = thresholds if thresholds is not None else self.thresholds()
        return (z > taus.unsqueeze(0)).sum(dim=-1)


class CornIndependentHead(nn.Module):
    def __init__(self, in_dim: int, num_thresholds: int = 3):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(num_thresholds)])

    def forward(self, h_fused: torch.Tensor) -> torch.Tensor:
        """h_fused: [B, in_dim] -> logits: [B, num_thresholds]"""
        return torch.cat([h(h_fused) for h in self.heads], dim=1)

    @torch.no_grad()
    def predict_class(self, logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (torch.sigmoid(logits) > threshold).sum(dim=-1)


def logits_to_class(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Shared inference rule for both head types: predicted class = number of
    cumulative thresholds whose P(Y>k) exceeds `threshold`."""
    return (torch.sigmoid(logits) > threshold).sum(dim=-1)
