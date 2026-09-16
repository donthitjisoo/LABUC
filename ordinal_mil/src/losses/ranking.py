"""Ordinal ranking loss on the continuous severity z, with *informative* pair
sampling rather than every pair in the batch.

Adjacent-class pairs (MES 0v1, 1v2, 2v3) are sampled most heavily since
they're the boundaries we actually care about discriminating; wider pairs
(0v2, 0v3, 1v3) are included at a lower rate as an easier auxiliary signal.
Pair counts actually drawn are returned every call so the caller can log
them (per your point 3: verify empirically, don't just assume).
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

NUM_CLASSES = 4


def sample_informative_pairs(
    y: torch.Tensor, pairs_per_delta: Dict[int, int], rng: random.Random,
) -> List[Tuple[int, int]]:
    """y: [B] int labels. Returns (i,j) index pairs with y[i] < y[j].

    pairs_per_delta: e.g. {1: 12, 2: 6, 3: 3} -- max pairs to draw for each
    class-distance. Fewer are drawn if the batch doesn't have that many
    distinct (class_a, class_b) index combinations available.
    """
    y_list = y.tolist()
    class_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, c in enumerate(y_list):
        class_to_indices[c].append(idx)

    pairs: List[Tuple[int, int]] = []
    for a in range(NUM_CLASSES):
        for b in range(a + 1, NUM_CLASSES):
            delta = b - a
            budget = pairs_per_delta.get(delta, 0)
            if budget <= 0:
                continue
            idx_a, idx_b = class_to_indices.get(a), class_to_indices.get(b)
            if not idx_a or not idx_b:
                continue
            n = min(budget, len(idx_a) * len(idx_b))
            for _ in range(n):
                pairs.append((rng.choice(idx_a), rng.choice(idx_b)))
    return pairs


class RankingLoss(nn.Module):
    def __init__(self, margin: float = 0.3, pairs_per_delta: Dict[int, int] = None, seed: int = 42):
        super().__init__()
        self.margin = margin
        self.pairs_per_delta = pairs_per_delta or {1: 12, 2: 6, 3: 3}
        self.rng = random.Random(seed)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, int]]:
        z = z.squeeze(-1)  # [B]
        pairs = sample_informative_pairs(y, self.pairs_per_delta, self.rng)

        stats: Dict[int, int] = defaultdict(int)
        if not pairs:
            return z.new_zeros(()), dict(stats)

        idx_lo = torch.tensor([p[0] for p in pairs], device=z.device)  # lower Mayo class
        idx_hi = torch.tensor([p[1] for p in pairs], device=z.device)  # higher Mayo class
        deltas = (y[idx_hi] - y[idx_lo]).float()
        for d in deltas.tolist():
            stats[int(d)] += 1

        per_pair = torch.clamp(self.margin - (z[idx_hi] - z[idx_lo]), min=0.0)
        loss = (deltas * per_pair).mean()
        return loss, dict(stats)
