"""Patch-level multiple-instance-learning pooling.

Input:  patch_tokens [B, N, D]
Output: pooled       [B, D]
        attn         [B, N]   (uniform 1/N for 'mean'/'max', which aren't
                                learned attention, but returned in the same
                                shape so downstream visualization code doesn't
                                need to special-case pooling type)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

POOLING_CHOICES = ("mean", "max", "topk", "attention", "gated_attention")


class MILPool(nn.Module):
    def __init__(self, pooling: str, dim: int, attn_dim: int = 128, topk_k: int = 32):
        super().__init__()
        if pooling not in POOLING_CHOICES:
            raise ValueError(f"pooling must be one of {POOLING_CHOICES}, got {pooling}")
        self.pooling = pooling
        self.topk_k = topk_k

        if pooling in ("attention", "gated_attention"):
            self.V = nn.Linear(dim, attn_dim)
            self.w = nn.Linear(attn_dim, 1)
            if pooling == "gated_attention":
                self.U = nn.Linear(dim, attn_dim)
        elif pooling == "topk":
            self.score = nn.Linear(dim, 1)

    def forward(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = patch_tokens.shape

        if self.pooling == "mean":
            pooled = patch_tokens.mean(dim=1)
            attn = patch_tokens.new_full((B, N), 1.0 / N)

        elif self.pooling == "max":
            pooled, _ = patch_tokens.max(dim=1)
            attn = patch_tokens.new_full((B, N), 1.0 / N)

        elif self.pooling == "topk":
            k = min(self.topk_k, N)
            scores = self.score(patch_tokens).squeeze(-1)          # [B, N]
            topk_val, topk_idx = scores.topk(k, dim=1)               # [B, k]
            topk_attn = F.softmax(topk_val, dim=1)                    # [B, k]
            attn = patch_tokens.new_zeros(B, N).scatter_(1, topk_idx, topk_attn)
            pooled = torch.bmm(attn.unsqueeze(1), patch_tokens).squeeze(1)

        elif self.pooling == "attention":
            a_logit = self.w(torch.tanh(self.V(patch_tokens))).squeeze(-1)   # [B, N]
            attn = F.softmax(a_logit, dim=1)
            pooled = torch.bmm(attn.unsqueeze(1), patch_tokens).squeeze(1)

        else:  # gated_attention
            gate = torch.tanh(self.V(patch_tokens)) * torch.sigmoid(self.U(patch_tokens))  # [B,N,attn_dim]
            a_logit = self.w(gate).squeeze(-1)                       # [B, N]
            attn = F.softmax(a_logit, dim=1)
            pooled = torch.bmm(attn.unsqueeze(1), patch_tokens).squeeze(1)

        return pooled, attn
