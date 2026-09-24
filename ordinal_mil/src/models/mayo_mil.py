"""Assembled model: backbone -> (optional MIL) -> fusion -> severity z ->
ordinal head, OR backbone -> fusion -> plain 4-way CE head for Baseline A.

Every component after the backbone is config-gated so the same class serves
every row of the ablation table (A-E in this pass; F/G to follow).

Shapes (see backbones.py / mil.py / ordinal_head.py for the pieces):
  images:          [B,3,H,W]
  global_token:     [B,D]
  patch_tokens:     [B,N,D]
  h (pre-fusion):   [B,D] or [B,2D] if use_mil
  h_fused:          [B,fusion_dim]
  z:                [B,1]
  ordinal_logits:   [B,3]
  ce_logits:        [B,4]
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .backbones import build_backbone
from .mil import MILPool
from .ordinal_head import CoralCutpointHead, CornIndependentHead


class FusionMLP(nn.Module):
    def __init__(self, in_dim: int, fusion_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MayoMIL(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.backbone = build_backbone(cfg["backbone"])
        D = self.backbone.embed_dim
        m = cfg["model"]

        self.use_mil = bool(m["use_mil"])
        self.head_type = m["head_type"]  # "ce", "cdw_ce", or "ordinal"

        if self.use_mil:
            self.mil_pool: Optional[MILPool] = MILPool(
                pooling=m["pooling"], dim=D, attn_dim=m["mil_attn_dim"], topk_k=m["topk_k"]
            )
            fusion_in = D * 2
        else:
            self.mil_pool = None
            fusion_in = D

        self.fusion = FusionMLP(fusion_in, m["fusion_dim"], m["dropout"])

        if self.head_type == "ce":
            # Plain CE baseline: 4-way softmax only, no continuous z, so it's
            # deliberately NOT combinable with the ranking/regression losses
            # (those need z) -- see config.py's validate_config.
            self.ce_head = nn.Linear(m["fusion_dim"], 4)
            self.severity_head = None
            self.ordinal_head = None
        elif self.head_type == "cdw_ce":
            # Same 4-way softmax head as "ce", but ALSO exposes a z (like the
            # ordinal head) so this can be combined with MIL and the
            # ranking/regression losses in the same ablation matrix as
            # head_type=="ordinal", not just as a standalone baseline.
            self.ce_head = nn.Linear(m["fusion_dim"], 4)
            self.severity_head = nn.Linear(m["fusion_dim"], 1)
            self.ordinal_head = None
        else:
            self.ce_head = None
            self.severity_head = nn.Linear(m["fusion_dim"], 1)
            if m["ordinal_head_type"] == "coral":
                self.ordinal_head = CoralCutpointHead(num_thresholds=3)
            elif m["ordinal_head_type"] == "corn":
                self.ordinal_head = CornIndependentHead(in_dim=m["fusion_dim"], num_thresholds=3)
            else:
                raise ValueError(f"Unknown ordinal_head_type: {m['ordinal_head_type']}")
            self.ordinal_head_type = m["ordinal_head_type"]

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        global_token, patch_tokens, grid_hw = self.backbone(images)

        if self.use_mil:
            h_mil, attn = self.mil_pool(patch_tokens)
            h = torch.cat([global_token, h_mil], dim=-1)
        else:
            h = global_token
            attn = None

        h_fused = self.fusion(h)

        out: Dict[str, torch.Tensor] = {"h_fused": h_fused, "grid_hw": grid_hw}
        if attn is not None:
            out["attn"] = attn

        if self.head_type == "ce":
            out["ce_logits"] = self.ce_head(h_fused)
        elif self.head_type == "cdw_ce":
            out["ce_logits"] = self.ce_head(h_fused)
            out["z"] = self.severity_head(h_fused)
        else:
            z = self.severity_head(h_fused)
            out["z"] = z
            if self.ordinal_head_type == "coral":
                out["ordinal_logits"] = self.ordinal_head(z)
            else:
                out["ordinal_logits"] = self.ordinal_head(h_fused)

        return out
