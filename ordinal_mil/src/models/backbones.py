"""Vision backbones exposing a common (global_token, patch_tokens, grid_hw)
interface, with freeze / partial-unfreeze control.

Shapes:
  images:          [B, 3, H, W]
  global_token:     [B, D]
  patch_tokens:     [B, N, D]      N = (H/patch_size) * (W/patch_size)
  grid_hw:          (H/patch_size, W/patch_size)   -- for reshaping attention back to 2D
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class BackboneWrapper(nn.Module):
    embed_dim: int
    patch_size: int

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        raise NotImplementedError

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int) -> None:
        raise NotImplementedError


class DINOv2Backbone(BackboneWrapper):
    """DINOv2 via torch.hub, register-token variant by default. Registers
    (if present) are kept separate from patch_tokens by the model itself --
    they're not spatial and shouldn't feed the MIL pooling / attention maps."""

    def __init__(self, name: str = "dinov2_vits14_reg", patch_size: int = 14):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", name)
        self.embed_dim = self.model.embed_dim
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        _, _, H, W = x.shape
        out = self.model.forward_features(x)
        global_token = out["x_norm_clstoken"]          # [B, D]
        patch_tokens = out["x_norm_patchtokens"]        # [B, N, D]
        grid_hw = (H // self.patch_size, W // self.patch_size)
        return global_token, patch_tokens, grid_hw

    def unfreeze_last_n_blocks(self, n: int) -> None:
        if n <= 0:
            return
        for block in self.model.blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.model.norm.parameters():
            p.requires_grad = True


class EndoViTBackbone(BackboneWrapper):
    """ViT-B/16, MAE-pretrained on GI endoscopy images (egeozsoy/EndoViT).
    Public, non-gated. Uses timm's forward_features to get the full token
    sequence (CLS + patches) rather than the pooled embedding used elsewhere
    in this repo's foundation_models.py."""

    MEAN = [0.3464, 0.2280, 0.2228]
    STD = [0.2520, 0.2128, 0.2093]

    def __init__(self, patch_size: int = 16):
        super().__init__()
        from functools import partial
        from pathlib import Path
        from huggingface_hub import snapshot_download
        from timm.models.vision_transformer import VisionTransformer

        model_dir = snapshot_download(repo_id="egeozsoy/EndoViT", revision="main")
        weights_path = Path(model_dir) / "pytorch_model.bin"

        self.model = VisionTransformer(
            patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0,
        )
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)["model"]
        self.model.load_state_dict(state_dict, strict=False)
        self.embed_dim = 768
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        _, _, H, W = x.shape
        tokens = self.model.forward_features(x)  # [B, 1+N, D]
        global_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        grid_hw = (H // self.patch_size, W // self.patch_size)
        return global_token, patch_tokens, grid_hw

    def unfreeze_last_n_blocks(self, n: int) -> None:
        if n <= 0:
            return
        for block in self.model.blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.model.norm.parameters():
            p.requires_grad = True


def build_backbone(cfg: dict) -> BackboneWrapper:
    name = cfg["name"]
    if name.startswith("dinov2"):
        backbone: BackboneWrapper = DINOv2Backbone(name=name, patch_size=cfg.get("patch_size", 14))
    elif name == "endovit":
        backbone = EndoViTBackbone(patch_size=cfg.get("patch_size", 16))
    else:
        raise ValueError(f"Unknown backbone: {name}")

    if cfg.get("freeze", True):
        backbone.freeze()
        backbone.unfreeze_last_n_blocks(cfg.get("unfreeze_last_n_blocks", 0))
    return backbone
