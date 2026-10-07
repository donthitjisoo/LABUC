"""Vision backbones exposing a common (global_token, patch_tokens, grid_hw)
interface, with freeze / partial-unfreeze control.

Shapes:
  images:          [B, 3, H, W]
  global_token:     [B, D]
  patch_tokens:     [B, N, D]      N = (H/patch_size) * (W/patch_size)
  grid_hw:          (H/patch_size, W/patch_size)   -- for reshaping attention back to 2D

Input normalization is NOT handled here: every backbone expects images
already normalized with its own statistics, which live in
utils/backbone_specs.resolve_norm_stats and are applied by the data transform.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ..utils.backbone_specs import DINOV3_HF_IDS, resolve_patch_size

__all__ = [
    "BackboneWrapper", "DINOv2Backbone", "DINOv3Backbone", "EndoViTBackbone",
    "build_backbone", "resolve_patch_size",
]


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
    in this repo's foundation_models.py.

    Expects EndoViT's own input statistics, not ImageNet's -- see
    utils/backbone_specs.ENDOVIT_MEAN/STD."""

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


class DINOv3Backbone(BackboneWrapper):
    """DINOv3 ViT (LVD-1689M web weights) via Hugging Face transformers
    (>= 4.56). Gated: accept the license on the model's page and be logged in.

    The model returns one already-LayerNorm'd sequence laid out as
    [CLS, register tokens, patch tokens]. The registers are not spatial, so
    they are dropped here exactly as DINOv2's are -- they must not reach the
    MIL pooling or the attention maps.
    """

    def __init__(self, name: str = "dinov3_vits16", patch_size: int = 16):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(DINOV3_HF_IDS[name])
        cfg = self.model.config
        if int(cfg.patch_size) != patch_size:
            raise ValueError(f"{name}: checkpoint patch size {cfg.patch_size} != expected {patch_size}")
        self.embed_dim = int(cfg.hidden_size)
        self.patch_size = patch_size
        self.num_prefix_tokens = 1 + int(getattr(cfg, "num_register_tokens", 0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        _, _, H, W = x.shape
        tokens = self.model(pixel_values=x).last_hidden_state  # [B, 1 + R + N, D]
        grid_hw = (H // self.patch_size, W // self.patch_size)
        global_token = tokens[:, 0]
        patch_tokens = tokens[:, self.num_prefix_tokens:]
        if patch_tokens.shape[1] != grid_hw[0] * grid_hw[1]:
            raise RuntimeError(
                f"DINOv3 token layout mismatch: got {patch_tokens.shape[1]} patch tokens after "
                f"dropping {self.num_prefix_tokens} prefix tokens, expected "
                f"{grid_hw[0]}x{grid_hw[1]}={grid_hw[0] * grid_hw[1]}."
            )
        return global_token, patch_tokens, grid_hw

    def _blocks(self) -> nn.ModuleList:
        # The block list is `model.layer` in transformers 4.56 and
        # `model.model.layer` on later versions, so find it instead of
        # hard-coding either path.
        depth = self.model.config.num_hidden_layers
        for module_name, module in self.model.named_modules():
            if (isinstance(module, nn.ModuleList) and module_name.split(".")[-1] == "layer"
                    and len(module) == depth):
                return module
        raise RuntimeError("Could not locate the DINOv3 transformer blocks to unfreeze.")

    def unfreeze_last_n_blocks(self, n: int) -> None:
        if n <= 0:
            return
        for block in self._blocks()[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.model.norm.parameters():
            p.requires_grad = True


def build_backbone(cfg: dict) -> BackboneWrapper:
    name = cfg["name"]
    patch_size = resolve_patch_size(name)  # also rejects unknown names
    if name.startswith("dinov2"):
        backbone: BackboneWrapper = DINOv2Backbone(name=name, patch_size=patch_size)
    elif name == "endovit":
        backbone = EndoViTBackbone(patch_size=patch_size)
    elif name.startswith("dinov3"):
        backbone = DINOv3Backbone(name=name, patch_size=patch_size)
    else:
        raise ValueError(f"Unknown backbone: {name}")

    if cfg.get("freeze", True):
        backbone.freeze()
        backbone.unfreeze_last_n_blocks(cfg.get("unfreeze_last_n_blocks", 0))
    return backbone
