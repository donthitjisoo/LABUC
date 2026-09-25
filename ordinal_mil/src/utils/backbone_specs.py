"""Backbone architecture facts that both config validation and model
construction need to agree on, with zero torch/timm/etc. dependency so
config loading/validation stays usable without a working ML-framework
install (e.g. plain config sanity-checking, docs generation).
"""
from __future__ import annotations


def resolve_patch_size(backbone_name: str) -> int:
    """Patch size is intrinsic to a pretrained backbone's architecture (baked
    into its conv-stem stride and position embeddings), not a free config
    knob -- DINOv2 hub models are patch14, EndoViT is patch16. Single source
    of truth for both src.utils.config's image_size validation and
    src.models.backbones' actual model construction -- they must never read
    two independently-set numbers that can silently drift apart.
    """
    if backbone_name.startswith("dinov2"):
        return 14
    if backbone_name == "endovit":
        return 16
    raise ValueError(f"Unknown backbone: {backbone_name}")
