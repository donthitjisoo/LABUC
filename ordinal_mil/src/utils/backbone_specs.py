"""Backbone facts that config validation, the data transform, and model
construction all have to agree on -- patch size and input normalization --
with zero torch/timm/transformers dependency, so config loading stays usable
without a working ML-framework install.

Both are intrinsic to a pretrained checkpoint, not free config knobs: patch
size is baked into the conv stem, and the mean/std are the statistics the
weights were trained to expect. This module is the ONLY place either is
written down. Nothing else may hard-code them.
"""
from __future__ import annotations

from typing import Dict, Tuple

Stats = Tuple[Tuple[float, float, float], Tuple[float, float, float]]

# Standard ImageNet statistics. Correct for DINOv2 (dinov2/data/transforms.py:
# IMAGENET_DEFAULT_MEAN/STD) and for DINOv3's LVD-1689M web-image weights
# (dinov3 README, "standard ImageNet evaluation transform").
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# EndoViT's own dataset statistics, from its model card's
# process_single_image(dataset_mean=..., dataset_std=...). NOT ImageNet.
ENDOVIT_MEAN = (0.3464, 0.2280, 0.2228)
ENDOVIT_STD = (0.2520, 0.2128, 0.2093)

# backbone.name -> Hugging Face repo. Only the LVD-1689M (web image) weights:
# DINOv3's SAT-493M satellite weights need different normalization
# ((0.430, 0.411, 0.296) / (0.213, 0.156, 0.143)) and are deliberately not
# offered here, so a name can never be paired with the wrong statistics.
# All are gated: accept the license on each repo's page and be logged in
# (`huggingface-cli login` or HF_TOKEN) before first use.
DINOV3_HF_IDS: Dict[str, str] = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",          # 21M params
    "dinov3_vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",  # 29M
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",          # 86M
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",          # 300M
    "dinov3_vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",  # 840M
    "dinov3_vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",        # 6.7B
}


def _family(backbone_name: str) -> str:
    if backbone_name.startswith("dinov2"):
        return "dinov2"
    if backbone_name == "endovit":
        return "endovit"
    if backbone_name in DINOV3_HF_IDS:
        return "dinov3"
    if backbone_name.startswith("dinov3"):
        raise ValueError(
            f"Unknown DINOv3 backbone '{backbone_name}'. Choose from: {sorted(DINOV3_HF_IDS)}"
        )
    raise ValueError(f"Unknown backbone: {backbone_name}")


def resolve_patch_size(backbone_name: str) -> int:
    """14 for DINOv2 hub models, 16 for EndoViT and DINOv3."""
    return {"dinov2": 14, "endovit": 16, "dinov3": 16}[_family(backbone_name)]


def resolve_norm_stats(backbone_name: str) -> Stats:
    """(mean, std) the checkpoint was trained to expect, per RGB channel on
    [0, 1]-scaled pixels."""
    family = _family(backbone_name)
    if family == "endovit":
        return ENDOVIT_MEAN, ENDOVIT_STD
    return IMAGENET_MEAN, IMAGENET_STD
