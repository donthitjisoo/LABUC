"""YAML config loading with defaults + validation.

Configs are plain nested dicts (accessed as cfg["model"]["use_mil"], etc.) --
deliberately not a dataclass hierarchy, so new fields can be added in configs
without touching this file.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULTS: Dict[str, Any] = {
    "experiment_name": "unnamed",
    "seed": 42,
    "data": {
        "mode": "directory",  # "csv" or "directory"
        "csv_path": None,
        "train_val_root": "../data/train_and_validation_sets",
        "test_root": "../data/test_set",
        "val_fraction": 0.15,
        # Controls ONLY the patient train/val partition, independent of the
        # top-level `seed` (which controls model init, the balanced sampler,
        # ranking-pair sampling, and bootstrap CIs). Deliberately separate:
        # a multi-seed robustness run should vary training stochasticity
        # while holding the split fixed, not silently reshuffle patients
        # into different partitions each time too. Defaults to the same
        # value as `seed` so existing configs are unaffected unless this is
        # set explicitly.
        "split_seed": 42,
        "image_size": [224, 224],  # [H, W], must each be divisible by patch_size
        "patch_size": 14,
        "num_workers": 4,
    },
    "sampler": {
        "type": "patient_aware_balanced",  # "patient_aware_balanced" or "random"
    },
    "backbone": {
        "name": "dinov2_vits14_reg",  # or "endovit"
        "freeze": True,
        "unfreeze_last_n_blocks": 0,
    },
    "model": {
        "head_type": "ordinal",  # "ce", "cdw_ce", or "ordinal"
        "ordinal_head_type": "coral",  # "coral" or "corn" -- only used when head_type=="ordinal"
        "use_mil": False,
        "pooling": "gated_attention",  # mean/max/topk/attention/gated_attention
        "mil_attn_dim": 128,
        "topk_k": 32,
        "fusion_dim": 256,
        "dropout": 0.3,
    },
    "loss": {
        "use_ranking": False,
        # Weight on the primary classification loss -- whichever one
        # head_type selects (OrdinalCoralLoss for "ordinal", CDWCELoss for
        # "cdw_ce"). Named lambda_ordinal for historical reasons; applies to
        # either.
        "lambda_ordinal": 1.0,
        "lambda_rank": 0.25,
        "lambda_reg": 0.15,
        "cdw_alpha": 5.0,  # class-distance exponent for CDWCELoss; only used when head_type=="cdw_ce"
        "class_weighting": "effective_number",  # "none", "inverse_freq", "effective_number"
        "effective_number_beta": 0.999,
        "ranking": {
            "margin": 0.3,
            # int keys, matching how YAML parses {1: 12, ...} -- a str-keyed
            # default here would silently accumulate alongside int-keyed
            # overrides in _deep_merge instead of being replaced by them.
            "pairs_per_delta": {1: 12, 2: 6, 3: 3},
        },
    },
    "train": {
        "epochs": 100,
        "batch_size": 64,
        "lr": 1.0e-3,
        "backbone_lr": 1.0e-5,
        "weight_decay": 1.0e-4,
        "warmup_epochs": 5,
        "grad_clip_norm": 1.0,
        "amp": True,
        "early_stop_patience": 15,
        "log_dir": "runs_ordinal_mil",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULTS, user_cfg)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    h, w = cfg["data"]["image_size"]
    p = cfg["data"]["patch_size"]
    if h % p != 0 or w % p != 0:
        raise ValueError(
            f"image_size {(h, w)} must be divisible by patch_size {p} "
            f"(got remainders {h % p}, {w % p})."
        )
    if cfg["model"]["head_type"] not in ("ce", "cdw_ce", "ordinal"):
        raise ValueError(f"Unknown head_type: {cfg['model']['head_type']}")
    if cfg["model"]["head_type"] == "ce" and cfg["loss"]["use_ranking"]:
        raise ValueError(
            "Ranking loss requires a continuous severity z, which head_type='ce' doesn't "
            "produce -- use 'ordinal' or 'cdw_ce' instead."
        )
