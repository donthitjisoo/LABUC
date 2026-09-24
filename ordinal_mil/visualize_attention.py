"""Attention-overlay visualization for the MIL configs (D/E). Consumes
attention.npz from `evaluate.py --save-attention`.

Attention is treated as a hypothesis, not ground-truth lesion localization
(per the design doc) -- this produces "high-attention region" overlays for
manual review, specifically to check whether attention concentrates on
mucosa or on scope borders/text overlays/specular highlights/artifacts. It
does not itself conclude anything.

Displays the MODEL'S ACTUAL INPUT (after the same resize+pad transform used
at eval), not the raw original image -- that's the only way the attention
grid is guaranteed to align spatially with what's shown.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.datasets.limuc import ResizeAndPadToMultiple

CLASS_NAMES = ["Mayo 0", "Mayo 1", "Mayo 2", "Mayo 3"]


def load_display_image(path: str, image_size) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    resize_pad = ResizeAndPadToMultiple(tuple(image_size))
    return np.array(resize_pad(img))  # [H,W,3] uint8


def attn_to_heatmap(attn_vec: np.ndarray, grid_hw, image_size) -> np.ndarray:
    gh, gw = grid_hw
    grid = attn_vec.reshape(gh, gw)
    grid = grid / max(grid.max(), 1e-8)
    heat = Image.fromarray((grid * 255).astype(np.uint8)).resize(
        (image_size[1], image_size[0]), Image.BILINEAR  # PIL wants (W, H)
    )
    return np.array(heat).astype(np.float32) / 255.0


def save_overlay(image: np.ndarray, heat: np.ndarray, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(image)
    axes[0].set_title("input")
    axes[0].axis("off")
    axes[1].imshow(image)
    axes[1].imshow(heat, cmap="jet", alpha=0.45)
    axes[1].set_title("attention")
    axes[1].axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_topk_patches(image: np.ndarray, attn_vec: np.ndarray, grid_hw, image_size, k: int, out_path: Path) -> None:
    gh, gw = grid_hw
    patch_h, patch_w = image_size[0] // gh, image_size[1] // gw
    top_idx = np.argsort(attn_vec)[::-1][:k]
    fig, axes = plt.subplots(1, k, figsize=(2 * k, 2))
    if k == 1:
        axes = [axes]
    for ax, idx in zip(axes, top_idx):
        r, c = divmod(int(idx), gw)
        crop = image[r * patch_h:(r + 1) * patch_h, c * patch_w:(c + 1) * patch_w]
        ax.imshow(crop)
        ax.set_title(f"{attn_vec[idx]:.3f}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Overlay MIL attention on its highest/lowest-error predictions.")
    parser.add_argument("--attention", required=True, help="attention.npz from evaluate.py --save-attention")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-per-category", type=int, default=6)
    parser.add_argument("--topk-patches", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.attention, allow_pickle=True)
    attn = data["attn"]
    grid_hw = tuple(int(x) for x in data["grid_hw"])
    paths = data["path"]
    y_true = data["y_true"].astype(int)
    y_pred = data["y_pred"].astype(int)
    z = data["z"] if "z" in data.files else None
    probs = data["probs"] if "probs" in data.files else None
    image_size = tuple(int(x) for x in data["image_size"])
    diff = np.abs(y_true - y_pred)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.attention).parent / "attention_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = {
        "correct": np.where(diff == 0)[0],
        "adjacent_error": np.where(diff == 1)[0],
        "severe_error": np.where(diff >= 2)[0],
    }

    rng = np.random.default_rng(args.seed)
    for cat_name, idxs in categories.items():
        if len(idxs) == 0:
            print(f"{cat_name}: no examples in this category, skipping.")
            continue
        chosen = rng.choice(idxs, size=min(args.n_per_category, len(idxs)), replace=False)
        cat_dir = output_dir / cat_name
        cat_dir.mkdir(exist_ok=True)
        for i in chosen:
            image = load_display_image(str(paths[i]), image_size)
            heat = attn_to_heatmap(attn[i], grid_hw, image_size)
            title = f"true={CLASS_NAMES[y_true[i]]} pred={CLASS_NAMES[y_pred[i]]}"
            if z is not None:
                title += f"  z={float(z[i]):.2f}"
            if probs is not None:
                title += (f"  P(Y>0,1,2)=({probs[i,0]:.2f},{probs[i,1]:.2f},{probs[i,2]:.2f})")
            stem = Path(str(paths[i])).stem
            save_overlay(image, heat, cat_dir / f"{stem}_overlay.png", title)
            save_topk_patches(image, attn[i], grid_hw, image_size, args.topk_patches,
                               cat_dir / f"{stem}_toppatches.png")
        print(f"{cat_name}: saved {len(chosen)} examples to {cat_dir}")

    print(f"\nDone -- inspect {output_dir}. Specifically check whether attention lands on "
          f"mucosa vs. scope borders / text overlays / specular highlights / debris.")


if __name__ == "__main__":
    main()
