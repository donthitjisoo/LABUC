"""Project extracted LIMUC features to 2D with t-SNE, UMAP, PaCMAP, TriMap, and
PHATE, and plot them colored by Mayo class.

Input is a .npz produced by src/embed.py. Extra deps (see requirements.txt):
umap-learn, pacmap, trimap, phate, matplotlib.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

CLASS_NAMES = ["Mayo 0", "Mayo 1", "Mayo 2", "Mayo 3"]
CLASS_COLORS = ["#4c72b0", "#55a868", "#dd8452", "#c44e52"]


def run_tsne(X, seed, perplexity, n_neighbors):
    return TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca").fit_transform(X)


def run_umap(X, seed, perplexity, n_neighbors):
    import umap
    return umap.UMAP(n_components=2, random_state=seed, n_neighbors=n_neighbors).fit_transform(X)


def run_pacmap(X, seed, perplexity, n_neighbors):
    import pacmap
    return pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors, random_state=seed).fit_transform(X)


def run_trimap(X, seed, perplexity, n_neighbors):
    import trimap
    return trimap.TRIMAP(n_inliers=n_neighbors).fit_transform(X)


def run_phate(X, seed, perplexity, n_neighbors):
    import phate
    return phate.PHATE(n_components=2, knn=n_neighbors, random_state=seed, verbose=False).fit_transform(X)


METHODS = {
    "tsne": run_tsne,
    "umap": run_umap,
    "pacmap": run_pacmap,
    "trimap": run_trimap,
    "phate": run_phate,
}


def plot_embedding(ax, coords, labels, title):
    for cls in range(len(CLASS_NAMES)):
        mask = labels == cls
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.7,
                   color=CLASS_COLORS[cls], label=CLASS_NAMES[cls])
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def save_one(method, coords, y, output_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_embedding(ax, coords, y, method.upper())
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path = output_dir / f"{method}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot LIMUC feature embeddings with multiple DR methods.")
    parser.add_argument("--features", required=True, help="npz file produced by src/embed.py")
    parser.add_argument("--methods", nargs="+", default=list(METHODS.keys()), choices=list(METHODS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE only")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP / PaCMAP / TriMap / PHATE")
    parser.add_argument("--max-points", type=int, default=5000,
                         help="Subsample to this many points for speed (t-SNE/TriMap/PHATE scale poorly).")
    parser.add_argument("--output-dir", default="runs/embeddings/dr_plots",
                         help="Directory to save one PNG per method into.")
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X, y = data["features"], data["labels"]

    if X.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(X.shape[0], args.max_points, replace=False)
        X, y = X[idx], y[idx]
        print(f"Subsampled to {args.max_points} points for speed.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coords_out = {}
    for method in args.methods:
        print(f"Running {method} ...")
        coords = METHODS[method](X, args.seed, args.perplexity, args.n_neighbors)
        coords_out[method] = coords
        save_one(method, coords, y, output_dir)

    npz_path = output_dir / "coords.npz"
    np.savez(npz_path, labels=y, **coords_out)
    print(f"Saved raw 2D coordinates to {npz_path}")


if __name__ == "__main__":
    main()
