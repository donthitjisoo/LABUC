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


def main():
    parser = argparse.ArgumentParser(description="Plot LIMUC feature embeddings with multiple DR methods.")
    parser.add_argument("--features", required=True, help="npz file produced by src/embed.py")
    parser.add_argument("--methods", nargs="+", default=list(METHODS.keys()), choices=list(METHODS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE only")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP / PaCMAP / TriMap / PHATE")
    parser.add_argument("--max-points", type=int, default=5000,
                         help="Subsample to this many points for speed (t-SNE/TriMap/PHATE scale poorly).")
    parser.add_argument("--output", default="runs/embeddings/dr_plots.png")
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X, y = data["features"], data["labels"]

    if X.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(X.shape[0], args.max_points, replace=False)
        X, y = X[idx], y[idx]
        print(f"Subsampled to {args.max_points} points for speed.")

    n = len(args.methods)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    coords_out = {}
    for ax, method in zip(axes, args.methods):
        print(f"Running {method} ...")
        coords = METHODS[method](X, args.seed, args.perplexity, args.n_neighbors)
        coords_out[method] = coords
        plot_embedding(ax, coords, y, method.upper())

    for ax in axes[len(args.methods):]:
        ax.axis("off")

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=len(CLASS_NAMES))
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

    npz_path = output_path.with_suffix(".coords.npz")
    np.savez(npz_path, labels=y, **coords_out)
    print(f"Saved raw 2D coordinates to {npz_path}")


if __name__ == "__main__":
    main()
