"""Find clusters in a PaCMAP projection and split the dataset into subsets.

Takes coords.npz (written by src/visualize_embeddings.py) and clusters the
'pacmap' 2D layout with HDBSCAN -- density-based, so it doesn't need a
pre-specified number of clusters (unlike k-means) and isn't sensitive to
PaCMAP's arbitrary coordinate scale the way a fixed-eps DBSCAN would be.
Points HDBSCAN can't confidently assign are labeled -1 ("noise"), not forced
into the nearest cluster.

Uses the SAME fixed SEED as visualize_embeddings.py.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN

SEED = 42
CLASS_NAMES = ["Mayo 0", "Mayo 1", "Mayo 2", "Mayo 3"]


def plot_clusters(coords, cluster_labels, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    unique = sorted(set(cluster_labels.tolist()))
    cmap = plt.get_cmap("tab20")
    for c in unique:
        mask = cluster_labels == c
        if c == -1:
            ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.4,
                       color="lightgray", marker="x", label="noise")
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.8,
                       color=cmap(c % 20), label=f"cluster {c}")
    ax.set_title("PaCMAP -- HDBSCAN clusters")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster the PaCMAP projection and split the dataset into per-cluster subsets."
    )
    parser.add_argument("--coords", required=True,
                         help="coords.npz written by visualize_embeddings.py (must include 'pacmap').")
    parser.add_argument("--min-cluster-size", type=int, default=25,
                         help="HDBSCAN min_cluster_size -- smallest group size to count as a cluster.")
    parser.add_argument("--min-samples", type=int, default=None,
                         help="HDBSCAN min_samples (defaults to min-cluster-size if omitted).")
    parser.add_argument("--output-dir", default=None,
                         help="Defaults to the coords.npz's own directory.")
    args = parser.parse_args()

    coords_path = Path(args.coords)
    data = np.load(coords_path, allow_pickle=True)
    if "pacmap" not in data:
        raise SystemExit(
            f"{coords_path} has no 'pacmap' entry -- rerun visualize_embeddings.py "
            f"with pacmap included in --methods first."
        )

    xy = data["pacmap"]
    labels = data["labels"]
    patient_ids = data["patient_ids"]
    paths = data["paths"]
    splits = data["splits"]

    clusterer = HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
    cluster_ids = clusterer.fit_predict(xy)

    n_clusters = len(set(cluster_ids.tolist()) - {-1})
    n_noise = int((cluster_ids == -1).sum())
    print(f"Found {n_clusters} clusters ({n_noise}/{len(cluster_ids)} points labeled as noise).")

    output_dir = Path(args.output_dir) if args.output_dir else coords_path.parent
    subset_dir = output_dir / "pacmap_clusters"
    subset_dir.mkdir(parents=True, exist_ok=True)

    # One combined CSV with every point's cluster assignment ...
    combined_path = output_dir / "pacmap_clusters.csv"
    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "path", "patient_id", "mayo_label", "split", "x", "y"])
        for i in range(len(cluster_ids)):
            writer.writerow([cluster_ids[i], paths[i], patient_ids[i], labels[i], splits[i],
                              xy[i, 0], xy[i, 1]])
    print(f"Saved {combined_path}")

    # ... plus one subset file per cluster, for actually pulling images out.
    for c in sorted(set(cluster_ids.tolist())):
        mask = cluster_ids == c
        name = "noise" if c == -1 else f"cluster_{c}"
        subset_path = subset_dir / f"{name}.csv"
        with open(subset_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "patient_id", "mayo_label", "split"])
            for i in np.where(mask)[0]:
                writer.writerow([paths[i], patient_ids[i], labels[i], splits[i]])
        print(f"  {name}: {mask.sum()} images -> {subset_path}")

    # Cross-tab: does each cluster line up with a Mayo class, or look mixed
    # (a sign it's a batch/scanner effect rather than a clinical one)?
    crosstab_path = output_dir / "pacmap_clusters_vs_mayo.csv"
    with open(crosstab_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "n_total"] + CLASS_NAMES)
        for c in sorted(set(cluster_ids.tolist())):
            mask = cluster_ids == c
            counts = [int(((labels == cls) & mask).sum()) for cls in range(len(CLASS_NAMES))]
            name = "noise" if c == -1 else str(c)
            writer.writerow([name, int(mask.sum())] + counts)
    print(f"Saved {crosstab_path}")

    plot_clusters(xy, cluster_ids, output_dir / "pacmap_clusters.png")


if __name__ == "__main__":
    main()
