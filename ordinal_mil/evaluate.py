"""One-shot evaluation on the held-out test set. Run exactly once per trained
checkpoint -- this is the only script that reads data/test_set.

Reports both default (each head's own 0.5 rule) and validation-tuned
threshold metrics, plus patient-cluster bootstrap 95% CIs for QWK, macro-F1,
MAE, accuracy, and balanced accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.limuc import LimucMILDataset, build_transform, load_from_csv, load_from_directory
from src.metrics.metrics import compute_metrics
from src.models.mayo_mil import MayoMIL
from src.utils.inference import classify_default, classify_tuned, run_inference
from src.utils.patient_bootstrap import patient_cluster_bootstrap
from src.utils.seed import get_device, seed_everything


def build_test_records(cfg: dict):
    if cfg["data"]["mode"] == "csv":
        records = load_from_csv(cfg["data"]["csv_path"])
        records = [r for r in records if r.split_hint == "test"]
        if not records:
            raise RuntimeError("No rows with split=='test' found in the CSV.")
        return records
    return load_from_directory(cfg["data"]["test_root"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="best.pt from train.py")
    parser.add_argument("--output-dir", default=None, help="Defaults to the checkpoint's directory.")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--save-attention", action="store_true",
                         help="Also save per-image MIL attention weights (attention.npz) for "
                              "use_mil configs (D/E) -- needed by visualize_attention.py.")
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    seed_everything(cfg["seed"])

    model = MayoMIL(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test_records = build_test_records(cfg)
    test_tf = build_transform(cfg["data"]["image_size"], train=False)
    test_loader = DataLoader(
        LimucMILDataset(test_records, test_tf), batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"],
    )

    capture_attn = args.save_attention and cfg["model"]["use_mil"]
    if args.save_attention and not cfg["model"]["use_mil"]:
        print("--save-attention given but this config has use_mil=False -- no attention to save.")
    result = run_inference(model, test_loader, device, capture_attn=capture_attn)
    head_type = cfg["model"]["head_type"]
    y_pred_default = classify_default(result, head_type)

    metrics = {"default": compute_metrics(
        result["y_true"], y_pred_default, z=result.get("z"), probs=result.get("probs")
    )}

    y_pred_for_csv = y_pred_default
    if ckpt.get("tuned_thresholds") is not None and "z" in result:
        taus = np.array(ckpt["tuned_thresholds"])
        y_pred_tuned = classify_tuned(result, taus)
        metrics["tuned"] = compute_metrics(
            result["y_true"], y_pred_tuned, z=result.get("z"), probs=result.get("probs")
        )
        metrics["tuned_thresholds"] = taus.tolist()
        y_pred_for_csv = y_pred_tuned  # the reportable "final" prediction uses tuned thresholds

    print("\n=== Test metrics: default threshold ===")
    print(json.dumps({k: v for k, v in metrics["default"].items() if not k.startswith("_")}, indent=2))
    if "tuned" in metrics:
        print("\n=== Test metrics: validation-tuned threshold ===")
        print(json.dumps({k: v for k, v in metrics["tuned"].items() if not k.startswith("_")}, indent=2))

    # Predictions CSV: the artifact statistical_analysis.py will consume later.
    pred_df = pd.DataFrame({
        "patient_id": result["patient_id"],
        "image_path": result["path"],
        "true_mayo": result["y_true"],
        "predicted_mayo": y_pred_for_csv,
    })
    # z and probs are captured independently -- head_type=="cdw_ce" has z
    # (from its auxiliary severity head) but no probs (no ordinal_logits to
    # take a sigmoid of; it classifies via ce_logits argmax instead).
    if "z" in result:
        pred_df["severity_z"] = result["z"]
    if "probs" in result:
        pred_df["prob_gt0"] = result["probs"][:, 0]
        pred_df["prob_gt1"] = result["probs"][:, 1]
        pred_df["prob_gt2"] = result["probs"][:, 2]

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved per-image predictions to {pred_path}")

    if capture_attn and "attn" in result:
        attn_path = output_dir / "attention.npz"
        # Only include z/probs keys when they actually exist -- np.savez(key=None)
        # doesn't omit the key, it silently stores a 0-d object array containing
        # None, which would make visualize_attention.py's `"z" in data.files`
        # check pass and then fail trying to index a 0-d array.
        attn_arrays = {
            "attn": result["attn"],            # [N, num_patches]
            "grid_hw": result["grid_hw"],       # (grid_h, grid_w)
            "path": result["path"],
            "patient_id": result["patient_id"],
            "y_true": result["y_true"],
            "y_pred": y_pred_for_csv,
            "image_size": np.array(cfg["data"]["image_size"]),
        }
        if "z" in result:
            attn_arrays["z"] = result["z"]
        if "probs" in result:
            attn_arrays["probs"] = result["probs"]
        np.savez(attn_path, **attn_arrays)
        print(f"Saved per-image attention weights to {attn_path}")

    # Patient-cluster bootstrap CIs (primary uncertainty estimate -- images
    # from the same patient are not independent).
    boot_df = pred_df.rename(columns={"true_mayo": "y_true", "predicted_mayo": "y_pred"})
    boot_results = patient_cluster_bootstrap(boot_df, n_boot=args.n_boot, seed=cfg["seed"])
    print(f"\n=== Patient-cluster bootstrap 95% CIs (n_boot={args.n_boot}, using tuned-threshold predictions) ===")
    for name, (point, lo, hi) in boot_results.items():
        print(f"  {name}: {point:.4f}  [{lo:.4f}, {hi:.4f}]")

    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            **metrics,
            "patient_cluster_bootstrap_95ci": {
                k: {"point": v[0], "ci_low": v[1], "ci_high": v[2]} for k, v in boot_results.items()
            },
        }, f, indent=2)
    print(f"Saved full test metrics to {metrics_path}")


if __name__ == "__main__":
    main()
