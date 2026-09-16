"""Train one ablation config (A-E in this pass) with a single patient-grouped
train/val holdout split. Model selection is by best validation QWK, not
accuracy or loss. See README.md for the full ablation table.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from src.datasets.limuc import (
    LimucMILDataset,
    PatientAwareBalancedSampler,
    build_transform,
    load_from_csv,
    load_from_directory,
    patient_group_holdout_split,
    print_split_coverage,
    validate_coverage,
)
from src.losses.ordinal import OrdinalCoralLoss, RegressionLoss, WeightedCELoss, build_class_weights
from src.losses.ranking import RankingLoss
from src.metrics.metrics import compute_metrics, tune_thresholds_for_qwk
from src.models.mayo_mil import MayoMIL
from src.utils.config import load_config
from src.utils.inference import classify_default, classify_tuned, run_inference
from src.utils.seed import get_device, seed_everything


def build_records(cfg: dict):
    if cfg["data"]["mode"] == "csv":
        all_records = load_from_csv(cfg["data"]["csv_path"])
    else:
        all_records = load_from_directory(cfg["data"]["train_val_root"])
    train_records, val_records = patient_group_holdout_split(
        all_records, val_fraction=cfg["data"]["val_fraction"], seed=cfg["seed"]
    )
    splits = {"train": train_records, "val": val_records}
    print_split_coverage(splits)
    validate_coverage(splits, min_patients_per_class=1)
    return train_records, val_records


def build_dataloaders(cfg: dict, train_records, val_records):
    image_size = cfg["data"]["image_size"]
    train_tf = build_transform(image_size, train=True)
    val_tf = build_transform(image_size, train=False)
    train_ds = LimucMILDataset(train_records, train_tf)
    val_ds = LimucMILDataset(val_records, val_tf)

    if cfg["sampler"]["type"] == "patient_aware_balanced":
        sampler = PatientAwareBalancedSampler(train_records, seed=cfg["seed"])
    else:
        sampler = RandomSampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], sampler=sampler,
        num_workers=cfg["data"]["num_workers"], drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    return train_loader, val_loader


def build_losses(cfg: dict, train_records, device):
    class_counts = np.bincount([r.mayo for r in train_records], minlength=4)
    print(f"Train class distribution: {dict(enumerate(class_counts.tolist()))}")
    class_weights = build_class_weights(
        class_counts, cfg["loss"]["class_weighting"], beta=cfg["loss"]["effective_number_beta"]
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)

    losses = {}
    if cfg["model"]["head_type"] == "ce":
        losses["cls"] = WeightedCELoss(class_weights)
    else:
        losses["ordinal"] = OrdinalCoralLoss(class_weights).to(device)
        if cfg["loss"]["lambda_reg"] > 0:
            losses["reg"] = RegressionLoss()
        if cfg["loss"]["use_ranking"]:
            losses["rank"] = RankingLoss(
                margin=cfg["loss"]["ranking"]["margin"],
                pairs_per_delta={int(k): v for k, v in cfg["loss"]["ranking"]["pairs_per_delta"].items()},
                seed=cfg["seed"],
            )
    return losses


def build_optimizer_scheduler(model, cfg: dict):
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]

    param_groups = [{"params": other_params, "lr": cfg["train"]["lr"]}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg["train"]["backbone_lr"]})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["train"]["weight_decay"])

    warmup_epochs = cfg["train"]["warmup_epochs"]
    total_epochs = cfg["train"]["epochs"]

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def forward_and_loss(model, images, y, cfg, losses, device):
    images, y = images.to(device), y.to(device)
    out = model(images)
    components = {}
    pair_stats = {}

    if cfg["model"]["head_type"] == "ce":
        loss = losses["cls"](out["ce_logits"], y)
        components["cls"] = loss.item()
        return loss, components, pair_stats

    l_ord = losses["ordinal"](out["ordinal_logits"], y)
    total = cfg["loss"]["lambda_ordinal"] * l_ord
    components["ordinal"] = l_ord.item()

    if "reg" in losses:
        l_reg = losses["reg"](out["z"], y)
        total = total + cfg["loss"]["lambda_reg"] * l_reg
        components["reg"] = l_reg.item()

    if "rank" in losses:
        l_rank, pair_stats = losses["rank"](out["z"], y)
        total = total + cfg["loss"]["lambda_rank"] * l_rank
        components["rank"] = float(l_rank.item()) if torch.is_tensor(l_rank) else l_rank

    return total, components, pair_stats


def evaluate_split(model, loader, cfg, device):
    result = run_inference(model, loader, device)
    head_type = cfg["model"]["head_type"]
    y_pred = classify_default(result, head_type)
    z = result.get("z")
    probs = result.get("probs")
    metrics = compute_metrics(result["y_true"], y_pred, z=z, probs=probs)
    return metrics, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = get_device()
    print(f"Experiment: {cfg['experiment_name']}  |  device: {device}")

    train_records, val_records = build_records(cfg)
    train_loader, val_loader = build_dataloaders(cfg, train_records, val_records)

    model = MayoMIL(cfg).to(device)
    losses = build_losses(cfg, train_records, device)
    optimizer, scheduler = build_optimizer_scheduler(model, cfg)

    log_dir = Path(cfg["train"]["log_dir"]) / cfg["experiment_name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))
    except ImportError:
        writer = None

    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_qwk, patience_counter, best_epoch = -2.0, 0, -1
    best_ckpt_path = log_dir / "best.pt"

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        epoch_components = defaultdict(list)
        epoch_pair_stats = defaultdict(int)

        for images, y, _patient_ids, _paths in train_loader:
            optimizer.zero_grad()
            # amp is only ever enabled when device.type=="cuda" (see use_amp above),
            # so device_type="cuda" here is safe even though it's unused when disabled.
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss, components, pair_stats = forward_and_loss(model, images, y, cfg, losses, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            for k, v in components.items():
                epoch_components[k].append(v)
            for delta, n in pair_stats.items():
                epoch_pair_stats[delta] += n

        scheduler.step()

        val_metrics, val_result = evaluate_split(model, val_loader, cfg, device)
        val_qwk = val_metrics["qwk"]

        loss_summary = "  ".join(f"{k}={np.mean(v):.4f}" for k, v in epoch_components.items())
        print(f"[epoch {epoch}] {loss_summary}  |  val_qwk={val_qwk:.4f}  val_mae={val_metrics['mae']:.4f}")
        if epoch_pair_stats:
            print(f"  ranking pairs used this epoch, by class-distance: {dict(epoch_pair_stats)}")

        if writer:
            for k, v in epoch_components.items():
                writer.add_scalar(f"train/{k}", np.mean(v), epoch)
            for k, v in val_metrics.items():
                if not k.startswith("_"):
                    writer.add_scalar(f"val/{k}", v, epoch)
            for delta, n in epoch_pair_stats.items():
                writer.add_scalar(f"train/rank_pairs_delta_{delta}", n, epoch)

        if val_qwk > best_qwk:
            best_qwk, best_epoch, patience_counter = val_qwk, epoch, 0
            tuned_taus = None
            if "z" in val_result:
                tuned_taus = tune_thresholds_for_qwk(val_result["z"], val_result["y_true"])
            torch.save({
                "config": cfg,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_qwk": val_qwk,
                "tuned_thresholds": tuned_taus.tolist() if tuned_taus is not None else None,
            }, best_ckpt_path)
            print(f"  -> new best (val_qwk={val_qwk:.4f}), saved to {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch} (no val_qwk improvement for "
                      f"{cfg['train']['early_stop_patience']} epochs).")
                break

    print(f"\nBest epoch: {best_epoch}  best val_qwk: {best_qwk:.4f}  checkpoint: {best_ckpt_path}")

    # Report both default- and tuned-threshold validation metrics for the best epoch.
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    val_metrics, val_result = evaluate_split(model, val_loader, cfg, device)
    summary = {"val_default": val_metrics}
    if ckpt.get("tuned_thresholds") is not None:
        taus = np.array(ckpt["tuned_thresholds"])
        y_pred_tuned = classify_tuned(val_result, taus)
        summary["val_tuned"] = compute_metrics(
            val_result["y_true"], y_pred_tuned, z=val_result.get("z"), probs=val_result.get("probs")
        )
        summary["tuned_thresholds"] = taus.tolist()

    with open(log_dir / "val_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved validation metrics (default + tuned thresholds) to {log_dir / 'val_metrics.json'}")


if __name__ == "__main__":
    main()
