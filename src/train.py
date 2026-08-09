"""Train ResNet-50 / ResNet-152 / VGG-19 from scratch on LIMUC with patient-grouped k-fold CV.

Only --data-root (the train_and_validation_sets pool) is read here and split
into k folds; each fold's held-out patients serve as validation for that
fold. The test set is never touched by this script -- run evaluate.py
separately, once, against data/test_set for the final reportable numbers.
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import LimucDataset, index_images, make_folds
from models import build_model
from utils import AverageMeter, get_device, seed_everything

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def class_weights_for(samples, num_classes=4) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for s in samples:
        counts[s.label] += 1
    counts = counts.clamp(min=1)
    return counts.sum() / (num_classes * counts)


def run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = outputs.argmax(dim=1)
            acc_meter.update((preds == labels).float().mean().item(), n=labels.size(0))
            loss_meter.update(loss.item(), n=labels.size(0))
    return loss_meter.avg, acc_meter.avg


def train_one_fold(model_name, fold_idx, train_samples, val_samples, args, device) -> float:
    train_tf, eval_tf = build_transforms(args.img_size)
    train_loader = DataLoader(
        LimucDataset(train_samples, transform=train_tf),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        LimucDataset(val_samples, transform=eval_tf),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = build_model(model_name).to(device)
    weight = class_weights_for(train_samples).to(device) if args.class_weighted else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.output_dir) / model_name / f"fold_{fold_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"])
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, device, criterion, optimizer=None)
            scheduler.step()
            elapsed = time.time() - t0
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc,
                              optimizer.param_groups[0]["lr"], round(elapsed, 1)])
            f.flush()
            print(f"[{model_name} fold {fold_idx}] epoch {epoch}/{args.epochs} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({elapsed:.1f}s)")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {"model_name": model_name, "fold": fold_idx, "epoch": epoch,
                     "val_acc": val_acc, "state_dict": model.state_dict()},
                    out_dir / "best.pt",
                )

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train LIMUC classifiers from scratch with k-fold CV.")
    parser.add_argument("--data-root", default="data/train_and_validation_sets",
                         help="Training pool only -- data/test_set is never read by this script.")
    parser.add_argument("--models", nargs="+", default=["resnet50", "resnet152", "vgg19"],
                         choices=["resnet50", "resnet152", "vgg19"])
    parser.add_argument("--folds", type=int, default=5, choices=[3, 5])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--class-weighted", action=argparse.BooleanOptionalAction, default=True,
                         help="Weight the loss by inverse class frequency (LIMUC classes are imbalanced).")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    samples = index_images(args.data_root)
    print(f"Indexed {len(samples)} images from {args.data_root} "
          f"({len({s.patient_id for s in samples})} patients).")

    summary_rows = []
    for model_name in args.models:
        fold_accs = []
        for fold_idx, (train_samples, val_samples) in enumerate(
                make_folds(samples, args.folds, seed=args.seed), start=1):
            best_val_acc = train_one_fold(model_name, fold_idx, train_samples, val_samples, args, device)
            fold_accs.append(best_val_acc)
            summary_rows.append([model_name, fold_idx, len(train_samples), len(val_samples), best_val_acc])
        mean_acc = sum(fold_accs) / len(fold_accs)
        print(f"== {model_name}: {args.folds}-fold CV val accuracy = "
              f"{mean_acc:.4f} (folds: {[round(a, 4) for a in fold_accs]})")

    summary_path = Path(args.output_dir) / "cv_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "fold", "n_train", "n_val", "best_val_acc"])
        writer.writerows(summary_rows)
    print(f"Wrote CV summary to {summary_path}")


if __name__ == "__main__":
    main()
