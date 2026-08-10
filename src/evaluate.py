"""Evaluate trained LIMUC checkpoints on the held-out test set (data/test_set).

This is the only script that reads data/test_set -- it must never be used
during training or fold-validation, per the LIMUC evaluation protocol.
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import LimucDataset, index_images
from models import build_model
from utils import get_device

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["Mayo 0", "Mayo 1", "Mayo 2", "Mayo 3"]


def eval_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_fold_checkpoints(run_dir: Path, model_name: str):
    ckpts = sorted((run_dir / model_name).glob("fold_*/best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {run_dir / model_name}/fold_*/best.pt")
    return ckpts


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        all_probs.append(F.softmax(model(images), dim=1).cpu())
        all_labels.append(labels)
    return torch.cat(all_probs), torch.cat(all_labels)


def report_confusion_and_f1(model_name, y_true, y_pred, run_dir: Path):
    labels = list(range(len(CLASS_NAMES)))
    # rows = predicted class, columns = true/original class
    cm = confusion_matrix(y_true, y_pred, labels=labels).T
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    print(f"\n{model_name} -- confusion matrix (rows=predicted, cols=actual):")
    header = "pred\\true".ljust(10) + "".join(f"{c:>8}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:<10}" + "".join(f"{v:>8d}" for v in row))

    print(f"\n{model_name} -- per-class metrics:")
    print(f"{'class':<10}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<10}{precision[i]:>10.4f}{recall[i]:>10.4f}{f1[i]:>10.4f}{support[i]:>10d}")

    cm_path = run_dir / f"{model_name}_confusion_matrix.csv"
    with open(cm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pred\\true"] + CLASS_NAMES)
        for i, row in enumerate(cm):
            writer.writerow([CLASS_NAMES[i]] + list(row))

    metrics_path = run_dir / f"{model_name}_class_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([name, precision[i], recall[i], f1[i], support[i]])

    print(f"Wrote {cm_path} and {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LIMUC checkpoints on the held-out test set.")
    parser.add_argument("--test-root", default="data/test_set")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--models", nargs="+", default=["resnet50", "resnet152", "vgg19"],
                         choices=["resnet50", "resnet152", "vgg19"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    samples = index_images(args.test_root)
    print(f"Indexed {len(samples)} test images from {args.test_root}.")
    test_loader = DataLoader(
        LimucDataset(samples, transform=eval_transform(args.img_size)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    results = []
    for model_name in args.models:
        ckpts = load_fold_checkpoints(Path(args.run_dir), model_name)
        fold_probs, labels = [], None
        for ckpt_path in ckpts:
            ckpt = torch.load(ckpt_path, map_location=device)
            model = build_model(model_name).to(device)
            model.load_state_dict(ckpt["state_dict"])
            probs, labels = predict_probs(model, test_loader, device)
            acc = (probs.argmax(dim=1) == labels).float().mean().item()
            print(f"{model_name} fold {ckpt['fold']} (val_acc={ckpt['val_acc']:.4f}) -> test_acc={acc:.4f}")
            results.append([model_name, ckpt["fold"], "single", acc])
            fold_probs.append(probs)

        ensemble_probs = torch.stack(fold_probs).mean(dim=0)
        ensemble_preds = ensemble_probs.argmax(dim=1)
        ensemble_acc = (ensemble_preds == labels).float().mean().item()
        print(f"== {model_name} {len(ckpts)}-fold ENSEMBLE test_acc={ensemble_acc:.4f}")
        results.append([model_name, "all", "ensemble", ensemble_acc])
        report_confusion_and_f1(model_name, labels.numpy(), ensemble_preds.numpy(), Path(args.run_dir))

    out_path = Path(args.run_dir) / "test_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "fold", "kind", "test_acc"])
        writer.writerows(results)
    print(f"Wrote test results to {out_path}")


if __name__ == "__main__":
    main()
