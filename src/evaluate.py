"""Evaluate trained LIMUC checkpoints on the held-out test set (data/test_set).

This is the only script that reads data/test_set -- it must never be used
during training or fold-validation, per the LIMUC evaluation protocol.
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import LimucDataset, index_images
from models import build_model
from utils import get_device

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        ensemble_acc = (ensemble_probs.argmax(dim=1) == labels).float().mean().item()
        print(f"== {model_name} {len(ckpts)}-fold ENSEMBLE test_acc={ensemble_acc:.4f}")
        results.append([model_name, "all", "ensemble", ensemble_acc])

    out_path = Path(args.run_dir) / "test_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "fold", "kind", "test_acc"])
        writer.writerows(results)
    print(f"Wrote test results to {out_path}")


if __name__ == "__main__":
    main()
