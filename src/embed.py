"""Extract per-image feature vectors for dimensionality-reduction visualization.

Three modes:
  --features-mode raw         resize + flatten raw pixels (no trained model needed)
  --features-mode backbone    forward pass through a trained checkpoint's backbone,
                               i.e. everything up to (but excluding) the final
                               classification layer
  --features-mode foundation  forward pass through a pretrained foundation model
                               (DINOv3 ViT-L/16, MedSigLIP-448, UNI2-h, or EndoDINO
                               if you have private weights) -- see foundation_models.py
                               for the required Hugging Face access/login steps.

Output is a single .npz with features, labels (Mayo class), patient_ids, and
source split, ready for src/visualize_embeddings.py.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import LimucDataset, index_images
from foundation_models import REGISTRY as FOUNDATION_REGISTRY
from models import build_model
from utils import get_device

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_backbone_extractor(model: nn.Module, model_name: str):
    """Return a callable(x) -> flattened feature tensor, using everything up
    to (excluding) the model's classification head."""
    if model_name.startswith("resnet"):
        trunk = nn.Sequential(*list(model.children())[:-1])  # drop fc

        def forward(x):
            return torch.flatten(trunk(x), 1)

        return forward
    else:  # vgg19
        features, avgpool = model.features, model.avgpool
        head = model.classifier[:-1]  # drop final Linear(4096 -> num_classes)

        def forward(x):
            x = torch.flatten(avgpool(features(x)), 1)
            return head(x)

        return forward


@torch.no_grad()
def extract_backbone_features(samples, checkpoint_path, model_name, img_size,
                               batch_size, num_workers, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    extractor = build_backbone_extractor(model, model_name)

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    loader = DataLoader(LimucDataset(samples, transform=tf), batch_size=batch_size,
                         shuffle=False, num_workers=num_workers)
    feats = []
    for images, _ in loader:
        feats.append(extractor(images.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


class _FoundationDataset(LimucDataset):
    """Applies a foundation extractor's own preprocessing instead of a fixed transform."""

    def __init__(self, samples, extractor):
        super().__init__(samples, transform=None)
        self.extractor = extractor

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        return self.extractor.preprocess(image), sample.label


def extract_foundation_features(samples, model_name, batch_size, num_workers, device,
                                 endodino_checkpoint=None):
    if model_name not in FOUNDATION_REGISTRY:
        raise ValueError(f"Unknown foundation model '{model_name}'. Choose from {list(FOUNDATION_REGISTRY)}.")
    extractor_cls = FOUNDATION_REGISTRY[model_name]
    extractor = (extractor_cls(device, checkpoint_path=endodino_checkpoint)
                 if model_name == "endodino" else extractor_cls(device))

    loader = DataLoader(_FoundationDataset(samples, extractor), batch_size=batch_size,
                         shuffle=False, num_workers=num_workers)
    feats = []
    for images, _ in loader:
        feats.append(extractor.embed(images).cpu().numpy())
    return np.concatenate(feats, axis=0)


def extract_raw_features(samples, img_size, batch_size, num_workers):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    loader = DataLoader(LimucDataset(samples, transform=tf), batch_size=batch_size,
                         shuffle=False, num_workers=num_workers)
    feats = []
    for images, _ in loader:
        feats.append(images.flatten(1).numpy())
    return np.concatenate(feats, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Extract feature vectors for DR visualization.")
    parser.add_argument("--data-roots", nargs="+", default=["data/train_and_validation_sets"],
                         help="One or more image roots to embed (e.g. add data/test_set too).")
    parser.add_argument("--features-mode", choices=["raw", "backbone", "foundation"], default="raw",
                         help="'raw' needs no trained model; 'backbone' uses a trained checkpoint; "
                              "'foundation' uses a pretrained foundation model (see foundation_models.py).")
    parser.add_argument("--model", default="resnet50", choices=["resnet50", "resnet152", "vgg19"],
                         help="Only used with --features-mode backbone.")
    parser.add_argument("--checkpoint", default=None,
                         help="Path to a trained fold checkpoint (best.pt). "
                              "Defaults to runs/<model>/fold_1/best.pt.")
    parser.add_argument("--foundation-model", default="dinov3_vitl16",
                         choices=list(FOUNDATION_REGISTRY.keys()),
                         help="Only used with --features-mode foundation. Requires accepting the "
                              "model's license on Hugging Face and being logged in (huggingface-cli "
                              "login or HF_TOKEN env var) -- see foundation_models.py.")
    parser.add_argument("--endodino-checkpoint", default=None,
                         help="Path to private EndoDINO weights, if you have them (no public release exists).")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--raw-size", type=int, default=64,
                         help="Resize used only for --features-mode raw (kept small to bound dimensionality).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    all_samples, splits = [], []
    for root in args.data_roots:
        root_samples = index_images(root)
        all_samples.extend(root_samples)
        splits.extend([Path(root).name] * len(root_samples))

    labels = np.array([s.label for s in all_samples])
    patient_ids = np.array([s.patient_id for s in all_samples])
    splits = np.array(splits)

    if args.features_mode == "raw":
        features = extract_raw_features(all_samples, args.raw_size, args.batch_size, args.num_workers)
        default_out = "runs/embeddings/raw_features.npz"
    elif args.features_mode == "backbone":
        device = get_device()
        checkpoint = args.checkpoint or f"runs/{args.model}/fold_1/best.pt"
        features = extract_backbone_features(all_samples, checkpoint, args.model, args.img_size,
                                              args.batch_size, args.num_workers, device)
        default_out = f"runs/embeddings/{args.model}_features.npz"
    else:  # foundation
        device = get_device()
        features = extract_foundation_features(all_samples, args.foundation_model, args.batch_size,
                                                args.num_workers, device,
                                                endodino_checkpoint=args.endodino_checkpoint)
        default_out = f"runs/embeddings/{args.foundation_model}_features.npz"

    output_path = Path(args.output or default_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, features=features, labels=labels, patient_ids=patient_ids, splits=splits)
    print(f"Saved {features.shape[0]} feature vectors (dim={features.shape[1]}) to {output_path}")


if __name__ == "__main__":
    main()
