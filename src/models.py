"""Model factory: ResNet-50, ResNet-152, VGG-19, trained from scratch (no ImageNet weights)."""
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4

_BUILDERS = {
    "resnet50": models.resnet50,
    "resnet152": models.resnet152,
    "vgg19": models.vgg19,
}


def build_model(name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(_BUILDERS)}.")
    model = _BUILDERS[name](weights=None)  # scratch init, no pretraining/finetuning
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # vgg19
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
