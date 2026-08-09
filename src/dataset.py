"""Indexing and patient-grouped k-fold splitting for the LIMUC dataset.

Handles both LIMUC folder layouts (patient folders containing Mayo-class
subfolders, or Mayo-class folders containing patient folders/filenames) by
scanning recursively and inferring the label and patient id from path parts.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

IMG_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAYO_CLASS_RE = re.compile(r"mayo[ _-]?([0-3])\b", re.IGNORECASE)
LEADING_DIGITS_RE = re.compile(r"^(\d+)")


@dataclass
class Sample:
    path: Path
    label: int
    patient_id: str


def _infer_label(path: Path) -> int:
    for part in path.parts:
        m = MAYO_CLASS_RE.search(part)
        if m:
            return int(m.group(1))
    raise ValueError(
        f"Could not infer Mayo class (0-3) from path: {path}. "
        f"Expected a folder named like 'Mayo 0'..'Mayo 3' somewhere in the tree."
    )


def _infer_patient_id(path: Path, root: Path) -> str:
    rel_parts = path.relative_to(root).parts
    for part in rel_parts[:-1]:
        if part.isdigit():
            return part
    m = LEADING_DIGITS_RE.match(path.stem)
    if m:
        return m.group(1)
    return path.stem


def index_images(root: str | Path) -> list[Sample]:
    root = Path(root)
    samples = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        samples.append(Sample(path=path, label=_infer_label(path), patient_id=_infer_patient_id(path, root)))
    if not samples:
        raise RuntimeError(
            f"No images found under {root}. Populate it with the LIMUC "
            f"train_and_validation_sets data before running training."
        )
    return samples


def make_folds(samples: list[Sample], n_folds: int, seed: int = 42):
    """Yield (train_samples, val_samples) per fold. Patient-disjoint, stratified by label.

    Only ever called on the training pool -- the held-out test set must not
    pass through here.
    """
    labels = [s.label for s in samples]
    groups = [s.patient_id for s in samples]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in sgkf.split(samples, labels, groups):
        yield [samples[i] for i in train_idx], [samples[i] for i in val_idx]


class LimucDataset(Dataset):
    def __init__(self, samples: list[Sample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label
