"""LIMUC data loading, patient-level group-stratified splitting, and a
patient-aware class-balanced sampler.

Two input modes:
  - CSV with columns: image_path, mayo_score, patient_id [, split]
  - Directory tree: infers Mayo class from a path component matching
    "Mayo 0".."Mayo 3" (case-insensitive) and patient id from a numeric
    folder name, falling back to leading digits in the filename.

Every split function is followed by a hard leakage assertion -- this is not
optional and not just logged, it raises.
"""
from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T
from torchvision.transforms import functional as TF

IMG_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAYO_CLASS_RE = re.compile(r"mayo[ _-]?([0-3])\b", re.IGNORECASE)
LEADING_DIGITS_RE = re.compile(r"^(\d+)")
NUM_CLASSES = 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LeakageError(RuntimeError):
    """Raised when a patient id appears in more than one data split."""


@dataclass
class LimucRecord:
    path: str
    patient_id: str
    mayo: int
    split_hint: Optional[str] = None  # e.g. "train_and_validation_sets" / "test_set"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_from_csv(csv_path: str | Path) -> List[LimucRecord]:
    records = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            records.append(LimucRecord(
                path=row["image_path"],
                patient_id=str(row["patient_id"]),
                mayo=int(row["mayo_score"]),
                split_hint=row.get("split"),
            ))
    if not records:
        raise RuntimeError(f"No rows found in {csv_path}")
    return records


def _infer_label(path: Path) -> int:
    for part in path.parts:
        m = MAYO_CLASS_RE.search(part)
        if m:
            return int(m.group(1))
    raise ValueError(f"Could not infer Mayo class (0-3) from path: {path}")


def _infer_patient_id(path: Path, root: Path) -> str:
    rel_parts = path.relative_to(root).parts
    for part in rel_parts[:-1]:
        if part.isdigit():
            return part
    m = LEADING_DIGITS_RE.match(path.stem)
    return m.group(1) if m else path.stem


def load_from_directory(root: str | Path, split_hint: Optional[str] = None) -> List[LimucRecord]:
    root = Path(root)
    records = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        records.append(LimucRecord(
            path=str(path),
            patient_id=_infer_patient_id(path, root),
            mayo=_infer_label(path),
            split_hint=split_hint or root.name,
        ))
    if not records:
        raise RuntimeError(f"No images found under {root}")
    return records


# --------------------------------------------------------------------------
# Patient-level group-stratified splitting, with a hard leakage check
# --------------------------------------------------------------------------

def assert_no_patient_leakage(*record_lists: Sequence[LimucRecord]) -> None:
    id_sets = [set(r.patient_id for r in records) for records in record_lists]
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            overlap = id_sets[i] & id_sets[j]
            if overlap:
                raise LeakageError(
                    f"{len(overlap)} patient id(s) appear in more than one split: "
                    f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
                )


def patient_group_holdout_split(
    records: List[LimucRecord], val_fraction: float = 0.15, seed: int = 42,
) -> Tuple[List[LimucRecord], List[LimucRecord]]:
    """Single patient-disjoint, class-stratified train/val holdout split.

    Implemented via StratifiedGroupKFold with n_splits = round(1/val_fraction)
    and taking one fold as validation, rather than a bespoke splitter --
    reuses sklearn's tested group-stratification logic.
    """
    n_splits = max(2, round(1 / val_fraction))
    labels = [r.mayo for r in records]
    groups = [r.patient_id for r in records]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(records, labels, groups))
    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    assert_no_patient_leakage(train, val)
    return train, val


def patient_group_kfold(
    records: List[LimucRecord], n_splits: int = 5, seed: int = 42,
) -> Iterator[Tuple[List[LimucRecord], List[LimucRecord]]]:
    """Repeated patient-level group-stratified CV, for confirming a winning
    config after the single-holdout ablation stage."""
    labels = [r.mayo for r in records]
    groups = [r.patient_id for r in records]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in sgkf.split(records, labels, groups):
        train = [records[i] for i in train_idx]
        val = [records[i] for i in val_idx]
        assert_no_patient_leakage(train, val)
        yield train, val


# --------------------------------------------------------------------------
# Coverage logging (image-level AND patient-level, per split, per class)
# --------------------------------------------------------------------------

def coverage_tables(splits: Dict[str, List[LimucRecord]]) -> Tuple[str, str]:
    """Returns two formatted tables: image counts and patient-containing-class
    counts, per split, per Mayo class."""
    class_cols = [f"MES{c}" for c in range(NUM_CLASSES)]

    header1 = f"{'split':<24}{'patients':>10}{'images':>10}" + "".join(f"{c:>8}" for c in class_cols)
    lines1 = [header1]
    header2 = f"{'split':<24}" + "".join(f"pts_w_{c:<5}" for c in class_cols)
    lines2 = [header2]

    for name, records in splits.items():
        patients = set(r.patient_id for r in records)
        img_counts = [sum(1 for r in records if r.mayo == c) for c in range(NUM_CLASSES)]
        lines1.append(
            f"{name:<24}{len(patients):>10}{len(records):>10}" + "".join(f"{n:>8}" for n in img_counts)
        )
        patients_with_class = [
            len({r.patient_id for r in records if r.mayo == c}) for c in range(NUM_CLASSES)
        ]
        lines2.append(f"{name:<24}" + "".join(f"{n:>11}" for n in patients_with_class))

    return "\n".join(lines1), "\n".join(lines2)


def print_split_coverage(splits: Dict[str, List[LimucRecord]]) -> None:
    img_table, patient_table = coverage_tables(splits)
    print("\n=== Split coverage (images) ===")
    print(img_table)
    print("\n=== Split coverage (patients containing each class) ===")
    print(patient_table)
    print()


def validate_coverage(splits: Dict[str, List[LimucRecord]], min_patients_per_class: int = 1) -> None:
    """Raises if any split is missing a Mayo class at the patient level --
    a silently-empty grade in a split breaks stratified metrics downstream."""
    problems = []
    for name, records in splits.items():
        for c in range(NUM_CLASSES):
            n_patients = len({r.patient_id for r in records if r.mayo == c})
            if n_patients < min_patients_per_class:
                problems.append(f"{name}: MES{c} has only {n_patients} patient(s)")
    if problems:
        raise RuntimeError(
            "Split coverage validation failed -- regenerate the split (different "
            "seed / fewer folds) or lower min_patients_per_class:\n  " + "\n  ".join(problems)
        )


# --------------------------------------------------------------------------
# Transform: resize preserving aspect ratio, pad to a patch_size multiple
# --------------------------------------------------------------------------

class ResizeAndPadToMultiple:
    """Resizes so the image fits within (target_h, target_w) preserving aspect
    ratio, then center-pads with the given fill to exactly (target_h, target_w).

    This avoids both the distortion of a naive non-aspect-preserving resize
    and the false-detail of upsampling well past native resolution -- e.g.
    LIMUC's native ~352x288 mapped to a 364x294 (26x21 patch, patch=14) box
    instead of stretched/upsampled to 224x224 or 518x518.
    """

    def __init__(self, target_hw: Tuple[int, int], fill: Tuple[int, int, int] = (0, 0, 0)):
        self.target_h, self.target_w = target_hw
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = min(self.target_w / w, self.target_h / h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = TF.resize(img, [new_h, new_w])
        pad_w, pad_h = self.target_w - new_w, self.target_h - new_h
        left, top = pad_w // 2, pad_h // 2
        right, bottom = pad_w - left, pad_h - top
        return TF.pad(img, [left, top, right, bottom], fill=self.fill)


def build_transform(image_size: Tuple[int, int], train: bool) -> T.Compose:
    ops = [ResizeAndPadToMultiple(tuple(image_size))]
    if train:
        ops += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class LimucMILDataset(Dataset):
    def __init__(self, records: List[LimucRecord], transform: T.Compose):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        image = Image.open(r.path).convert("RGB")
        image = self.transform(image)
        return image, r.mayo, r.patient_id, r.path


# --------------------------------------------------------------------------
# Patient-aware balanced sampler
#
# Plain per-image class-balanced sampling lets one prolific patient's many
# frames absorb most of a rare class's sampling weight. This instead samples
# class -> patient -> image, so a class's weight is spread across patients
# who have that class, not across that class's raw image count.
# --------------------------------------------------------------------------

class PatientAwareBalancedSampler(Sampler[int]):
    def __init__(self, records: List[LimucRecord], num_samples: Optional[int] = None, seed: int = 42):
        self.records = records
        self.num_samples = num_samples or len(records)
        self.rng = random.Random(seed)

        self.class_to_patients: Dict[int, List[str]] = defaultdict(list)
        self.patient_class_to_indices: Dict[Tuple[str, int], List[int]] = defaultdict(list)
        for idx, r in enumerate(records):
            self.patient_class_to_indices[(r.patient_id, r.mayo)].append(idx)
        for (patient_id, mayo), idxs in self.patient_class_to_indices.items():
            if patient_id not in self.class_to_patients[mayo]:
                self.class_to_patients[mayo].append(patient_id)

        self.classes = [c for c in range(NUM_CLASSES) if self.class_to_patients.get(c)]
        counts = np.array([sum(1 for r in records if r.mayo == c) for c in self.classes], dtype=np.float64)
        inv_freq = 1.0 / np.maximum(counts, 1)
        self.class_probs = (inv_freq / inv_freq.sum()).tolist()

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples):
            c = self.rng.choices(self.classes, weights=self.class_probs, k=1)[0]
            patient_id = self.rng.choice(self.class_to_patients[c])
            idx = self.rng.choice(self.patient_class_to_indices[(patient_id, c)])
            yield idx
