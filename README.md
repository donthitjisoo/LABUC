# LABUC — LIMUC Ulcerative Colitis Severity Classification

Training and evaluation pipeline for the [LIMUC dataset](https://zenodo.org/records/5827695)
(Labeled Images for Ulcerative Colitis), classifying the 4-class Mayo endoscopic
score (0–3) from colonoscopy images. Includes:

- ResNet-50 / ResNet-152 / VGG-19 trained **from scratch** (no ImageNet weights)
  with patient-grouped k-fold cross-validation.
- Held-out test-set evaluation with confusion matrix + per-class precision/recall/F1.
- Dimensionality-reduction visualization of the dataset (t-SNE, UMAP, PaCMAP,
  TriMap, PHATE) using raw pixels, your own trained backbones, or pretrained
  foundation models (DINOv3, MedSigLIP, UNI2-h).

## Repository layout

```
data/
  train_and_validation_sets/   # training pool -- used for k-fold CV, never for final test numbers
  test_set/                    # held out; only evaluate.py reads this
src/
  dataset.py                # image indexing + patient-grouped StratifiedGroupKFold splitting
  models.py                 # resnet50 / resnet152 / vgg19 factory (weights=None, from scratch)
  utils.py                  # seeding, device selection, running-average helper
  train.py                  # k-fold CV training entry point
  evaluate.py                # held-out test-set evaluation (accuracy, confusion matrix, F1)
  embed.py                   # extract feature vectors (raw / trained backbone / foundation model)
  foundation_models.py       # DINOv3, MedSigLIP, UNI2-h, (stub) EndoDINO loaders
  visualize_embeddings.py    # t-SNE / UMAP / PaCMAP / TriMap / PHATE plots from embed.py output
runs/                        # all training/evaluation/embedding outputs land here
```

## 1. Setup

Requires Python 3.9+ (3.10 recommended).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data layout

Populate these two folders (copy the LIMUC data in yourself):

```
data/train_and_validation_sets/   # e.g. <patient_id>/Mayo 0/*.bmp, or Mayo 0/<patient_id>/*.bmp
data/test_set/                    # same layout
```

`src/dataset.py` scans recursively and infers, per image:
- **label** from any path component matching `Mayo 0`..`Mayo 3` (case-insensitive)
- **patient id** from a numeric folder name in the path, falling back to leading
  digits in the filename

so either LIMUC folder convention (patient-folder-then-class or
class-then-patient-folder) works without reorganizing anything.

**`data/test_set` is only ever read by `evaluate.py`.** Training and
cross-validation only touch `data/train_and_validation_sets`, so the test set
is never used as a validation set.

## 3. Train (k-fold CV, from scratch)

```bash
python src/train.py --folds 5 --epochs 100
```

- Trains resnet50, resnet152, and vgg19 by default (`--models resnet50` to do just one).
- `--folds 3` for 3-fold instead of 5-fold.
- Folds are patient-disjoint and stratified by Mayo class (`StratifiedGroupKFold`),
  so no patient's images leak across the train/validation split of a fold.
- Loss is class-weighted by default (LIMUC is imbalanced: ~54% Mayo 0 vs ~8%
  Mayo 3) — disable with `--no-class-weighted`.
- Useful knobs: `--batch-size`, `--lr`, `--img-size`, `--seed`.

Outputs per model/fold:
```
runs/<model>/fold_<k>/history.csv   # per-epoch train/val loss & accuracy
runs/<model>/fold_<k>/best.pt       # checkpoint with the best validation accuracy
runs/cv_summary.csv                 # best val accuracy per model per fold
```

## 4. Evaluate on the held-out test set

Run once, after training completes:

```bash
python src/evaluate.py
```

For each model, this loads every fold's `best.pt`, reports per-fold test
accuracy plus a fold-averaged **ensemble** accuracy, and writes:

```
runs/test_results.csv                        # accuracy per fold + ensemble, per model
runs/<model>_confusion_matrix.csv             # rows = predicted Mayo class, cols = true class
runs/<model>_class_metrics.csv                # precision / recall / F1 / support per class
```

The confusion matrix and per-class metrics are computed on the ensemble
predictions (averaged softmax across folds).

## 5. Visualize the dataset with dimensionality reduction

Two-step pipeline: extract a feature vector per image, then project to 2D
with multiple methods.

### 5a. Extract features (`src/embed.py`)

Three interchangeable modes:

| Mode | What it uses | Needs training first? |
|---|---|---|
| `raw` (default) | resized + flattened grayscale pixels | No |
| `backbone` | your own trained checkpoint's penultimate layer | Yes |
| `foundation` | a pretrained foundation model | No (but see below) |

```bash
# Raw pixels -- works immediately
python src/embed.py --features-mode raw \
    --data-roots data/train_and_validation_sets data/test_set

# Your own trained model's learned features
python src/embed.py --features-mode backbone --model resnet50 \
    --checkpoint runs/resnet50/fold_1/best.pt \
    --data-roots data/train_and_validation_sets data/test_set

# A pretrained foundation model
python src/embed.py --features-mode foundation --foundation-model dinov3_vitl16 \
    --data-roots data/train_and_validation_sets data/test_set
```

Each run writes a `.npz` (features + Mayo labels + patient ids + split) to
`runs/embeddings/<name>_features.npz`.

**Foundation models** (`--foundation-model {dinov3_vitl16, medsiglip_448, uni2_h, endovit}`):

| Model | Gated? | Notes |
|---|---|---|
| [DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | Yes | accept license on the model page |
| [MedSigLIP-448](https://huggingface.co/google/medsiglip-448) | Yes | approval is near-instant |
| [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) | Yes | account email must match an institutional address; approval isn't instant |
| [EndoViT](https://huggingface.co/egeozsoy/EndoViT) | **No** | Apache 2.0, public, no login needed |

For the three gated ones:
1. Log into Hugging Face with your own account and accept the license /
   access request on the model's page.
2. Authenticate locally: `huggingface-cli login`, or set `export HF_TOKEN=hf_...`.
3. First run downloads the weights (DINOv3-L ≈1.2GB, MedSigLIP ≈1.6GB, UNI2-h ≈2.7GB,
   EndoViT ≈330MB).

EndoViT needs none of that — `--foundation-model endovit` just works once
`requirements.txt` is installed.

`endodino` is registered as a stub only — EndoDINO (arXiv:2501.05488) has no
public checkpoint release. Passing `--foundation-model endodino
--endodino-checkpoint <path>` will only work once you've filled in its
loading logic in `foundation_models.py` to match weights you've obtained
directly from the authors.

### 5b. Plot (`src/visualize_embeddings.py`)

```bash
python src/visualize_embeddings.py --features runs/embeddings/raw_features.npz \
    --output-dir runs/embeddings/dr_plots_raw
```

Runs t-SNE, UMAP, PaCMAP, TriMap, and PHATE (`--methods` to pick a subset),
saving each as its own PNG (`tsne.png`, `umap.png`, ...) into `--output-dir`
as soon as it finishes, colored by Mayo class. Raw 2D coordinates for every
method are also saved together in `coords.npz` in that folder.

`--max-points` (default 5000) subsamples before running t-SNE/TriMap/PHATE,
since those scale poorly to tens of thousands of points.

## Results

Full `runs/` output (checkpoints, history CSVs, confusion matrices, embedding
plots) is uploaded here rather than committed to the repo:
[Google Drive folder](https://drive.google.com/drive/folders/1k_55COozL5Racaz8ijJPlHuSvNY-d-p_?usp=drive_link)

## Notes

- All three models are trained **from scratch** — `weights=None` in
  `models.py`, no ImageNet pretraining or fine-tuning.
- Device is auto-detected (CUDA → MPS → CPU) in `utils.get_device()`.
- Mayo classes are 0–3 (not 1–4) throughout.
