# Ordinal, Lesion-Aware Mayo Score Classification

A separate, more ambitious research pipeline from the ResNet/VGG baselines in
`../src/` — same dataset, different scientific question:

> **Can patch-level lesion-aware ordinal modeling improve discrimination
> between adjacent Mayo Endoscopic Scores (0↔1, 1↔2, 2↔3)?**

And its self-supervised extension (not yet implemented, see "Not in this
pass" below):

> **Can self-supervised patch representations discover latent mucosal
> phenotypes associated with adjacent-MES transitions, without lesion-level
> annotations?**

Model selection throughout is by **validation QWK**, not accuracy.

## Architecture (current)

```
Endoscopic Image
       |
DINOv2 ViT-S/14-reg or EndoViT  (frozen by default)
       |
   ------------------------------
   |                            |
global CLS token           patch tokens
   |                            |
   |                   gated-attention MIL pooling
   |                   (mean/max/topk/attention/gated_attention)
   |                            |
   -------------- concat --------
              |
        Fusion MLP -> h_fused
              |
   ---------------------------
   |                         |
z = severity scalar   ordinal cutpoint head
   |                  logits_k = scale*(z - tau_k), tau_0<tau_1<tau_2
   ------------------------- (monotonicity is structural, not just learned)
              |
      P(Y>0), P(Y>1), P(Y>2) -> MES 0-3
```

No clinical concept annotations required or used — LIMUC doesn't ship them,
so no head in the current model depends on them.

## Ablation table (this pass: A-E)

| Config | Ordinal | MIL | Rank | File |
|---|---|---|---|---|
| A — CE baseline | ✗ | ✗ | ✗ | `configs/baseline_ce.yaml` |
| B — Ordinal | ✓ | ✗ | ✗ | `configs/ordinal.yaml` |
| C — Ordinal + rank | ✓ | ✗ | ✓ | `configs/ordinal_rank.yaml` |
| D — Ordinal + MIL | ✓ | ✓ | ✗ | `configs/ordinal_mil.yaml` |
| **E — Ordinal + MIL + rank (proposed)** | ✓ | ✓ | ✓ | `configs/ordinal_mil_rank.yaml` |
| E, native-resolution variant | ✓ | ✓ | ✓ | `configs/ordinal_mil_rank_native_res.yaml` (294×364, 546 patches vs 224×224's 256) |

**Not in this pass** (deliberately deferred, per the staged-complexity
decision): F (latent mucosal prototype discovery) and G (ordinal contrastive
loss). These get their own configs once A–E are verified against real data —
adding them now would repeat the "five losses from day one" mistake this
design explicitly moved away from. When prototypes are added later, they'll
be reported strictly as **"latent concepts" / "visual phenotypes" / "mucosal
prototypes"** — never named after clinical findings (ulceration, erythema,
etc.) without expert validation.

## Methodological safeguards built into this pass

- **Patient leakage**: every split (train/val, and any k-fold) is followed by
  a hard assertion (`assert_no_patient_leakage`) that patient ids don't
  overlap between partitions — raises, doesn't just warn.
- **Coverage logging**: before training, prints both an image-count and a
  patient-count-per-class table per split (`print_split_coverage`), and
  `validate_coverage` raises if any split is missing a class at the
  *patient* level.
- **Patient-aware balanced sampling**: `PatientAwareBalancedSampler` samples
  class → patient → image, so one prolific patient's many frames can't
  dominate a class's sampling weight the way plain per-image class-balanced
  sampling would.
- **Informative ranking pairs, not all pairs**: `RankingLoss` samples
  adjacent-class pairs (0v1, 1v2, 2v3) most heavily and wider pairs (0v2,
  0v3, 1v3) at a lower configurable rate, and returns the realized
  class-distance distribution every batch — logged to TensorBoard and
  stdout every epoch, so batch composition is verified empirically, not
  assumed.
- **Native-resolution-preserving input**: `ResizeAndPadToMultiple` resizes
  preserving aspect ratio and pads to a patch-size multiple, instead of
  distorting the aspect ratio or upsampling past native detail. The
  224×224 vs native-resolution (294×364) comparison is a first-class
  ablation, not an assumption that bigger is better.
- **Dual threshold reporting**: every experiment reports both the head's own
  default (probability 0.5) decision rule and a separately validation-tuned
  cutpoint set (`tune_thresholds_for_qwk`, fit on validation z only, frozen
  before touching `data/test_set`) — so any performance gain is attributable
  to representation learning vs. threshold tuning, not conflated.
- **Patient-cluster bootstrap**: `patient_cluster_bootstrap` is the primary
  uncertainty estimate (resamples patients with replacement, pools their
  images, computes the metric) rather than a naive per-image bootstrap that
  would treat correlated same-patient images as independent.
- **z is never compared raw across independently-trained models** — only
  within one model (ranking, distribution plots, Spearman correlation, which
  is scale-invariant). Cross-model z comparison would need standardization
  against that model's own training-set mean/std first (not yet needed since
  no cross-model z visualization exists in this pass).

## Data

Two modes, set via `data.mode` in the config:

- `"directory"` (default): points at `../data/train_and_validation_sets` and
  `../data/test_set` (same LIMUC data as the rest of this repo — same Mayo-
  class-folder / patient-folder inference logic).
- `"csv"`: a CSV with columns `image_path,mayo_score,patient_id[,split]`
  (rows with `split=="test"` go to `evaluate.py`, everything else is
  train/val-split by `train.py`).

## Usage

```bash
pip install -r requirements.txt

# Train one ablation config
python train.py --config configs/ordinal_mil_rank.yaml

# Evaluate its best checkpoint on the held-out test set (run once)
python evaluate.py --checkpoint runs_ordinal_mil/E_ordinal_mil_rank/best.pt
```

`train.py` prints the split coverage tables and ranking-pair distance
distribution every epoch, does early stopping on validation QWK (patience
15), and writes `val_metrics.json` (default + tuned thresholds) alongside
the checkpoint. `evaluate.py` writes `test_predictions.csv` (patient_id,
image_path, true_mayo, predicted_mayo, severity_z, prob_gt0/1/2) and
`test_metrics.json` (full metric suite + patient-cluster bootstrap 95% CIs)
next to the checkpoint it was given.

## Requires Python 3.9+ (3.10 recommended) — same as the rest of this repo.

## Not yet implemented (next passes)

- `run_ablation.py` to sweep all configs into one `ablation_results.csv`.
- `statistical_analysis.py` (paired bootstrap between models, QWK CIs,
  McNemar test) — will consume the `test_predictions.csv` files this pass
  already produces.
- Visualization suite (attention overlays, top-attended patches, z
  distribution plots, embedding UMAP/t-SNE, correct/incorrect/severe-error
  galleries, artifact-sensitivity masking test).
- Latent prototype discovery module (F) and ordinal contrastive loss (G).
- k-fold confirmation of the winning single-holdout config
  (`patient_group_kfold` already exists in `src/datasets/limuc.py` for this).
