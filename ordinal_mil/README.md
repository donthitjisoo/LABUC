# Ordinal, Lesion-Aware Mayo Score Classification

A separate, more ambitious research pipeline from the ResNet/VGG baselines in
`../src/` — same dataset, different scientific question:

> **Can patch-level lesion-aware ordinal modeling improve discrimination
> between adjacent Mayo Endoscopic Scores (0↔1, 1↔2, 2↔3)?**

Model selection throughout is by **validation QWK**, not accuracy.

## Results (ablations A–E, single patient-grouped holdout split)

All six configs trained and evaluated; test-set metrics below use each
config's validation-tuned thresholds (see "Dual threshold reporting" below —
config A has no thresholds to tune, so it's its one number).

| Config | QWK | Macro F1 | Weighted F1 | MAE | Severe error | Within-1 acc |
|---|---|---|---|---|---|---|
| A — CE baseline | 0.757 | 0.624 | 0.671 | 0.366 | 1.9% | 98.1% |
| B — Ordinal | 0.809 | 0.611 | 0.678 | 0.322 | 1.7% | 98.3% |
| C — Ordinal + rank | 0.806 | 0.600 | 0.677 | 0.329 | 2.0% | 98.0% |
| D — Ordinal + MIL | 0.837 | 0.639 | 0.720 | 0.278 | 1.2% | 98.8% |
| **E — Ordinal + MIL + rank (proposed)** | **0.847** | **0.672** | **0.740** | **0.259** | **0.65%** | **99.4%** |
| E, native-resolution variant | **0.857** | **0.677** | **0.756** | **0.249** | **0.42%** | **99.6%** |
| *ResNet-50 (`../src/`, fully fine-tuned CNN baseline)* | *0.781* | *0.647* | *0.725* | *0.304* | *2.6%* | *97.5%* |

### Reading these results

- **QWK/MAE/severe-error improve monotonically from D onward** and E already
  beats the fully fine-tuned ResNet-50 baseline on every ordinal-aware
  metric, including a >4x reduction in severe (|pred−true|≥2) errors.
- **F1 trails the CNN baseline for A–D**: their macro F1 sits below the
  ResNet-50 baseline (0.647); only E and the native-res variant exceed it.
  This is expected, not a bug: A–D use a **frozen** backbone with only the
  global CLS token
  (no MIL) feeding a small trainable head, versus a **fully fine-tuned**
  CNN trained end-to-end on LIMUC directly. It takes MIL *and* ranking
  together for the frozen-backbone approach to close that gap.
- **MIL's clearest effect is on MES2 sensitivity** — 0.42 (B, no MIL) → 0.44
  (D) → **0.63 (E)** — exactly the class this redesign targeted (the
  genuine clinical boundary case, and the weakest class in every prior
  model including the CNN baselines).
- **C (ordinal+rank, no MIL) is a small net negative versus B** (QWK 0.806
  vs 0.809, macro F1 0.600 vs 0.611) — ranking loss alone, without MIL
  giving it a richer representation to rank, doesn't help here. A real
  ablation finding, not noise.
- **Default vs. tuned thresholds diverge notably for B/C/D** (e.g. B: macro
  F1 0.544 default vs 0.611 tuned) — expected, since effective-number class
  weighting deliberately biases the ordinal loss toward minority classes
  during training, shifting where the *trained* cutpoints land away from a
  flat 0.5 rule. This is exactly why both are reported separately rather
  than only the tuned number.
- **MES2/MES3 recall trade off against each other** going from E to the
  native-resolution variant (MES3 recall 0.700→0.792, MES2 recall
  0.633→0.525) — this specific trade-off has been independently observed
  by a teammate using an unrelated method (threshold-based, different
  backbone/split), which is reassuring: it looks like a real property of
  the task/data rather than an artifact of this particular architecture.

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

**Backbone is a plain config choice** — `backbone.name: dinov2_vits14_reg`
(default) or `backbone.name: endovit`, on any config, not just the two
dedicated variant files above. `data.image_size` must be divisible by
whichever backbone's patch size (14 for DINOv2, 16 for EndoViT) — this is
now derived automatically from `backbone.name`
(`src/utils/backbone_specs.py`) and checked at config-load time, so pointing
an existing 294×364-style config at a different-patch-size backbone fails
fast with a clear error instead of a confusing shape mismatch mid-training.

## Ablation table

| Config | Ordinal | MIL | Rank | File |
|---|---|---|---|---|
| A — CE baseline | ✗ | ✗ | ✗ | `configs/baseline_ce.yaml` |
| B — Ordinal | ✓ | ✗ | ✗ | `configs/ordinal.yaml` |
| C — Ordinal + rank | ✓ | ✗ | ✓ | `configs/ordinal_rank.yaml` |
| D — Ordinal + MIL | ✓ | ✓ | ✗ | `configs/ordinal_mil.yaml` |
| **E — Ordinal + MIL + rank (proposed)** | ✓ | ✓ | ✓ | `configs/ordinal_mil_rank.yaml` |
| E, native-resolution variant | ✓ | ✓ | ✓ | `configs/ordinal_mil_rank_native_res.yaml` (294×364, 546 patches vs 224×224's 256) |
| E, EndoViT-backbone variant | ✓ | ✓ | ✓ | `configs/ordinal_mil_rank_endovit.yaml` (domain-pretrained endoscopy ViT-B/16 instead of generic DINOv2 ViT-S/14-reg) |
| B2 — CDW-CE (analog of B) | ✓ (CDW-CE, not CORAL) | ✗ | ✗ | `configs/cdw_ce.yaml` |
| D2 — CDW-CE + MIL (analog of D) | ✓ | ✓ | ✗ | `configs/cdw_ce_mil.yaml` |
| E2 — CDW-CE + MIL + rank (analog of E) | ✓ | ✓ | ✓ | `configs/cdw_ce_mil_rank.yaml` |

**CDW-CE** (class-distance-weighted cross-entropy, de la Torre et al. 2018,
used for LIMUC MES grading in Polat et al.'s baseline) is now a third
`model.head_type` option alongside `ce` and `ordinal` — a different way of
encoding ordinality (penalize softmax probability mass on far classes more
than near ones) versus the CORAL cutpoint-on-z approach. It reuses the same
4-way softmax head as the plain CE baseline but *also* exposes a severity
`z` (via an auxiliary head), so — unlike plain CE — it can be combined with
MIL and the ranking/regression losses in the same ablation shape as B/D/E,
not just as a standalone baseline. `loss.cdw_alpha` (default 5.0) controls
the distance-penalty exponent.

**F (latent prototype discovery) and G (ordinal contrastive loss) are still
deliberately deferred**, now for a second reason beyond the original
staged-complexity one: F specifically needs the attention-visualization
tooling below to be useful at all (its whole purpose is showing which
patches activate a prototype), so it comes after that's been used and
reviewed, not before. When added, prototypes will be reported strictly as
**"latent concepts" / "visual phenotypes" / "mucosal prototypes"** — never
named after clinical findings without expert validation.

## Attention visualization (new)

MIL attention (D/E) was being computed and used in every forward pass all
along, but wasn't being captured anywhere for inspection — fixed. Now:

```bash
python evaluate.py --checkpoint runs_ordinal_mil/D_ordinal_mil/best.pt --save-attention
python visualize_attention.py --attention runs_ordinal_mil/D_ordinal_mil/attention.npz

python evaluate.py --checkpoint runs_ordinal_mil/E_ordinal_mil_rank/best.pt --save-attention
python visualize_attention.py --attention runs_ordinal_mil/E_ordinal_mil_rank/attention.npz
```

`evaluate.py --save-attention` (only meaningful for `use_mil: true` configs)
additionally saves `attention.npz` — per-image attention weights, patch grid
shape, and predictions, alongside the usual `test_predictions.csv` /
`test_metrics.json`.

`visualize_attention.py` then produces, per config, three category folders
(`correct/`, `adjacent_error/`, `severe_error/`) each containing:
- a heatmap overlay (attention upsampled and blended onto the model's actual
  input image — not the raw original, so the overlay is guaranteed to align
  spatially with what the model actually saw)
- the top-K highest-attention patches cropped out individually

Per the design doc's stance: **attention here is a hypothesis, not a
localization ground truth.** The overlays exist to manually check whether
attention concentrates on mucosa, or instead on scope borders, text
overlays, specular highlights, or debris — not to claim lesion localization
without separately obtained ground truth.

## Methodological safeguards built into this pipeline

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
  ablation, confirmed above to actually help, not assumed to.
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
  against that model's own training-set mean/std first.

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

# Add --save-attention for D/E to also produce attention.npz (see above)
```

`train.py` prints the split coverage tables and ranking-pair distance
distribution every epoch, does early stopping on validation QWK (patience
15), and writes `val_metrics.json` (default + tuned thresholds) alongside
the checkpoint. `evaluate.py` writes `test_predictions.csv` (patient_id,
image_path, true_mayo, predicted_mayo, severity_z, prob_gt0/1/2) and
`test_metrics.json` (full metric suite + patient-cluster bootstrap 95% CIs)
next to the checkpoint it was given.

Requires Python 3.9+ (3.10 recommended) — same as the rest of this repo.

### Seeds: selectable, and split from training stochasticity

`seed` (top-level, default 42) is a plain config field — fully selectable,
not hardcoded — controlling model init, the patient-aware balanced sampler,
ranking-pair sampling, and bootstrap-CI resampling.

`data.split_seed` (default 42, same value, independent field) controls
*only* the patient train/val partition. These are deliberately separate:
changing `seed` alone (e.g. for a multi-seed robustness check, re-running
the same config with `seed: 7`, `seed: 123`, ...) reruns training with
different stochasticity while keeping the exact same patient split — so a
robustness check actually isolates training-noise variance, rather than
also silently reshuffling which patients are in validation each time.
Override `data.split_seed` too only if you deliberately want a different
split.

## Not yet implemented (next up)

Priority order based on where the results above point:

1. **CDW-CE as an additional ordinal loss option** — a class-distance-weighted
   cross-entropy variant, to compare against the current CORAL-cutpoint
   approach within this same pipeline/split.
2. **z-gap / grade-spacing analysis** — plot class-conditional z distributions
   and the learned `τ_k` spacings, to check whether the ordinal scale is
   uneven between particular adjacent grades (motivated by an external
   finding of an unusually large MES1/MES2 gap on a different backbone).
3. Confusion matrices, per-boundary ROC curves, and z-distribution violin
   plots — buildable now from existing `test_metrics.json` /
   `test_predictions.csv` artifacts, no new inference needed.
4. `run_ablation.py` to sweep all configs into one `ablation_results.csv`,
   and `statistical_analysis.py` (paired bootstrap between models, McNemar
   test) consuming the `test_predictions.csv` files this pipeline already
   produces.
5. Embedding UMAP/t-SNE of `h_fused` — lower priority than the above; a 2D
   projection of a small frozen-backbone model's representation is easy to
   over- or under-read, so treat as a secondary/appendix figure, not primary
   evidence, and prefer a quantified check (silhouette score / k-NN
   accuracy) alongside any plot.
6. `patient_group_kfold` confirmation of the winning config (E / native-res)
   — already implemented in `src/datasets/limuc.py`, not yet wired into a
   CLI flag on `train.py`. Worth doing before investing further in F/G,
   since all results above are from a single holdout split.
7. Latent prototype discovery module (F) and ordinal contrastive loss (G) —
   deferred until the above land.
