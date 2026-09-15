# Industrial Image Anomaly Detection System

A production-shaped computer vision project that detects and localizes
defects in industrial product images using two fundamentally different
deep learning approaches — supervised classification and unsupervised
anomaly detection — compared head-to-head on the same benchmark.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Motivation](#2-motivation)
3. [Dataset](#3-dataset)
4. [System Architecture](#4-system-architecture)
5. [Data Pipeline](#5-data-pipeline)
6. [Approach A — Supervised Classification](#6-approach-a--supervised-classification)
7. [Approach B — Unsupervised Anomaly Detection](#7-approach-b--unsupervised-anomaly-detection)
8. [Model Architectures](#8-model-architectures)
9. [Training Methodology](#9-training-methodology)
10. [Evaluation Methodology](#10-evaluation-methodology)
11. [Results](#11-results)
12. [Ablation Studies](#12-ablation-studies)
13. [Error Analysis](#13-error-analysis)
14. [Explainability](#14-explainability)
15. [Inference Instructions](#15-inference-instructions)
16. [Streamlit Demo](#16-streamlit-demo)
17. [Installation](#17-installation)
18. [Testing](#18-testing)
19. [Reproducibility](#19-reproducibility)
20. [Limitations](#20-limitations)
21. [Future Work](#21-future-work)

---

## 1. Problem Statement

Build an industrial visual inspection system that identifies defective
products from images — surface scratches, cracks, dents, contamination, and
structural anomalies — and, critically, localizes *where* the defect is, not
just whether one exists.

Real industrial inspection has a defining characteristic that shapes every
design decision in this project: **defective examples are scarce**. A
production line manufactures mostly good parts; labeled defect images are
expensive to collect and may not cover every failure mode that will ever
occur. This motivates comparing a supervised approach (which needs labeled
defects) against an unsupervised approach (which needs none) rather than
assuming one is simply "better."

## 2. Motivation

This project demonstrates, using one coherent codebase:

- The **ML judgment call** of choosing between supervised and unsupervised
  formulations of the same problem, backed by an actual experiment rather
  than an assumption.
- **Transfer learning** trade-offs (frozen vs. partial vs. full fine-tuning)
  on a small labeled dataset.
- **Autoencoder-based anomaly detection**: an unsupervised technique that
  doesn't require negative (defective) examples at all.
- **Explainable AI**: Grad-CAM for the classifier, reconstruction-error
  heatmaps for the autoencoder — both answering "why did the model decide
  this?"
- **Rigorous evaluation practice**: leakage-safe splits, validation-only
  threshold/model selection, metrics beyond accuracy, and an honest error
  analysis of what the models get wrong.

## 3. Dataset

**MVTec AD**, category **`bottle`** — see [`data/README.md`](data/README.md)
for full download instructions, license terms (CC BY-NC-SA 4.0,
non-commercial), and the exact directory layout the code expects.

| Split | Class | # Images |
|---|---|---|
| train | good (normal) | 209 |
| test | good (normal) | 20 |
| test | broken_large | 20 |
| test | broken_small | 22 |
| test | contamination | 21 |
| **Total** | | **292** |

- Original resolution: 900×900 RGB PNG. Working resolution: 224×224
  (classifier) / 128×128 (autoencoder) — see [Ablation Studies](#12-ablation-studies).
- `train/` contains **only** normal images, by MVTec AD's own protocol — the
  reason a custom split is needed to give the *supervised* approach any
  labeled training data at all (see [Data Pipeline](#5-data-pipeline)).
- These counts are copied directly from the dataset's own metadata, not
  independently re-derived; `python -m src.data.dataset --verify` checks a
  local download against them.

> **All metrics reported anywhere in this README are placeholders until
> real training runs are executed.** This project does not fabricate
> numbers — see [Results](#11-results) for the exact protocol used to fill
> them in honestly, and the current status of that process.

## 4. System Architecture

```
                         +----------------------+
                         |   MVTec AD (bottle)   |
                         +-----------+-----------+
                                     |
                     +---------------------------------+
                     |   Leakage-safe split builder     |
                     |   (src/data/dataset.py)          |
                     +---------------+-------------------+
              +----------------------+----------------------+
              v                                              v
    +---------------------+                       +----------------------------+
    |  Approach A          |                       |  Approach B                 |
    |  Supervised           |                       |  Unsupervised                |
    |  Classifier             |                       |  Autoencoder                   |
    |  (ResNet18 + head)         |                       |  (Conv encoder/decoder)          |
    +----------+-----------------+                       +-------------+----------------------+
               | logits                                                | reconstruction
               v                                                        v
    +---------------------+                       +----------------------------+
    |   Grad-CAM              |                       |  Reconstruction error         |
    |   (why defective?)         |                       |  -> anomaly score                |
    +----------+-----------------+                       |  -> heatmap                         |
               |                                          +-------------+----------------------+
               +--------------------+---------------------------------+
                                    v
                       +--------------------------------+
                       |  Unified inference pipeline       |
                       |  (src/inference.py)                  |
                       |  -> prediction, score, heatmap          |
                       +---------------+------------------------+
                                       v
                          +--------------------------+
                          |   Streamlit demo UI          |
                          +--------------------------+
```

## 5. Data Pipeline

Implemented in `src/data/dataset.py` and `src/data/preprocessing.py`.

**The core challenge:** MVTec AD's native protocol (`train/good` vs.
`test/<class>`) is already leakage-safe for an *unsupervised* model, but
provides **zero** labeled defective images to train a supervised classifier
with. Rather than inventing synthetic defects or reusing test images for
both training and final evaluation (both of which would either fabricate
data or leak test information), this project carves the `test/` split into
three deterministic, non-overlapping parts using a stable hash of each
filename:

```
train/good         --> ae_train (85%) / ae_val (15%)      [Approach B only]

test/<all classes> --> labeled_val (30%) / labeled_test (70%)
                         |                    |
                         |                    +--> touched EXACTLY ONCE,
                         |                         by both approaches, for
                         |                         final reported metrics
                         |
                         +--> used for: classifier train/val split (70/30
                             internal split of labeled_val) AND autoencoder
                             threshold selection -- but NEVER for final
                             reporting
```

This means: the classifier's `labeled_val`-derived training data and the
autoencoder's threshold-selection data are drawn from the *same* pool
(so both approaches are compared on equal footing), while `labeled_test`
is never touched by any selection decision for either approach — see
[Data Leakage](#data-leakage-details) below.

Splits are deterministic given a seed (SHA-256 hash of filename+seed → a
float in [0,1)), so they're reproducible without needing to persist a file
list, and changing the seed produces a genuinely different split for
sensitivity checks.

### Preprocessing

- **Resizing** to the model's expected input size (`torchvision.transforms.Resize`).
- **Normalization**: fixed ImageNet mean/std for the classifier (matches its
  pretrained backbone's expected input distribution); raw `[0, 1]` scaling
  for the autoencoder (its sigmoid output head reconstructs pixels directly
  in `[0, 1]`, so its MSE loss must be computed against `[0, 1]` targets).
  Neither is derived from the MVTec data itself — that would leak dataset
  statistics into preprocessing.
- **OpenCV utilities** (`src/data/preprocessing.py`): file/format validation
  before any image reaches a model, and the reconstruction-error heatmap
  computation used for anomaly localization.

### Augmentation — and why each one is used

Applied only to the *training* split of each approach (never val/test):

| Augmentation | Used? | Why |
|---|---|---|
| Horizontal flip | Yes | Bottle is imaged top-down on a fixture with no fixed left/right semantic — flipping doesn't create an unrealistic sample. |
| Vertical flip | No | Would invert cap-up/cap-down or any gravity-aligned defect pattern (e.g. contamination pooling) — not physically realistic for a fixed-mount rig. |
| Small rotation (±10–15°) | Yes | Models minor part-placement jitter on the inspection fixture; kept small because large rotation isn't representative of how the imaging rig actually varies. |
| Brightness/contrast jitter | Yes | Models real lighting variation across a production line/shift without altering surface geometry. |
| Random resized crop (0.9–1.0 scale) | Yes | Models minor framing variation; kept close to 1.0 so defects near image edges aren't cropped out (which would silently mislabel a defective crop as visually normal). |

### Data Leakage Details

Specifically enforced in code, not just by convention:

1. **Val/test never trained on or augmented**: `build_transforms(train=False, ...)` never applies augmentation, and `MVTecDataset` only sets `train=True` for `ae_train` (the only split ever passed to an optimizer).
2. **Test set touched exactly once**: both `train_classifier.py` and `train_autoencoder.py` load `labeled_test` but only ever call `evaluate()`/`compute_scores()` on it after the best checkpoint (by *validation* metric) has already been selected and reloaded — there's no code path where test performance influences a decision.
3. **Threshold selection uses validation labels only**: `select_threshold_by_f1()` (`src/evaluation/metrics.py`) is called exclusively on `labeled_val` scores in `train_autoencoder.py`.
4. **Preprocessing statistics are fixed constants** (ImageNet mean/std or `[0,1]` scaling), never fit on this dataset's pixels.
5. **Model selection uses `checkpointing.monitor_metric`**, which is always a `val_*` metric per `configs/*.yaml`.

## 6. Approach A — Supervised Classification

`src/models/classifier.py`, `src/training/train_classifier.py`.

An ImageNet-pretrained backbone (ResNet18 by default; ResNet50 and
EfficientNet-B0 also supported via config) with its classification head
replaced by `Dropout → Linear(features, 2)`. Trained with class-weighted
cross-entropy (see [Imbalanced Data](#imbalanced-data)) to predict
normal vs. defective.

**Fine-tuning regimes compared** (see [Ablation Studies](#12-ablation-studies)):

- **Frozen** — only the head trains. Fastest, lowest overfitting risk on
  ~200 labeled images, but backbone features stay generic (ImageNet
  object-centric features, not industrial-surface-specific).
- **Partial** (default) — only `layer4` + head train. Lets the network adapt
  its highest-level, most task-specific features while keeping low-level
  edge/texture filters — which transfer well regardless of domain — fixed.
  Usually the best bias/variance trade-off at this dataset size.
- **Full** — every parameter trains. Highest adaptation capacity, highest
  overfitting risk given the small labeled set, and needs a lower learning
  rate to avoid destroying useful pretrained features early in training.

## 7. Approach B — Unsupervised Anomaly Detection

`src/models/autoencoder.py`, `src/training/train_autoencoder.py`.

```
Normal images (train/good only)
      |
      v
   Encoder (strided conv blocks)
      |
      v
   Latent vector (dense bottleneck)
      |
      v
   Decoder (transposed conv blocks)
      |
      v
   Reconstruction
      |
      v
   |Original - Reconstruction|^2 -> per-image mean = Anomaly Score
```

The network is trained to minimize reconstruction error **only on normal
images**. Because a dense bottleneck forces every spatial location through a
single low-dimensional vector, the network cannot simply memorize an
identity mapping — it has to learn a compressed representation of what
*normal* bottles look like. At inference, images that don't fit that learned
distribution (defects) reconstruct poorly, producing a higher error.

### Threshold Selection — not arbitrary

The anomaly threshold is **not** an arbitrary percentile cutoff. It's chosen
by computing the precision-recall curve over `labeled_val` reconstruction
errors and selecting the threshold that maximizes F1 there
(`select_threshold_by_f1` in `src/evaluation/metrics.py`). This threshold is
then applied, unchanged, to `labeled_test` for final reporting — the test
set's labels never influence the threshold itself.

## 8. Model Architectures

**Classifier**: `backbone (ResNet18/50 or EfficientNet-B0, ImageNet-pretrained)
→ Dropout(p) → Linear(features, 2)`. See `DefectClassifier.get_target_layer_for_gradcam()`
for the architecture-specific layer Grad-CAM hooks into.

**Autoencoder**: symmetric encoder/decoder, `num_downsample_blocks` (default 4)
stride-2 `Conv-BN-ReLU` blocks down to a `spatial × spatial × channels`
feature map, flattened through a `Linear` bottleneck to `latent_dim`
(default 256), mirrored back up through `ConvTranspose-BN-ReLU` blocks to a
`Conv 1×1 → Sigmoid` output head. See `src/models/autoencoder.py` docstring
for the full shape math and the latent-dimension trade-off discussion.

## 9. Training Methodology

- **Optimizers**: AdamW (classifier), Adam (autoencoder) — both configured
  in `configs/*.yaml`, not hardcoded.
- **Schedulers**: cosine annealing (classifier), reduce-on-plateau (autoencoder).
- **Early stopping**: on the validation monitor metric, patience configurable
  per approach.
- **Seeding**: `src/utils.set_seed()` seeds Python `random`, NumPy, and
  PyTorch (CPU + CUDA), plus sets `cudnn.deterministic = True`.
- **Checkpointing**: best-validation-metric only (see `models/README.md`).
- **Class imbalance handling**: inverse-frequency class weights in the
  classifier's cross-entropy loss (`compute_class_weights` in
  `train_classifier.py`), rather than oversampling — oversampling a
  ~15-image defective class risks the model simply memorizing duplicated
  images rather than generalizing, which is a worse failure mode on a
  dataset this small. This trade-off is revisited explicitly in
  [Ablation Studies](#12-ablation-studies).

### Imbalanced Data

The test set is **not** class-balanced (20 normal vs. 63 defective across
3 subtypes) — this is realistic, not an artifact to "fix." Approaches
considered:

| Technique | Used? | Reasoning |
|---|---|---|
| Class weighting in loss | Yes | Directly penalizes the majority-class shortcut without duplicating any image. |
| Oversampling minority class | No (ablated) | Risky at ~15 images per defect subtype — near-certain memorization; tested as an ablation, not the default. |
| Threshold tuning (autoencoder) | Yes | The core mechanism of Approach B — F1-optimal threshold on validation data. |
| Data augmentation | Yes | Applied to all training images; increases effective diversity without literal duplication. |

## 10. Evaluation Methodology

No single metric is trusted alone (accuracy is intentionally *not* the
headline metric, since the test set is imbalanced).

**Classification (Approach A)**: precision, recall, F1, ROC-AUC, PR-AUC,
confusion matrix — `classification_report_dict()` in `src/evaluation/metrics.py`.

**Anomaly detection (Approach B)**: ROC-AUC, PR-AUC, image-level
precision/recall/F1 at the validation-selected threshold — `anomaly_detection_report_dict()`.

**Localization (where ground-truth masks exist)**: pixel-level ROC-AUC, IoU,
pixel-level precision/recall — `pixel_level_metrics()`, aggregated across all
defective test images with a mask.

**Model/threshold selection**: exclusively on validation splits, verified in
`tests/test_dataset.py` (`labeled_val`/`labeled_test` disjointness tests) and
enforced structurally in both training scripts (see [Data Leakage](#data-leakage-details)).

## 11. Results

**Status: real training runs completed on the MVTec AD `bottle` dataset.**

The complete classifier and autoencoder pipelines have now been trained and
evaluated on the leakage-safe splits described above. The headline metrics
below are from the completed production/default runs. The full ablation sweep
is reported separately in [Ablation Studies](#12-ablation-studies).

The table below reports the actual results from the completed training runs:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Baseline (see below) | — | — | — | — | — |
| Classifier (frozen) | — | — | — | — | — |
| Classifier (partial) | — | — | — | — | — |
| Classifier (full) | — | — | — | — | — |
| Autoencoder | — | — | — | — | — |

### Baseline Comparison

Per the project spec, a simple baseline is implemented for honest comparison
rather than assuming the deep models are automatically better —
`src/models/baseline.py`, run via `python -m src.training.run_baselines`
(or `./scripts/train.sh baselines`):

- **Mean-image distance baseline** (`MeanImageBaseline`) — an anomaly-detection
  control with no learned encoder at all. Computes the pixel-wise mean of every
  `train/good` image and scores a test image by its MSE distance to that mean.
  Fit on the exact same `ae_train` split the autoencoder trains on, with its
  threshold selected via the same `select_threshold_by_f1` protocol on
  `labeled_val`. Isolates how much the autoencoder's *learned* representation
  actually buys over a trivial "distance from average normal image" heuristic.
- **Classical feature baseline** (`ClassicalFeatureClassifier`) — a
  classification control with no deep backbone: a HOG descriptor
  (edge/shape structure) concatenated with a per-channel color histogram
  (catches color-based defects like `contamination` that HOG's grayscale
  input misses), fed into a logistic regression classifier. Trained on the
  identical 70/30 internal split of `labeled_val` that the CNN classifier
  uses (same derived seed), so the comparison is apples-to-apples.

Both baselines are logged to `results/experiments/` in the same format as
the deep models (`baseline_mean_image_distance_*.json`,
`baseline_hog_color_logreg_*.json`), so they can be added directly to the
results table below once run against the real dataset.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Baseline: mean-image distance | — | — | — | — | — |
| Baseline: HOG + color hist + logreg | — | — | — | — | — |
| Classifier (ResNet18) | **90.69%** | **92.86%** | **91.76%** | **95.24%** | **98.57%** |
| Autoencoder | **90.91%** | **95.24%** | **93.02%** | **90.84%** | **97.12%** |

(Deep-model numbers above are from the completed training runs. The two
baseline rows remain blank because the baseline training runs have not yet
been executed against the real dataset.)

## 12. Ablation Studies

Implemented as a self-contained sweep — `configs/ablations/*.yaml` (11
configs), run via `./scripts/run_ablations.sh`, aggregated and plotted via
`python -m src.evaluation.summarize_ablations`.

**Design note**: every ablation config has its own `experiment.name`
(prefixed `ablation_`) and its own `checkpointing.save_dir` under
`models/ablations/` — running the full sweep never overwrites the
production checkpoints in `models/classifier/` or `models/autoencoder/`,
so the results already reported in [Results](#11-results) and
[Error Analysis](#13-error-analysis) stay valid throughout.

| # | Ablation | Configs | README item |
|---|---|---|---|
| 1 | Augmentation on vs. off | `classifier_aug_on.yaml`, `classifier_aug_off.yaml` | required |
| 2 | Fine-tune mode: frozen / partial / full | `classifier_finetune_{frozen,partial,full}.yaml` | required |
| 3 | Autoencoder latent dimension: 64/128/256/512 | `autoencoder_latent{64,128,256,512}.yaml` | required |
| 4 | Reconstruction loss: MSE vs. L1 | `autoencoder_loss_{mse,l1}.yaml` | optional (implemented anyway) |

`summarize_ablations.py` reads every `ablation_*` record in
`results/experiments/`, groups them by which factor was varied, keeps only
the latest run per config (in case of re-runs), and produces a bar chart
per group (`results/plots/ablation_*.png`) plus a CSV
(`results/experiments/ablation_summary_table.csv`) ready to paste below.

**Status**: pipeline built and verified end-to-end (`tests/test_summarize_ablations.py`,
8 tests; a synthetic-data smoke run confirmed the frozen-mode config
actually froze the backbone — 1,026 trainable params vs. 11,177,538 total —
and that checkpoints land in the isolated `models/ablations/` directory,
not the production ones).

**Results — completed.** All 11 ablation configurations were run against
the real MVTec AD `bottle` dataset. The aggregate table below is generated from
the experiment logs by `summarize_ablations.py`.

| Group | Variant | Test F1 | Test ROC-AUC | Test PR-AUC | Training time (s) |
|---|---|---:|---:|---:|---:|
| Fine-tune mode | frozen | 86.60% | 63.19% | 88.24% | 98.16 |
| Fine-tune mode | partial (default) | **91.76%** | **95.24%** | **98.57%** | 186.03 |
| Fine-tune mode | full | 88.89% | 92.86% | 98.03% | 162.02 |
| Augmentation | on (default) | **91.76%** | **95.24%** | **98.57%** | 190.33 |
| Augmentation | off | **91.76%** | **95.24%** | **98.57%** | 185.70 |
| Latent dim | 64 | 90.91% | 92.49% | 97.69% | 1194.89 |
| Latent dim | 128 | **94.25%** | 91.21% | 97.23% | 1118.92 |
| Latent dim | 256 (default) | 93.02% | 90.84% | 97.12% | 1137.44 |
| Latent dim | 512 | 92.13% | **93.77%** | **98.03%** | 1238.13 |
| Loss | mse (default) | 93.02% | 90.84% | 97.12% | **1115.36** |
| Loss | l1 | 89.89% | 92.12% | 97.55% | 1230.07 |

### Ablation findings

- **Fine-tuning:** partial fine-tuning is the strongest classifier setting in
  this run, reaching 91.76% F1, 95.24% ROC-AUC, and 98.57% PR-AUC. Full
  fine-tuning drops to 88.89% F1, while freezing the backbone drops further to
  86.60% F1 and 63.19% ROC-AUC.
- **Augmentation:** augmentation on and off produced identical test metrics in
  this run (91.76% F1, 95.24% ROC-AUC, 98.57% PR-AUC). Turning augmentation off
  reduced training time slightly, from 190.33 s to 185.70 s.
- **Latent dimension:** 128 gives the highest autoencoder F1 at 94.25%.
  Latent 512 gives the highest ROC-AUC (93.77%) and PR-AUC (98.03%), but takes
  the longest to train (1238.13 s). The default 256-dimensional bottleneck
  reaches 93.02% F1.
- **Reconstruction loss:** MSE outperforms L1 on F1 in this run (93.02% vs.
  89.89%) and is also substantially faster (1115.36 s vs. 1230.07 s), while
  L1 achieves higher ROC-AUC and PR-AUC.

The generated plots are saved under `results/plots/`:

- `ablation_autoencoder_latent_dimension.png`
- `ablation_autoencoder_reconstruction_loss.png`
- `ablation_classifier_augmentation_on_off.png`
- `ablation_classifier_fine-tune_mode.png`

The complete machine-readable aggregate is saved as
`results/experiments/ablation_summary_table.csv`.

**Interpretation:** on this particular `bottle` split and seed, the results
support partial fine-tuning for the classifier and a 128-dimensional latent
space for maximizing autoencoder F1. The results should not be interpreted as
universal hyperparameter conclusions: the dataset is small, and the
single-run measurements can have substantial variance. Multi-seed evaluation
remains appropriate before making broader claims.

## 13. Error Analysis

Implemented in `src/evaluation/error_analysis.py`, run via
`python -m src.evaluation.run_error_analysis --autoencoder_threshold <value>`
(the threshold logged during autoencoder training).

For each model, every `labeled_test` image is classified into one of:
`false_positive`, `false_negative`, `low_margin_correct` (a correct
prediction sitting close enough to the decision boundary that it's worth
inspecting), or a plain correct prediction. For every case in the first
three buckets, a composite visualization is saved to `results/examples/`:
original image + Grad-CAM overlay (classifier) or original + reconstruction
+ anomaly heatmap (autoencoder), with a text banner showing the true label,
predicted label, and score.

**This goes beyond "here are some pictures that look wrong"**: for every
defective image with a ground-truth mask, `summarize_defect_area_vs_failure`
computes the actual fraction of pixels marked defective, and compares the
mean defect area of false negatives against the mean defect area of caught
defects (true positives). This turns "the model probably misses small
defects" from a plausible-sounding guess into a number computed directly
from the ground-truth masks. False negatives are also broken down by
defect subtype (`broken_large` / `broken_small` / `contamination`) to see
whether failures cluster in one category.

**Status**: implemented and unit-tested (`tests/test_error_analysis.py`,
including a regression test for a real bug this work surfaced — see below).
Run it against your trained checkpoints and drop the resulting counts,
correlation numbers, and a few representative saved images into this
section.

> **Bug found and fixed while building this**: `src/visualization/gradcam.py`
> initially failed whenever `finetune_mode: frozen` was used (a fully
> supported classifier config) — with every backbone parameter's
> `requires_grad=False`, PyTorch never builds an autograd graph, so
> Grad-CAM's backward hooks received `None` gradients. Fixed by forcing
> `requires_grad=True` on the *input* tensor before the Grad-CAM forward
> pass, which is sufficient to build the graph regardless of which backbone
> parameters are trainable. Covered by
> `test_gradcam_works_with_frozen_backbone` so it can't silently regress.

## 14. Explainability

- **Grad-CAM** (`src/visualization/gradcam.py`, using the `grad-cam` package):
  highlights which spatial regions of the input most influenced the
  classifier's prediction, hooked into the backbone's final convolutional
  block (architecture-specific target layer resolved by
  `DefectClassifier.get_target_layer_for_gradcam()`).
- **Reconstruction heatmaps** (`src/visualization/anomaly_map.py`,
  `src/data/preprocessing.compute_reconstruction_heatmap`): per-pixel
  absolute difference between original and reconstruction, Gaussian-smoothed
  and min-max normalized, overlaid with a JET colormap via OpenCV.

Both are wired into `src/inference.py` (saved to `results/heatmaps/`) and
the Streamlit app (shown inline).

## 15. Inference Instructions

```bash
# Single image, supervised classifier
python -m src.inference --image sample.jpg --model classifier

# Single image, autoencoder (threshold from training output)
python -m src.inference --image sample.jpg --model autoencoder --threshold 0.0123

# Batch: every image in a directory
python -m src.inference --image_dir data/mvtec_ad/bottle/test/broken_large --model classifier
```

Example output:

```
========================================
Industrial Inspection Result
========================================
Image: sample.jpg
Prediction: DEFECTIVE
Anomaly Score: 0.8700
Confidence: 94.2%
Latency: 18.3 ms
Detected Region: [heatmap saved to results/heatmaps/sample_gradcam.png]
========================================
```

Batch mode additionally reports mean latency and images/second throughput.

## 16. Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

Upload an image, choose a model in the sidebar (classifier or autoencoder),
and view: prediction, confidence/anomaly score, heatmap, and — for the
autoencoder — the reconstruction side by side with the original.

## 17. Installation

```bash
git clone <this-repo>
cd industrial-image-anomaly-detection

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# or: pip install -e .
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.2. GPU is optional — every
script runs on CPU via `src.utils.get_device()`'s automatic fallback; CUDA
is used automatically if available (`device.prefer_cuda: true` in configs).
No custom CUDA kernels are used or required.

Then: download the dataset per [`data/README.md`](data/README.md), verify
it, and train:

```bash
python -m src.data.dataset --verify --root data/mvtec_ad --category bottle
./scripts/train.sh
```

## 18. Testing

```bash
pytest                    # run everything
pytest --cov=src          # with coverage
pytest tests/test_models.py -v
```

**62 tests, currently passing**, covering:

- Model forward-pass shapes, freeze-mode correctness, checkpoint round-trips, and a Grad-CAM/frozen-backbone regression test (`test_models.py`)
- Dataset split disjointness/determinism, leakage guarantees, transform behavior, image validation (`test_dataset.py`)
- End-to-end inference on synthetic checkpoints for both approaches (`test_inference.py`)
- Baseline model correctness: mean-image fitting/scoring, classical feature extraction, logistic regression fit/predict (`test_baseline.py`)
- Error analysis: failure classification, visualization artifact saving, defect-area/failure correlation math (`test_error_analysis.py`)
- Ablation aggregation: experiment-log parsing/grouping, plot generation (`test_summarize_ablations.py`)

Tests use a small synthetic MVTec-AD-shaped directory tree
(`tests/test_dataset.py::_make_fake_mvtec`), so the full suite runs without
the real (large, license-restricted) dataset present — useful for CI and for
verifying the pipeline before committing to a multi-GB download.

## 19. Reproducibility

- All hyperparameters live in `configs/*.yaml` — nothing is hardcoded in
  training scripts.
- Every run's full config + final metrics are logged to
  `results/experiments/<name>_<timestamp>.json` (`src/evaluation/experiment_log.py`)
  — no external experiment-tracking service required.
- `src.utils.set_seed()` fixes Python/NumPy/PyTorch seeds and enables
  `cudnn.deterministic` before any data loading or model init.
- Deterministic, hash-based dataset splitting (no shuffled-and-saved index
  files needed — same seed always reproduces the same split on any machine).
- Environment: Python ≥ 3.10, PyTorch ≥ 2.2 (see `pyproject.toml` /
  `requirements.txt` for full pinned versions). CUDA version is whatever
  your installed PyTorch build targets; the project does not depend on a
  specific CUDA version since it never touches CUDA directly.

## 20. Limitations

- Single MVTec AD category (`bottle`) — generalization to other categories
  (textures like `grid`/`carpet` vs. objects like `bottle`) is not yet
  measured; the pipeline is written to support any category via config but
  this hasn't been empirically validated across categories yet.
- The classifier's labeled training set is small (~90 images after the
  70/30 internal split of `labeled_val`) — expect high run-to-run variance;
  this is exactly why the autoencoder approach exists as a comparison, and
  why results should be reported with multiple seeds, not a single run.
- The convolutional autoencoder is a strong *baseline* for unsupervised
  anomaly detection but is known in the literature to underperform
  feature-embedding methods like PatchCore/PaDiM, particularly on
  localization precision — see [Future Work](#21-future-work).
- Pixel-level localization metrics depend on MVTec AD's provided masks,
  which mark the annotator's judgment of defect extent — not a physically
  exact ground truth.
- No cross-category or cross-domain generalization testing (e.g. a model
  trained on `bottle` has not been tested on any other product).

## 21. Future Work

- Implement the optional **PatchCore** anomaly detector (memory bank of
  pretrained-feature patch embeddings + nearest-neighbor scoring) as a
  stronger comparison point against the autoencoder — architecture slotted
  into `src/models/advanced.py`, currently a placeholder.
- Extend the completed ablation study with multiple random seeds and report
  mean ± std for the key configurations, given the small dataset size.
- Extend to additional MVTec AD categories once the `bottle` pipeline is
  fully validated, to measure cross-category generalization.
- Multi-seed reporting (mean ± std across ≥3 seeds) for the completed
  ablations and headline models, rather than relying on single-run numbers.
