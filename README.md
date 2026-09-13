**# Industrial Image Anomaly Detection System**

A production-shaped computer vision project that detects and localizes

defects in industrial product images using two fundamentally different

deep learning approaches — supervised classification and unsupervised

anomaly detection — compared head-to-head on the same benchmark.

**---**

**## Table of Contents**

1\. [Problem Statement]\(#1-problem-statement)

2\. [Motivation]\(#2-motivation)

3\. [Dataset]\(#3-dataset)

4\. [System Architecture]\(#4-system-architecture)

5\. [Data Pipeline]\(#5-data-pipeline)

6\. [Approach A — Supervised Classification]\(#6-approach-a--supervised-classification)

7\. [Approach B — Unsupervised Anomaly Detection]\(#7-approach-b--unsupervised-anomaly-detection)

8\. [Model Architectures]\(#8-model-architectures)

9\. [Training Methodology]\(#9-training-methodology)

10\. [Evaluation Methodology]\(#10-evaluation-methodology)

11\. [Results]\(#11-results)

12\. [Ablation Studies]\(#12-ablation-studies)

13\. [Error Analysis]\(#13-error-analysis)

14\. [Explainability]\(#14-explainability)

15\. [Inference Instructions]\(#15-inference-instructions)

16\. [Streamlit Demo]\(#16-streamlit-demo)

17\. [Installation]\(#17-installation)

18\. [Testing]\(#18-testing)

19\. [Reproducibility]\(#19-reproducibility)

20\. [Limitations]\(#20-limitations)

21\. [Future Work]\(#21-future-work)

**---**

**## 1. Problem Statement**

Build an industrial visual inspection system that identifies defective

products from images — surface scratches, cracks, dents, contamination, and

structural anomalies — and, critically, localizes *\*where\** the defect is, not

just whether one exists.

Real industrial inspection has a defining characteristic that shapes every

design decision in this project: **\*\*defective examples are scarce\*\***. A

production line manufactures mostly good parts; labeled defect images are

expensive to collect and may not cover every failure mode that will ever

occur. This motivates comparing a supervised approach (which needs labeled

defects) against an unsupervised approach (which needs none) rather than

assuming one is simply "better."

**## 2. Motivation**

This project demonstrates, using one coherent codebase:

\- The **\*\*ML judgment call\*\*** of choosing between supervised and unsupervised

  formulations of the same problem, backed by an actual experiment rather

  than an assumption.

\- **\*\*Transfer learning\*\*** trade-offs (frozen vs. partial vs. full fine-tuning)

  on a small labeled dataset.

\- **\*\*Autoencoder-based anomaly detection\*\***: an unsupervised technique that

  doesn't require negative (defective) examples at all.

\- **\*\*Explainable AI\*\***: Grad-CAM for the classifier, reconstruction-error

  heatmaps for the autoencoder — both answering "why did the model decide

  this?"

\- **\*\*Rigorous evaluation practice\*\***: leakage-safe splits, validation-only

  threshold/model selection, metrics beyond accuracy, and an honest error

  analysis of what the models get wrong.

**## 3. Dataset**

**\*\*MVTec AD\*\***, category **\*\*\`bottle\`\*\*** — see [\`data/README.md\`]\(data/README.md)

for full download instructions, license terms (CC BY-NC-SA 4.0,

non-commercial), and the exact directory layout the code expects.

\| Split | Class | # Images |

\|---|---|---|

\| train | good (normal) | 209 |

\| test | good (normal) | 20 |

\| test | broken\_large | 20 |

\| test | broken\_small | 22 |

\| test | contamination | 21 |

\| **\*\*Total\*\*** | | **\*\*292\*\*** |

\- Original resolution: 900×900 RGB PNG. Working resolution: 224×224

  (classifier) / 128×128 (autoencoder) — see [Ablation Studies]\(#12-ablation-studies).

\- \`train/\` contains **\*\*only\*\*** normal images, by MVTec AD's own protocol — the

  reason a custom split is needed to give the *\*supervised\** approach any

  labeled training data at all (see [Data Pipeline]\(#5-data-pipeline)).

\- These counts are copied directly from the dataset's own metadata, not

  independently re-derived; \`python -m src.data.dataset --verify\` checks a

  local download against them.

\> \*\*All metrics reported anywhere in this README are placeholders until

\> real training runs are executed.\*\* This project does not fabricate

\> numbers — see [Results]\(#11-results) for the exact protocol used to fill

\> them in honestly, and the current status of that process.

**## 4. System Architecture**

\`\`\`

                         +----------------------+

                         |   MVTec AD (bottle)   |

                         +-----------+-----------+

                                     |

                     +---------------------------------+

                     |   Deterministic split builder     |

                     |   (src/data/dataset.py)          |

                     +---------------+-------------------+

              +----------------------+----------------------+

              v                                              v

    +---------------------+                       +----------------------------+

    |  Approach A          |                       |  Approach B                 |

    |  Supervised           |                       |  Unsupervised                |

    |  Classifier             |                       |  Autoencoder                   |

    |  (ResNet18 + head)         |                       |  (Conv encoder/decoder)          |

    +----------+-----------------+                       +-------------+----------------------+

               | logits                                                | reconstruction

               v                                                        v

    +---------------------+                       +----------------------------+

    |   Grad-CAM              |                       |  Reconstruction error         |

    |   (why defective?)         |                       |  -> anomaly score                |

    +----------+-----------------+                       |  -> heatmap                         |

               |                                          +-------------+----------------------+

               +--------------------+---------------------------------+

                                    v

                       +--------------------------------+

                       |  Unified inference pipeline       |

                       |  (src/inference.py)                  |

                       |  -> prediction, score, heatmap          |

                       +---------------+------------------------+

                                       v

                          +--------------------------+

                          |   Streamlit demo UI          |

                          +--------------------------+

\`\`\`

**## 5. Data Pipeline**

Implemented in \`src/data/dataset.py\` and \`src/data/preprocessing.py\`.

**\*\*The core challenge:\*\*** MVTec AD's native protocol (\`train/good\` vs.

\`test/\<class>\`) is already leakage-safe for an *\*unsupervised\** model, but

provides **\*\*zero\*\*** labeled defective images to train a supervised classifier

with. Rather than inventing synthetic defects or reusing test images for

both training and final evaluation (both of which would either fabricate

data or leak test information), this project carves the \`test/\` split into

three deterministic, non-overlapping parts using a stable hash of each

filename:

\`\`\`

train/good         --> ae\_train (85%) / ae\_val (15%)      [Approach B only]

test/\<all classes> --> labeled\_val (30%) / labeled\_test (70%)

                         |                    |

                         |                    +--> touched EXACTLY ONCE,

                         |                         by both approaches, for

                         |                         final reported metrics

                         |

                         +--> used for: classifier train/val split (70/30

                             internal split of labeled\_val) AND autoencoder

                             threshold selection -- but NEVER for final

                             reporting

\`\`\`

This means: the classifier's \`labeled\_val\`-derived training data and the

autoencoder's threshold-selection data are drawn from the *\*same\** pool

(so both approaches are compared on equal footing), while \`labeled\_test\`

is never touched by any selection decision for either approach — see

[Data Leakage]\(#data-leakage-details) below.

Splits are deterministic given a seed (SHA-256 hash of filename+seed → a

float in [0,1)), so they're reproducible without needing to persist a file

list, and changing the seed produces a genuinely different split for

sensitivity checks.

**### Preprocessing**

\- **\*\*Resizing\*\*** to the model's expected input size (\`torchvision.transforms.Resize\`).

\- **\*\*Normalization\*\***: fixed ImageNet mean/std for the classifier (matches its

  pretrained backbone's expected input distribution); raw \`[0, 1]\` scaling

  for the autoencoder (its sigmoid output head reconstructs pixels directly

  in \`[0, 1]\`, so its MSE loss must be computed against \`[0, 1]\` targets).

  Neither is derived from the MVTec data itself — that would leak dataset

  statistics into preprocessing.

\- **\*\*OpenCV utilities\*\*** (\`src/data/preprocessing.py\`): file/format validation

  before any image reaches a model, and the reconstruction-error heatmap

  computation used for anomaly localization.

**### Augmentation — and why each one is used**

Applied only to the *\*training\** split of each approach (never val/test):

\| Augmentation | Used? | Why |

\|---|---|---|

\| Horizontal flip | Yes | Bottle is imaged top-down on a fixture with no fixed left/right semantic — flipping doesn't create an unrealistic sample. |

\| Vertical flip | No | Would invert cap-up/cap-down or any gravity-aligned defect pattern (e.g. contamination pooling) — not physically realistic for a fixed-mount rig. |

\| Small rotation (±10–15°) | Yes | Models minor part-placement jitter on the inspection fixture; kept small because large rotation isn't representative of how the imaging rig actually varies. |

\| Brightness/contrast jitter | Yes | Models real lighting variation across a production line/shift without altering surface geometry. |

\| Random resized crop (0.9–1.0 scale) | Yes | Models minor framing variation; kept close to 1.0 so defects near image edges aren't cropped out (which would silently mislabel a defective crop as visually normal). |

**### Data Leakage Details**

The following safeguards are enforced in code:

1\. **\*\*Val/test never trained on or augmented\*\***: \`build\_transforms(train=False, ...)\` never applies augmentation, and \`MVTecDataset\` only sets \`train=True\` for \`ae\_train\` (the only split ever passed to an optimizer).

2\. **\*\*Test set touched exactly once\*\***: both \`train\_classifier.py\` and \`train\_autoencoder.py\` load \`labeled\_test\` but only ever call \`evaluate()\`/\`compute\_scores()\` on it after the best checkpoint (by *\*validation\** metric) has already been selected and reloaded — there's no code path where test performance influences a decision.

3\. **\*\*Threshold selection uses validation labels only\*\***: \`select\_threshold\_by\_f1()\` (\`src/evaluation/metrics.py\`) is called exclusively on \`labeled\_val\` scores in \`train\_autoencoder.py\`.

4\. **\*\*Preprocessing statistics are fixed constants\*\*** (ImageNet mean/std or \`[0,1]\` scaling), never fit on this dataset's pixels.

5\. **\*\*Model selection uses \`checkpointing.monitor\_metric\`\*\***, which is always a \`val\_\*\` metric per \`configs/\*.yaml\`.

**## 6. Approach A — Supervised Classification**

\`src/models/classifier.py\`, \`src/training/train\_classifier.py\`.

An ImageNet-pretrained backbone (ResNet18 by default; ResNet50 and

EfficientNet-B0 also supported via config) with its classification head

replaced by \`Dropout → Linear(features, 2)\`. Trained with class-weighted

cross-entropy (see [Imbalanced Data]\(#imbalanced-data)) to predict

normal vs. defective.

**\*\*Fine-tuning regimes compared\*\*** (see [Ablation Studies]\(#12-ablation-studies)):

\- **\*\*Frozen\*\*** — only the head trains. Fastest, lowest overfitting risk on

  \~200 labeled images, but backbone features stay generic (ImageNet

  object-centric features, not industrial-surface-specific).

\- **\*\*Partial\*\*** (default) — only \`layer4\` + head train. Lets the network adapt

  its highest-level, most task-specific features while keeping low-level

  edge/texture filters — which transfer well regardless of domain — fixed.

  Usually the best bias/variance trade-off at this dataset size.

\- **\*\*Full\*\*** — every parameter trains. Highest adaptation capacity, highest

  overfitting risk given the small labeled set, and needs a lower learning

  rate to avoid destroying useful pretrained features early in training.

**## 7. Approach B — Unsupervised Anomaly Detection**

\`src/models/autoencoder.py\`, \`src/training/train\_autoencoder.py\`.

\`\`\`

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

\`\`\`

The network is trained to minimize reconstruction error \*\*only on normal

images\*\*. Because a dense bottleneck forces every spatial location through a

single low-dimensional vector, the network cannot simply memorize an

identity mapping — it has to learn a compressed representation of what

*\*normal\** bottles look like. At inference, images that don't fit that learned

distribution (defects) reconstruct poorly, producing a higher error.

**### Threshold Selection — not arbitrary**

The anomaly threshold is **\*\*not\*\*** an arbitrary percentile cutoff. It's chosen

by computing the precision-recall curve over \`labeled\_val\` reconstruction

errors and selecting the threshold that maximizes F1 there

(\`select\_threshold\_by\_f1\` in \`src/evaluation/metrics.py\`). This threshold is

then applied, unchanged, to \`labeled\_test\` for final reporting — the test

set's labels never influence the threshold itself.

**## 8. Model Architectures**

**\*\*Classifier\*\***: \`backbone (ResNet18/50 or EfficientNet-B0, ImageNet-pretrained)

→ Dropout(p) → Linear(features, 2)\`. See \`DefectClassifier.get\_target\_layer\_for\_gradcam()\`

for the architecture-specific layer Grad-CAM hooks into.

**\*\*Autoencoder\*\***: symmetric encoder/decoder, \`num\_downsample\_blocks\` (default 4)

stride-2 \`Conv-BN-ReLU\` blocks down to a \`spatial × spatial × channels\`

feature map, flattened through a \`Linear\` bottleneck to \`latent\_dim\`

(default 256), mirrored back up through \`ConvTranspose-BN-ReLU\` blocks to a

\`Conv 1×1 → Sigmoid\` output head. See \`src/models/autoencoder.py\` docstring

for the full shape math and the latent-dimension trade-off discussion.

**## 9. Training Methodology**

\- **\*\*Optimizers\*\***: AdamW (classifier), Adam (autoencoder) — both configured

  in \`configs/\*.yaml\`, not hardcoded.

\- **\*\*Schedulers\*\***: cosine annealing (classifier), reduce-on-plateau (autoencoder).

\- **\*\*Early stopping\*\***: on the validation monitor metric, patience configurable

  per approach.

\- **\*\*Seeding\*\***: \`src/utils.set\_seed()\` seeds Python \`random\`, NumPy, and

  PyTorch (CPU + CUDA), plus sets \`cudnn.deterministic = True\`.

\- **\*\*Checkpointing\*\***: best-validation-metric only (see \`models/README.md\`).

\- **\*\*Class imbalance handling\*\***: inverse-frequency class weights in the

  classifier's cross-entropy loss (\`compute\_class\_weights\` in

  \`train\_classifier.py\`), rather than oversampling — oversampling a

  \~15-image defective class risks the model simply memorizing duplicated

  images rather than generalizing, which is a worse failure mode on a

  dataset this small. This trade-off is revisited explicitly in

  [Ablation Studies]\(#12-ablation-studies).

**### Imbalanced Data**

The test set is **\*\*not\*\*** class-balanced (20 normal vs. 63 defective across

3 subtypes) — this is realistic, not an artifact to "fix." Approaches

considered:

\| Technique | Used? | Reasoning |

\|---|---|---|

\| Class weighting in loss | Yes | Directly penalizes the majority-class shortcut without duplicating any image. |

\| Oversampling minority class | No (ablated) | Risky at \~15 images per defect subtype — near-certain memorization; tested as an ablation, not the default. |

\| Threshold tuning (autoencoder) | Yes | The core mechanism of Approach B — F1-optimal threshold on validation data. |

\| Data augmentation | Configured | Training augmentation is defined in config; the current dataset implementation applies it to `ae_train`. |

**## 10. Evaluation Methodology**

No single metric is trusted alone (accuracy is intentionally *\*not\** the

headline metric, since the test set is imbalanced).

**\*\*Classification (Approach A)\*\***: precision, recall, F1, ROC-AUC, PR-AUC,

confusion matrix — \`classification\_report\_dict()\` in \`src/evaluation/metrics.py\`.

**\*\*Anomaly detection (Approach B)\*\***: ROC-AUC, PR-AUC, image-level

precision/recall/F1 at the validation-selected threshold — \`anomaly\_detection\_report\_dict()\`.

**\*\*Localization (where ground-truth masks exist)\*\***: pixel-level ROC-AUC, IoU,

pixel-level precision/recall — \`pixel\_level\_metrics()\`, aggregated across all

defective test images with a mask.

**\*\*Model/threshold selection\*\***: exclusively on validation splits, verified in

\`tests/test\_dataset.py\` (\`labeled\_val\`/\`labeled\_test\` disjointness tests) and

enforced structurally in both training scripts (see [Data Leakage]\(#data-leakage-details)).

**## 11. Results

The following results are from completed training runs on the MVTec AD `bottle`
dataset.

### Final test metrics

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| **Classifier (ResNet18)** | **90.69%** | **92.86%** | **91.76%** | **95.24%** | **98.57%** |
| **Autoencoder** | **90.91%** | **95.24%** | **93.02%** | **90.84%** | **97.12%** |

**Classifier test confusion matrix:** `[[9, 4], [3, 39]]`

**Autoencoder test confusion matrix:** `[[9, 4], [2, 40]]`

The autoencoder's selected anomaly threshold was **0.00170 reconstruction MSE**.
It is applied unchanged during final test evaluation and Streamlit inference.

Experiment logs are stored under `results/experiments/`, including the
configuration and final metrics for each run.

### Interpretation

The supervised classifier achieves the stronger ROC-AUC (**95.24%**) and PR-AUC
(**98.57%**), while the autoencoder achieves the higher image-level F1
(**93.02%**) on this particular test partition. These results are specific to
this custom `bottle` experimental protocol and are not a claim of
state-of-the-art performance on the full MVTec AD benchmark.

### Baseline Comparison

Baseline implementations and a complete baseline comparison are **future work**.
Planned baselines include:

- **Pixel-level reconstruction baseline**: mean-image subtraction and raw
  pixel-space distance from the normal training-image mean.
- **Classical feature baseline**: HOG or color-histogram features with a linear
  SVM/logistic-regression classifier.

## 12. Ablation Studies**

At least three are planned, each just a config change + re-run:

1\. **\*\*Augmentation on vs. off\*\*** — \`configs/classifier.yaml: augmentation.enabled\`.

2\. **\*\*Frozen vs. partial vs. fine-tuned backbone\*\*** — \`configs/classifier.yaml: model.finetune\_mode\`.

3\. **\*\*Autoencoder latent dimension sweep\*\*** (64 / 128 / 256 / 512) — \`configs/autoencoder.yaml: model.latent\_dim\`. Expected trade-off: too small under-reconstructs normal texture (more false positives), too large starts reconstructing defects too (more false negatives) — see \`src/models/autoencoder.py\` docstring.

4\. *\*(Optional)\** Image resolution (128 vs. 224) — cost vs. accuracy trade-off.

5\. *\*(Optional)\** Reconstruction loss: MSE vs. L1 vs. SSIM-based.

Results tables/plots go in \`results/plots/ablation\_comparison.png\`, produced

by \`notebooks/02\_model\_analysis.ipynb\` once multiple runs are logged.

**## 13. Error Analysis**

**\*\*Planned methodology\*\*** (see \`notebooks/02\_model\_analysis.ipynb\` section 4):

for each trained model, walk the test set, isolate false positives (normal

flagged defective), false negatives (defective flagged normal), and the

lowest-confidence correct predictions, then render each with its

explanation artifact (Grad-CAM or reconstruction heatmap) to reason about

*\*why\** — e.g. "false negatives cluster on \`broken\_small\`, where the defect

region is under N% of image area and the autoencoder's downsampling may be

smoothing it out." Saved to \`results/examples/\`.

**## 14. Explainability**

\- **\*\*Grad-CAM\*\*** (\`src/visualization/gradcam.py\`, using the \`grad-cam\` package):

  highlights which spatial regions of the input most influenced the

  classifier's prediction, hooked into the backbone's final convolutional

  block (architecture-specific target layer resolved by

  \`DefectClassifier.get\_target\_layer\_for\_gradcam()\`).

\- **\*\*Reconstruction heatmaps\*\*** (\`src/visualization/anomaly\_map.py\`,

  \`src/data/preprocessing.compute\_reconstruction\_heatmap\`): per-pixel

  absolute difference between original and reconstruction, Gaussian-smoothed

  and min-max normalized, overlaid with a JET colormap via OpenCV.

Both are wired into \`src/inference.py\` (saved to \`results/heatmaps/\`) and

the Streamlit app (shown inline).

**## 15. Inference Instructions**

\`\`\`bash

\# Single image, supervised classifier

python -m src.inference --image sample.jpg --model classifier

\# Single image, autoencoder (threshold from training output)

python -m src.inference --image sample.jpg --model autoencoder --threshold 0.00170

\# Batch: every image in a directory

python -m src.inference --image\_dir data/mvtec\_ad/bottle/test/broken\_large --model classifier

\`\`\`

The command prints the image path, prediction, score/confidence, latency, and
the path to the generated explanation artifact.

Batch mode additionally reports mean latency and images/second throughput.

**## 16. Streamlit Demo**

\`\`\`bash

streamlit run app/streamlit\_app.py

\`\`\`

Upload an image, choose a model in the sidebar (classifier or autoencoder),

and view: prediction, confidence/anomaly score, heatmap, and — for the

autoencoder — the reconstruction side by side with the original.

**## 17. Installation**

\`\`\`bash

git clone \<this-repo>

cd industrial-image-anomaly-detection

python3 -m venv .venv

source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

\# or: pip install -e .

\`\`\`

**\*\*Requirements\*\***: Python ≥ 3.10, PyTorch ≥ 2.2. GPU is optional — every

script runs on CPU via \`src.utils.get\_device()\`'s automatic fallback; CUDA

is used automatically if available (\`device.prefer\_cuda: true\` in configs).

No custom CUDA kernels are used or required.

Then: download the dataset per [\`data/README.md\`]\(data/README.md), verify

it, and train:

\`\`\`bash

python -m src.data.dataset --verify --root data/mvtec\_ad --category bottle

./scripts/train.sh

\`\`\`

**## 18. Testing**

\`\`\`bash

pytest                    # run everything

pytest --cov=src          # with coverage

pytest tests/test\_models.py -v

\`\`\`

**\*\*38 tests, currently passing\*\***, covering:

\- Model forward-pass shapes, freeze-mode correctness, checkpoint round-trips (\`test\_models.py\`)

\- Dataset split disjointness/determinism, leakage guarantees, transform behavior, image validation (\`test\_dataset.py\`)

\- End-to-end inference on synthetic checkpoints for both approaches (\`test\_inference.py\`)

Tests use a small synthetic MVTec-AD-shaped directory tree

(\`tests/test\_dataset.py::\_make\_fake\_mvtec\`), so the full suite runs without

the real (large, license-restricted) dataset present — useful for CI and for

verifying the pipeline before committing to a multi-GB download.

**## 19. Reproducibility**

\- All hyperparameters live in \`configs/\*.yaml\` — nothing is hardcoded in

  training scripts.

\- Every run's full config + final metrics are logged to

  \`results/experiments/\<name>\_\<timestamp>.json\` (\`src/evaluation/experiment\_log.py\`)

  — no external experiment-tracking service required.

\- \`src.utils.set\_seed()\` fixes Python/NumPy/PyTorch seeds and enables

  \`cudnn.deterministic\` before any data loading or model init.

\- Deterministic, hash-based dataset splitting (no shuffled-and-saved index

  files needed — same seed always reproduces the same split on any machine).

\- Environment: Python ≥ 3.10, PyTorch ≥ 2.2 (see \`pyproject.toml\` /

  \`requirements.txt\` for full pinned versions). CUDA version is whatever

  your installed PyTorch build targets; the project does not depend on a

  specific CUDA version since it never touches CUDA directly.

**## 20. Limitations**

\- Single MVTec AD category (\`bottle\`) — generalization to other categories

  (textures like \`grid\`/\`carpet\` vs. objects like \`bottle\`) is not yet

  measured; the pipeline is written to support any category via config but

  this hasn't been empirically validated across categories yet.

\- The classifier's labeled training set is small (\~90 images after the

  70/30 internal split of \`labeled\_val\`) — expect high run-to-run variance;

  this is exactly why the autoencoder approach exists as a comparison, and

  why results should be reported with multiple seeds, not a single run.

\- The convolutional autoencoder is a strong *\*baseline\** for unsupervised

  anomaly detection but is known in the literature to underperform

  feature-embedding methods like PatchCore/PaDiM, particularly on

  localization precision — see [Future Work]\(#21-future-work).

\- Pixel-level localization metrics depend on MVTec AD's provided masks,

  which mark the annotator's judgment of defect extent — not a physically

  exact ground truth.

\- No cross-category or cross-domain generalization testing (e.g. a model

  trained on \`bottle\` has not been tested on any other product).

**## 21. Future Work**

\- Implement the optional **\*\*PatchCore\*\*** anomaly detector (memory bank of

  pretrained-feature patch embeddings + nearest-neighbor scoring) as a

  stronger comparison point against the autoencoder — architecture slotted

  into \`src/models/advanced.py\`, currently a placeholder.

\- Implement the baseline models described in [Results]\(#11-results) (mean-image

  distance, classical-feature + linear classifier) for a complete

  baseline → CNN → autoencoder → advanced-method comparison chain.

\- Run the full ablation matrix and populate [Results]\(#11-results) and

  [Ablation Studies]\(#12-ablation-studies) with real numbers.

\- Extend to additional MVTec AD categories once the \`bottle\` pipeline is

  fully validated, to measure cross-category generalization.

\- Multi-seed reporting (mean ± std across ≥3 seeds) given the small dataset

  size, rather than single-run numbers.