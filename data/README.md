# Dataset: MVTec AD — `bottle` category

This project uses the **MVTec Anomaly Detection (MVTec AD)** dataset, a widely used
benchmark for unsupervised/semi-supervised industrial anomaly detection, introduced in:

> Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019).
> *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.*
> CVPR 2019.

## License

MVTec AD is released by MVTec Software GmbH for **non-commercial, scientific and
educational use only** (CC BY-NC-SA 4.0). Redistribution of the raw dataset is not
permitted — this repository does **not** ship the images; you must download them
yourself from the official source below and agree to the license terms.

## Download

1. Go to: https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Register (free) and download either:
   - The full dataset (`mvtec_anomaly_detection.tar.xz`, ~4.9 GB), **or**
   - Just the `bottle` category if the site offers per-category downloads.
3. Extract it so the layout below is satisfied.

## Required directory layout

After extraction, this is exactly what the code in `src/data/dataset.py` expects:

```
data/mvtec_ad/bottle/
├── train/
│   └── good/                     # 209 normal images — used ONLY for training
│       ├── 000.png
│       └── ...
├── test/
│   ├── good/                     # 20 normal images — held out for evaluation
│   ├── broken_large/             # 20 defective images
│   ├── broken_small/             # 22 defective images
│   └── contamination/            # 21 defective images
└── ground_truth/
    ├── broken_large/             # binary pixel-level masks (*_mask.png)
    ├── broken_small/
    └── contamination/
```

## Documented dataset statistics (bottle category)

These numbers are taken directly from the MVTec AD paper / dataset metadata —
**not** re-derived or estimated. They will be re-verified programmatically in
`notebooks/01_dataset_exploration.ipynb` once the data is downloaded, and any
discrepancy will be corrected in this file.

| Split | Class            | # Images | Has pixel mask? |
|-------|------------------|----------|------------------|
| train | good (normal)    | 209      | n/a              |
| test  | good (normal)    | 20       | n/a              |
| test  | broken_large     | 20       | yes              |
| test  | broken_small     | 22       | yes              |
| test  | contamination    | 21       | yes              |
| **Total** |              | **292**  |                  |

- **Original image resolution:** 900×900 px, RGB, PNG.
- **Working resolution for this project:** 224×224 (resized) — matches the input
  size expected by ImageNet-pretrained backbones (ResNet18/EfficientNet) and keeps
  training tractable on CPU/single-GPU hardware. This is a documented design
  choice, not a dataset property — see ablation on image resolution in
  `results/experiments/`.
- **Class balance (test set):** 20 normal vs. 63 defective (3 defect subtypes) —
  i.e. the test set is deliberately *not* balanced 50/50, which is representative
  of the class-imbalance problem discussed in the README.
- **Train set contains only normal ("good") images** — by design, per the MVTec AD
  protocol. This is why Approach B (autoencoder) trains exclusively on `good`
  images, and why Approach A (supervised classifier) needs a custom train/val split
  carved out of the *test* set's labeled images (with strict leakage controls —
  see `docs`/README section "Data Leakage" for how this split is constructed and
  why the true test subset used for final reporting is never touched during that
  process).

## Why `bottle`

`bottle` is one of the most frequently reported categories in anomaly-detection
papers (PatchCore, PaDiM, DRAEM, etc.), which makes it possible to sanity-check
this project's ROC-AUC against published numbers. It's a single centered object
(not a texture), which makes both Grad-CAM and reconstruction-error heatmaps easy
to interpret visually.

## Verifying your download

Once extracted, run:

```bash
python -m src.data.dataset --verify --root data/mvtec_ad --category bottle
```

This checks the expected folder structure and image counts against the table
above and will raise a clear error if anything is missing or mismatched.
