# Model Checkpoints

This directory holds trained model weights, produced by the training scripts
in `src/training/`. It is **not** pre-populated — checkpoints are large
binaries and are gitignored (see `.gitignore`); you generate them locally by
running training.

## Layout

```
models/
├── classifier/
│   └── best_model.pt      # produced by: python -m src.training.train_classifier
├── autoencoder/
│   └── best_model.pt      # produced by: python -m src.training.train_autoencoder
├── advanced/
│   └── best_model.pt      # produced by the optional PatchCore/PaDiM extension, if implemented
└── ablations/
    ├── classifier_finetune_frozen/best_model.pt
    ├── classifier_finetune_full/best_model.pt
    ├── classifier_aug_off/best_model.pt
    ├── autoencoder_latent64/best_model.pt
    ├── ...                 # one subdirectory per configs/ablations/*.yaml,
    │                       # named identically to that config's filename
    └── (never overwrites classifier/ or autoencoder/ above — see
        configs/ablations/*.yaml `checkpointing.save_dir`)
```

## What's inside a checkpoint

Each `best_model.pt` is a dict with:

- `model_state_dict` — the PyTorch state dict, loadable directly into a
  freshly constructed model of the matching architecture.
- `config` — the *entire* YAML config used for that training run, saved
  verbatim. This is what lets `src/inference.py` reconstruct the exact model
  architecture (image size, latent dim, fine-tune mode, etc.) without the
  caller needing to specify it again.
- `epoch` — the epoch at which this checkpoint was saved (i.e. the best
  epoch by the validation metric configured in `checkpointing.monitor_metric`).

## Selection policy

Both training scripts save a checkpoint **only** when a validation-set
metric improves (`val_f1` for the classifier, `val_reconstruction_error` for
the autoencoder) — never based on test-set performance. This is enforced in
code, not just by convention: the test split is loaded and evaluated exactly
once, after the best checkpoint has already been selected and reloaded. See
README "Data Leakage" for the full reasoning.
