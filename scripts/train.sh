#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET="${1:-both}"
if [[ "$TARGET" == "baselines" || "$TARGET" == "both" ]]; then
    echo "=== Running Phase 3 baselines (mean-image + HOG/logreg) ==="
    python -m src.training.run_baselines --config configs/classifier.yaml
fi
if [[ "$TARGET" == "classifier" || "$TARGET" == "both" ]]; then
    echo "=== Training supervised classifier (Approach A) ==="
    python -m src.training.train_classifier --config configs/classifier.yaml
fi
if [[ "$TARGET" == "autoencoder" || "$TARGET" == "both" ]]; then
    echo "=== Training autoencoder anomaly detector (Approach B) ==="
    python -m src.training.train_autoencoder --config configs/autoencoder.yaml
fi
echo "Done. Checkpoints in models/, experiment logs in results/experiments/."
