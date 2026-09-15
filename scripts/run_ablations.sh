#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET="${1:-all}"
if [[ "$TARGET" == "classifier" || "$TARGET" == "all" ]]; then
    for cfg in configs/ablations/classifier_*.yaml; do
        echo "=== Running $cfg ==="
        python -m src.training.train_classifier --config "$cfg"
        echo ""
    done
fi
if [[ "$TARGET" == "autoencoder" || "$TARGET" == "all" ]]; then
    for cfg in configs/ablations/autoencoder_*.yaml; do
        echo "=== Running $cfg ==="
        python -m src.training.train_autoencoder --config "$cfg"
        echo ""
    done
fi
echo "All ablation runs logged to results/experiments/ (prefixed 'ablation_')."
echo "Checkpoints saved under models/ablations/ (production checkpoints untouched)."
echo "Aggregate and plot with: python -m src.evaluation.summarize_ablations"
