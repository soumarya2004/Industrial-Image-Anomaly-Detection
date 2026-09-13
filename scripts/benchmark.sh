#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CATEGORY="${1:-bottle}"
SAMPLE_DIR="data/mvtec_ad/${CATEGORY}/test/good"

if [[ ! -d "$SAMPLE_DIR" ]]; then
    echo "Sample directory not found: $SAMPLE_DIR"
    echo "See data/README.md for dataset setup instructions."
    exit 1
fi
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
"
echo ""
echo "=== Classifier: single-image + batch latency ==="
time python -m src.inference --image_dir "$SAMPLE_DIR" --model classifier
echo ""
echo "=== Autoencoder: single-image + batch latency ==="
echo "(requires --threshold — pass as second arg, e.g. ./scripts/benchmark.sh bottle 0.0123)"
if [[ -n "${2:-}" ]]; then
    time python -m src.inference --image_dir "$SAMPLE_DIR" --model autoencoder --threshold "$2"
fi
