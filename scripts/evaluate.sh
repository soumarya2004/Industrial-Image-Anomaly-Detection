#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CATEGORY="${1:-bottle}"
TEST_DIR="data/mvtec_ad/${CATEGORY}/test"
if [[ ! -d "$TEST_DIR" ]]; then
    echo "Test directory not found: $TEST_DIR"
    echo "See data/README.md for dataset setup instructions."
    exit 1
fi
echo "=== Batch inference: classifier ==="
for defect_dir in "$TEST_DIR"/*/; do
    python -m src.inference --image_dir "$defect_dir" --model classifier
done
echo ""
echo "=== Batch inference: autoencoder ==="
echo "NOTE: pass the threshold reported at the end of autoencoder training,"
echo "      e.g.: ./scripts/evaluate.sh bottle --threshold 0.0123"
if [[ "${2:-}" == "--threshold" && -n "${3:-}" ]]; then
    for defect_dir in "$TEST_DIR"/*/; do
        python -m src.inference --image_dir "$defect_dir" --model autoencoder --threshold "$3"
    done
else
    echo "Skipping autoencoder batch inference (no --threshold provided)."
fi
