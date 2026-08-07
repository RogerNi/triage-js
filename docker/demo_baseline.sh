#!/usr/bin/env bash
# Classical ML baseline (Random Forest / XGBoost / LogReg / SVM) on the demo
# subset. CPU-only.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CUDA="${CUDA:-0}" "$HERE/run.sh" \
  python baseline.py --dataset full --data_folder dataset_demo \
    --pooling avg --seed 2025
