#!/bin/bash

# Get process ID and timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
logdir="logs/train_${timestamp}_baselines"


# Create log directory
mkdir -p "$logdir"

# Loop through pooling options
# for pooling in attn max avg; do
for pooling in avg; do
  for seed in 2025 2026 2027 2028 2029; do
    echo "Running with pooling=$pooling"
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python baseline.py --pooling "$pooling" --seed "$seed" --map_random_seed_to_ckpt 2>&1 | tee "$logdir/baseline_${pooling}_${seed}.log"
  done
done

echo "Logs saved to: $logdir"