#!/usr/bin/env bash
# Run any command inside the triage-js container with the flags the pipeline
# needs. The artifact code and the demo data (dataset_demo/) are baked into the
# image, so only a HuggingFace cache is mounted (to avoid re-downloading models).
#
#   * --gpus all                 : expose GPUs (pick which with CUDA=...)
#   * --ipc=host                 : shared memory for NCCL (required for multi-GPU)
#   * -v $HF_CACHE:...           : persist downloaded model weights across runs
#   * wandb                      : disabled inside the image (self-contained)
#
# Usage:
#   CUDA=0    docker/run.sh python train.py --no_llm ...        # single GPU
#   CUDA=0,1  docker/run.sh python train.py --optimize_llm no_quant ...   # 2 GPUs
#
# Env overrides: IMAGE (default triage-js:latest), CUDA (default 0),
#                SHM (default 16g), HF_CACHE (default $HOME/hf_cache).
set -euo pipefail

IMAGE="${IMAGE:-triage-js:latest}"
CUDA="${CUDA:-0}"
SHM="${SHM:-16g}"
HF_CACHE="${HF_CACHE:-$HOME/hf_cache}"

mkdir -p "$HF_CACHE"

set -x
exec docker run --rm --gpus all \
  -e CUDA_VISIBLE_DEVICES="$CUDA" \
  --ipc=host --shm-size="$SHM" \
  -v "$HF_CACHE":/root/.cache/huggingface \
  "$IMAGE" "$@"
