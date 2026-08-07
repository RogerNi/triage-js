#!/usr/bin/env bash
# GNN-only path on the demo subset. Single GPU, no model download.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CUDA="${CUDA:-0}" "$HERE/run.sh" \
  python train.py --model_name=demo_gnn --model_type=llama --no_llm \
    --do_train --do_test --data_folder=dataset_demo \
    --feature_file_name=node_information_per_graph.csv --code_feature=per_package \
    --num_output_layers 1 --epochs 3 --batch_size 16 --learning_rate 1e-3 \
    --delete_checkpoint --seed 2025 --output_dir=_demo_out
