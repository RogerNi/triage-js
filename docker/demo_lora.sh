#!/usr/bin/env bash
# LLM LoRA fine-tuning on the demo subset. Single GPU (4-bit + LoRA).
# Downloads the LLM on first run (cached afterwards).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B}"
CUDA="${CUDA:-0}" "$HERE/run.sh" \
  python train.py --model_name=demo_lora --model_type=llama \
    --model_name_or_path="$MODEL" --optimize_llm full --no_gnn \
    --lora_rank 128 --lora_alpha 128 \
    --do_train --do_test --data_folder=dataset_demo \
    --feature_file_name=node_information_per_graph.csv --code_feature=per_package \
    --num_output_layers 1 --block_size 512 --batch_size 2 --learning_rate 1e-4 \
    --epochs 1 --delete_checkpoint --seed 2025 --output_dir=_demo_out
