#!/usr/bin/env bash
# Zero-shot classification with a local instruct LLM: no training (--epochs 0),
# the model is prompted (prompts/zero-shot) and generates "Yes"/"No" per sample.
# Single GPU. Downloads the instruct model on first run (cached afterwards).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
CUDA="${CUDA:-0}" "$HERE/run.sh" \
  python train.py --model_name=demo_zeroshot --model_type=llama \
    --model_name_or_path="$MODEL" --optimize_llm none --no_gnn \
    --epochs 0 --do_train --do_test --data_folder=dataset_demo \
    --feature_file_name=node_information_per_graph.csv --code_feature=per_package \
    --block_size 512 --preprompt=prompts/zero-shot \
    --use_sample_to_generate --use_causal_lm --no_tokenization_in_dataloader \
    --use_chat_template --generate_batch_size 2 \
    --seed 2025 --output_dir=_demo_out
