#!/bin/bash

# Get process ID and timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
logdir="logs/train_${timestamp}_zero_shot"

# Create log directory
mkdir -p "$logdir"

for seed in 2025 2026 2027 2028 2029; do
# Loop through pooling options
for model in "anthropic/claude-sonnet-4.5" "openai/gpt-5" "openai/o4-mini-high" "openai/gpt-4.1" "google/gemini-2.5-pro" "deepseek/deepseek-r1-0528"; do
    python openrouter.py --model "$model" --prompt_file prompts/zero-shot --seed "$seed" | tee "$logdir/zero_shot_${model//\//_}_${seed}.log" 2>&1
done
done
echo "Logs saved to: $logdir"