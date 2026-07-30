#!/bin/bash
#SBATCH --job-name=zero-shot
#SBATCH --output=slurm_logs/slurm_%A.out     # Output log: job ID (%A)
#SBATCH --error=slurm_logs/slurm_%A.err      # Error log
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

# Get timestamp for the current experiment batch
timestamp=$(date +%Y%m%d_%H%M%S)
script_name=llama_8B_zero_shot
logdir="logs/train_${timestamp}_${script_name}"

# Create the log directory
mkdir -p "$logdir"

# Loop over seeds
for seed in 2025 2026 2027 2028 2029; do
# for seed in 2025; do
  echo "Running training with seed=$seed"

  export CUBLAS_WORKSPACE_CONFIG=:4096:8

  python -u train.py \
    --model_name=$script_name \
    --tb_dir=./tensorboard/ \
    --model_type=llama \
    --model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
    --output_dir=./saved_models \
    --epochs 0 \
    --do_train \
    --do_test \
    --data_folder=dataset \
    --feature_file_name=node_information_per_graph.csv \
    --block_size 1280 \
    --evaluate_during_training \
    --no_gnn \
    --code_feature=per_package \
    --optimize_llm none \
    --preprompt=prompts/zero-shot \
    --use_sample_to_generate \
    --use_causal_lm \
    --no_tokenization_in_dataloader \
    --use_chat_template \
    --generate_batch_size=2 \
    --seed $seed "$@" 2>&1 | tee "$logdir/train_${script_name}_${seed}.log"
done

echo "Logs saved to: $logdir"