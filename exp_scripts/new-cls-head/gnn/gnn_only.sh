#!/bin/bash
#SBATCH --job-name=gnn-only
#SBATCH --output=slurm_logs/slurm_%A_%a.out      # Output log per array task
#SBATCH --error=slurm_logs/slurm_%A_%a.err       # Error log per array task
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --array=0-4
#SBATCH --requeue

set -euo pipefail

# Define the seed list
SEEDS=(2025 2026 2027 2028 2029)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Get timestamp for the current experiment batch
timestamp=$(date +%Y%m%d_%H%M%S)
script_name=gnn_only
logdir="logs/train_${timestamp}_${script_name}"

# Create the log directory
mkdir -p "$logdir"

echo "Running training with seed=$SEED"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -u train.py \
  --model_name="$script_name" \
  --tb_dir=./tensorboard/ \
  --model_type=llama \
  --output_dir=./saved_models \
  --do_train \
  --do_test \
  --data_folder=dataset \
  --epochs 150 \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --no_llm \
  --num_output_layers 1 \
  --dropout 0 \
  --weight_decay 1e-1 \
  --eval_steps 150 \
  --seed "$SEED" "$@" 2>&1 | tee "$logdir/train_${script_name}_${SEED}.log"

echo "Logs saved to: $logdir"
