#!/bin/bash
#SBATCH --job-name=lora
#SBATCH --output=slurm_logs/slurm_%A_%a.out      # Output log per array task
#SBATCH --error=slurm_logs/slurm_%A_%a.err       # Error log per array task
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --array=0-24                              # 5 seeds × 5 ranks = 25 jobs
#SBATCH --requeue

# Define seeds and LoRA ranks
SEEDS=(2025 2026 2027 2028 2029)
LORA_RANKS=(16 32 64 128 256)

# Compute current seed and LoRA rank from task ID
seed_idx=$(( SLURM_ARRAY_TASK_ID / ${#LORA_RANKS[@]} ))
rank_idx=$(( SLURM_ARRAY_TASK_ID % ${#LORA_RANKS[@]} ))

SEED=${SEEDS[$seed_idx]}
LORA_RANK=${LORA_RANKS[$rank_idx]}

# Get timestamp for the current experiment batch
timestamp=$(date +%Y%m%d_%H%M%S)
script_name=qwen_7B_lora_hp_tune_${LORA_RANK}
logdir="logs/train_${timestamp}_${script_name}_seed${SEED}"
mkdir -p "$logdir"

echo "Running training with seed=$SEED and lora_rank=$LORA_RANK"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -u train.py \
  --model_name=$script_name \
  --tb_dir=./tensorboard/ \
  --model_type=llama \
  --model_name_or_path=Qwen/Qwen2.5-Coder-7B \
  --output_dir=./saved_models \
  --do_train \
  --do_test \
  --data_folder=dataset \
  --feature_file_name=node_information_per_graph.csv \
  --epochs 3 \
  --block_size 1280 \
  --batch_size 2 \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --no_gnn \
  --code_feature=per_package \
  --optimize_llm full \
  --weight_decay 0.01 \
  --num_output_layers 1 \
  --lora_rank $LORA_RANK \
  --lora_alpha $((LORA_RANK / 2)) \
  --delete_checkpoint \
  --seed $SEED $@ 2>&1 | tee "$logdir/train_${script_name}_seed${SEED}.log"

echo "Logs saved to: $logdir"