#!/bin/bash
#SBATCH --job-name=full
#SBATCH --output=slurm_logs/slurm_%A_%a.out      # Output log per array task
#SBATCH --error=slurm_logs/slurm_%A_%a.err       # Error log per array task
#SBATCH --gres=gpu:2
#SBATCH --time=8:00:00
#SBATCH --array=0-4                              # 5 seeds: index 0–4

# Define the seed list
SEEDS=(2025 2026 2027 2028 2029)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Timestamp and naming
timestamp=$(date +%Y%m%d_%H%M%S)
script_name=ds_distill_llama_8B_full_gnn
logdir="logs/train_${timestamp}_${script_name}"
mkdir -p "$logdir"

echo "Running training with seed=$SEED"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

  python -u train.py \
    --model_name=$script_name \
    --tb_dir=./tensorboard/ \
    --model_type=llama \
    --model_name_or_path=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
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
    --code_feature=per_package \
    --optimize_llm no_quant \
    --weight_decay 0.01 \
    --num_output_layers 1 \
    --delete_checkpoint \
    --seed $SEED $@ 2>&1 | tee "$logdir/train_${script_name}_${SEED}.log"

echo "Logs saved to: $logdir"