#!/bin/bash
#SBATCH --job-name=lin-probe
#SBATCH --output=slurm_logs/slurm_%A.out     # Output log: job ID (%A) and array index (%a)
#SBATCH --error=slurm_logs/slurm_%A.err      # Error log
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

# Get timestamp for the current experiment batch
timestamp=$(date +%Y%m%d_%H%M%S)
script_name=llama31_8B_cls_head_only
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
    --model_name_or_path=meta-llama/Llama-3.1-8B \
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
    --optimize_llm none \
    --weight_decay 0.01 \
    --num_output_layers 1 \
    --delete_checkpoint \
    --seed $seed "$@" 2>&1 | tee "$logdir/train_${script_name}_${seed}.log"
done

echo "Logs saved to: $logdir"