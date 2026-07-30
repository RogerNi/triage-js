#! /bin/bash

if [ $# -lt 2 ]
then
echo "Usage: $0 <seed> <generate_batch_size> [args]"
exit 1
fi

seed=$1
generate_batch_size=$2
shift
shift

current_datetime=$(date +"%Y%m%d_%H%M%S")
script_name=qwen25_coder_7B_zero_shot_CoT

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -u train.py \
  --model_name=$script_name \
  --tb_dir=./tensorboard/ \
  --model_type=llama \
  --model_name_or_path=Qwen/Qwen2.5-Coder-7B-Instruct \
  --output_dir=./saved_models \
  --do_train \
  --do_test \
  --data_folder=../dataset_util \
  --feature_file_name=node_information_per_graph.csv \
  --epochs 0 \
  --block_size 1280 \
  --batch_size 2 \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --no_gnn \
  --use_causal_lm \
  --use_chat_template \
  --code_feature=per_package \
  --weight_decay 0.01 \
  --num_output_layers 1 \
  --preprompt "Please analyze the JavaScript code snippet below and determine if it contains any exploitable ACE or ACI vulnerabilities. In your response, include your reasoning related to this question. Then, on a new line at the end, provide only your final answer as either “Yes” or “No” with no additional text.\n\n" \
  --no_tokenization_in_dataloader \
  --generate_batch_size $generate_batch_size \
  --generate_look_at_last_n_tokens 5 \
  --seed $seed $@ 2>&1 | tee "logs/train_${current_datetime}_${script_name}.log"