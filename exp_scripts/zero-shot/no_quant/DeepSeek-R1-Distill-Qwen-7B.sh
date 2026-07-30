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
script_name=ds_r1_distill_qwen_7B_zero_shot_no_quant

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -u train.py \
  --model_name=$script_name \
  --tb_dir=./tensorboard/ \
  --model_type=llama \
  --model_name_or_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --output_dir=./saved_models \
  --do_train \
  --do_test \
  --data_folder=dataset \
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
  --preprompt "Answer ONLY with ‘Yes’ or ‘No’ and provide no additional text:\nDoes the following JavaScript code snippet contain any exploitable ACE or ACI vulnerabilities?\n\n" \
  --no_tokenization_in_dataloader \
  --generate_batch_size $generate_batch_size \
  --optimize_llm no_quant \
  --seed $seed $@ 2>&1 | tee "logs/train_${current_datetime}_${script_name}.log"