from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
              "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
              "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 
              "meta-llama/Llama-3.1-8B-Instruct", 
              "Qwen/Qwen2.5-Coder-7B-Instruct", 
              "Qwen/Qwen2.5-Coder-14B-Instruct"]

for model_n in model_name:
    model = AutoModelForSequenceClassification.from_pretrained(model_n)
    tokenizer = AutoTokenizer.from_pretrained(model_n)

