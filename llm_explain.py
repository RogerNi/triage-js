import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BitsAndBytesConfig, EarlyStoppingCallback, pipeline, AutoModel
from torch.utils.data import Dataset
from datamodule import ProvenanceGraphDataModule as Prov
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import random
import numpy as np

import os
from datetime import datetime

import shap

from tqdm import tqdm

from model import MyModelForSequenceClassification, GNNModel
import argparse, json


print(f"Number of GPUs available: {torch.cuda.device_count()}")

# llm_model_name = "codellama/CodeLlama-7b-Instruct-hf"
llm_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

seed = 2025
random.seed(seed)  # Python's random module
np.random.seed(seed)  # NumPy
torch.manual_seed(seed)  # PyTorch (CPU)
# torch.cuda.manual_seed(seed)  # PyTorch (GPU)
torch.cuda.manual_seed_all(seed)  # All GPUs
# torch.backends.cudnn.deterministic = True  # CuDNN
# torch.backends.cudnn.benchmark = False  # CuDNN
# torch.use_deterministic_algorithms(True)


# Example custom DataLoader-like dataset
class CustomTextDataset(Dataset):
    def __init__(self, dataloader, tokenizer, max_length=1024):
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # Preload all data for Hugging Face Trainer
        for texts_batch, labels_batch, _ in dataloader:
            tokenized = self.tokenizer(
                texts_batch,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            
            labels_tensor = torch.tensor(labels_batch, dtype=torch.long) 
            
            labels_one_hot = torch.nn.functional.one_hot(
                labels_tensor, num_classes=2
            ).float()  # Ensure one-hot labels are float for compatibility

            # Extend samples with tokenized data and one-hot encoded labels
            self.samples.extend(
                [{"input_ids": t, "attention_mask": a, "labels": l} for t, a, l in zip(
                    tokenized["input_ids"], tokenized["attention_mask"], labels_one_hot
                )]
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
class CustomTextDatasetMultiDataModule(CustomTextDataset):
    def __init__(self, dataloaders, tokenizer, max_length=1024):
        self.dataloaders = dataloaders
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # Preload all data for Hugging Face Trainer
        for dataloader in dataloaders:
            for texts_batch, labels_batch, gids in dataloader:
                tokenized = self.tokenizer(
                    texts_batch,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                labels_tensor = torch.tensor(labels_batch, dtype=torch.long)

                labels_one_hot = torch.nn.functional.one_hot(
                    labels_tensor, num_classes=2
                ).float()
                
                self.samples.extend(
                    [{"input_ids": t, "attention_mask": a, "labels": l, "gids": g} for t, a, l, g in zip(
                        tokenized["input_ids"], tokenized["attention_mask"], labels_one_hot, gids
                    )]
                )  

class FullNoTokenizeDataset(CustomTextDataset):
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.samples = []
        for dataloader in dataloaders:
            for texts_batch, labels_batch, gids in dataloader:
                self.samples.extend(
                    [{"text": t, "labels": l, "gids": g} for t, l, g in zip(texts_batch, labels_batch, gids)]
                )
    
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 2 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1087, 553], device=model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    labels = labels.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# Initialize tokenizer and dataset
datamodule = Prov(
    data_path="dataset",
    batch_size=4,
    seed=0,
    load_features=True,
    undersample=None,
    # oversample="v1.0",
    block_size=1024,
    # preprompt="Does the following JavaScript code snippet have an ACE or ACI vulnerability?",
    train_workers=0,
    code_feature="per_package",
    feature_file_name="node_information_per_graph.csv",
    no_tokenization=True,
    return_dict=False,
)

tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
# Use eos_token as pad_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = CustomTextDataset(datamodule.train_dataloader(), tokenizer)
eval_dataset = CustomTextDataset(datamodule.val_dataloader(), tokenizer)
test_dataset = CustomTextDataset(datamodule.test_dataloader(), tokenizer)

# check dataset
print(len(dataset))
print(dataset[0])

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Define the model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

llm_model = AutoModel.from_pretrained(
    llm_model_name,
    quantization_config=bnb_config,
    # device_map="auto",
)

llm_model.config.pad_token_id = tokenizer.pad_token_id
model = MyModelForSequenceClassification(
        llm_model=llm_model,
        gnn_model_t=GNNModel,
        gnn_encoder=None,
        global_args=argparse.Namespace(**{
            "no_gnn": True,
            "dropout": 0.1,
            "optimize_llm": "none",
            "no_llm": False,
            "num_output_layers": 1}),
        class_weights=datamodule.get_train_class_num(),
    )

model.to(dtype=torch.float32) # for ds-distill-qwen-7B-cls-head-only

# Path to directory containing model shards
# checkpoint_dir = "saved_models/2025-02-27_16-54-18_PID65746/checkpoint-1200" # ds-distill-qwen-7B-full
checkpoint_dir = "saved_models/2025-02-27_14-57-25_PID3369604/checkpoint-300" # ds-distill-qwen-7B-cls-head-only

# Identify the first shard (it typically contains the weight map)
weight_map_file = os.path.join(checkpoint_dir, "pytorch_model.bin.index.json")

if os.path.exists(weight_map_file):
    # Load the index JSON file that maps layer names to shard files
    with open(weight_map_file, "r") as f:
        index_data = json.load(f)

    # Get a list of all unique .bin shard files
    shard_files = list(set(index_data["weight_map"].values()))

    # Initialize an empty state dictionary
    merged_state_dict = {}

    # Load and merge all shards
    for shard in shard_files:
        shard_path = os.path.join(checkpoint_dir, shard)
        print(f"Loading {shard_path}...")
        shard_state_dict = torch.load(shard_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        merged_state_dict.update(shard_state_dict)

    print("All shards loaded and merged successfully!")

    # Load the merged state_dict into the model
    model.load_state_dict(merged_state_dict)
else:
    # otherwise directly load binary file
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin"), map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

model.config.pad_token_id = tokenizer.pad_token_id


# lora_config = LoraConfig(
#     r=8, lora_alpha=8, target_modules=['score.weight'], lora_dropout=0.1, bias="none", task_type=TaskType.SEQ_CLS,
# )

# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

# Get current timestamp and process ID
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
pid = os.getpid()  # Get current process ID
output_dir = f"./simple_llm_results/{timestamp}_PID{pid}"
os.makedirs(output_dir, exist_ok=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir, 
    evaluation_strategy="steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    save_strategy="best",
    # save_steps=250,                         # Save every 500 steps
    save_total_limit=10,
    load_best_model_at_end=True,             # Reload the best model at the end of training
    metric_for_best_model="eval_loss",       # Use evaluation loss to determine the best model
    greater_is_better=False,                 # Lower evaluation loss is better
    logging_steps=50,
    eval_steps=150,
    learning_rate=1e-5,
    # fp16=True, 
    bf16=True,
    logging_dir="./logs",
    max_grad_norm=1.0,
    save_only_model=True,
    weight_decay=0.01,
)

# Define the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # eval_dataset=eval_dataset,  # Using the same dataset for simplicity
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # Add the metrics function here
)

# instead of training, load model from checkpoint

print("Interpreting model...")
pipeline = pipeline(
    "text-classification", model=model, tokenizer=tokenizer, return_all_scores=True
)
# Create an output directory for the HTML files
output_dir = f"{output_dir}/explanations"
os.makedirs(output_dir, exist_ok=True)

explainer = shap.Explainer(pipeline)


# Process each input text, generate attributions, and output colored HTML
# for idx, inputs in tqdm(enumerate(FullNoTokenizeDataset([datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()])), total=len(dataset)+len(eval_dataset)+len(test_dataset)):
for idx, inputs in tqdm(enumerate(FullNoTokenizeDataset([datamodule.test_dataloader()])), total=len(test_dataset)):
    try:
        text = inputs["text"]
        label = inputs["labels"]
        gid = inputs["gids"]
        # Compute the explanation (word attributions)
        shap_values = explainer([text])
        
        print("Shap values: ", shap_values)
            
        html_explanation = shap.plots.text(shap_values[0], display=False, grouping_threshold=0.05)
        html_explanation += f"<p>Label: {label}</p>"

        with open(f"{output_dir}/shap_explanation_{gid}.html", "w") as f:
            f.write(html_explanation)
    except Exception as e:
        print(f"Error processing sample {idx}: {e}, skipping...")

print("Proceeding to test dataset...")
# Evaluate on the test dataset
test_metrics = trainer.evaluate(eval_dataset=test_dataset)

# Print metrics
print("Test Metrics:", test_metrics)

# predict on all 3 splits
full_dataset = CustomTextDatasetMultiDataModule([datamodule.train_dataloader(), datamodule.val_dataloader(), datamodule.test_dataloader()], tokenizer)
predictions = trainer.predict(full_dataset)
# get gids from full_dataset
gids = [sample["gids"] for sample in full_dataset.samples]
# sort predictions by gids
sorted_predictions = [x for _, x in sorted(zip(gids, predictions.predictions), key=lambda pair: pair[0])]

# write to file, one element per line
with open(f"{output_dir}/predictions.txt", "w") as f:
    for pred in sorted_predictions:
        f.write(f"{pred[0]}\t{pred[1]}\n")
