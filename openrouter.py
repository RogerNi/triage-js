#! /usr/bin/env python

import os
import argparse
from openai import OpenAI
from datamodule import ProvenanceGraphDataModule as Prov
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import tqdm

def load_api_key():
    """Load OpenRouter API key from environment or fallback file."""
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key

    key_path = "openrouter.key"
    if os.path.exists(key_path):
        with open(key_path, "r") as f:
            return f.read().strip()

    raise RuntimeError("OpenRouter API key not found. Set OPENROUTER_API_KEY or provide openrouter.key file.")

# Argument parsing
parser = argparse.ArgumentParser(description="Run OpenRouter model on test dataset")
parser.add_argument("--model", type=str, required=True, help="Model name for OpenRouter (e.g., 'openai/o3-mini-high')")
parser.add_argument("--prompt_file", type=str, required=True, help="Path to the prompt text file")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

# Load API key
key = load_api_key()

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=key,
)

# Load prompt
with open(args.prompt_file, "r") as pf:
    prompt = pf.read()

# Load test data
datamodule = Prov(
    data_path="dataset",
    batch_size=2,
    seed=42,
    load_features=True,
    train_workers=0,
    code_feature="per_package",
    feature_file_name="node_information_per_graph.csv",
    use_chat_template=False,
    no_tokenization=True,
)

labels = []
predictions = []
count = 0

# Inference loop
for x in tqdm.tqdm(datamodule.test):
    label = x['labels'].argmax().item()

    completion = client.chat.completions.create(
        extra_body={},
        model=args.model,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": x['func']}],
        seed=args.seed,
    )

    response = completion.choices[0].message.content.strip()
    predictions.append(response)
    labels.append(label)
    count += 1

print(f"Total predictions: {count}")

# Write predictions to file
output_file = f"saved_llm_outputs/openrouter_predictions_{args.model.replace('/', '_')}.txt"
with open(output_file, "w") as f:
    for pred, label in zip(predictions, labels):
        f.write(f"{pred}\n")
        f.write(f"{label}\n")
        f.write(f"{'=' * 20}\n")

# Convert textual predictions to binary
binary_preds_strict = []
binary_preds_filtered = []
binary_labels_strict = []
binary_labels_filtered = []

invalid_count = 0
invalid_indices = []

for i, (pred, label) in enumerate(zip(predictions, labels)):
    response = pred.strip()
    
    if "Yes" in response:
        binary_preds_strict.append(1)
        binary_preds_filtered.append(1)
        binary_labels_strict.append(1 if label == 1 else 0)
        binary_labels_filtered.append(1 if label == 1 else 0)
        
    elif "No" in response:
        binary_preds_strict.append(0)
        binary_preds_filtered.append(0)
        binary_labels_strict.append(1 if label == 1 else 0)
        binary_labels_filtered.append(1 if label == 1 else 0)
        
    else:
        # Treat as incorrect in strict evaluation
        binary_preds_strict.append(0)
        binary_labels_strict.append(1 if label == 1 else 0)
        invalid_count += 1
        invalid_indices.append(i)

# Strict evaluation (invalids treated as "No")
f1_strict = f1_score(binary_labels_strict, binary_preds_strict)
precision_strict = precision_score(binary_labels_strict, binary_preds_strict)
recall_strict = recall_score(binary_labels_strict, binary_preds_strict)
accuracy_strict = accuracy_score(binary_labels_strict, binary_preds_strict)

# Filtered evaluation (invalids excluded)
if binary_preds_filtered:
    f1_filtered = f1_score(binary_labels_filtered, binary_preds_filtered)
    precision_filtered = precision_score(binary_labels_filtered, binary_preds_filtered)
    recall_filtered = recall_score(binary_labels_filtered, binary_preds_filtered)
    accuracy_filtered = accuracy_score(binary_labels_filtered, binary_preds_filtered)
else:
    f1_filtered = precision_filtered = recall_filtered = accuracy_filtered = float('nan')

# Print results
print(f"Total predictions: {len(predictions)}")
print(f"Invalid responses (neither Yes nor No): {invalid_count}")

print("\n[Strict Evaluation] (Invalids treated as incorrect):")
print(f"F1: {f1_strict:.4f}")
print(f"Precision: {precision_strict:.4f}")
print(f"Recall: {recall_strict:.4f}")
print(f"Accuracy: {accuracy_strict:.4f}")

print("\n[Filtered Evaluation] (Only valid Yes/No responses):")
print(f"F1: {f1_filtered:.4f}")
print(f"Precision: {precision_filtered:.4f}")
print(f"Recall: {recall_filtered:.4f}")
print(f"Accuracy: {accuracy_filtered:.4f}")