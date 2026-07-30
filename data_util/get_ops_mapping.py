import json
import torch
import numpy as np
from collections import Counter

# Load JSON data
with open("dataset/filtered_flows.json", "r") as f:
    data = json.load(f)

flows = data["flows"]

# Step 1: Reproduce the random training split
random_seed = 42
total_len = len(flows)
train_len = int(0.8 * total_len)
val_len = int(0.1 * total_len)
test_len = total_len - train_len - val_len

generator = torch.Generator().manual_seed(random_seed)
all_indices = torch.randperm(total_len, generator=generator).tolist()

train_indices = all_indices[:train_len]
val_indices = all_indices[train_len:train_len + val_len]
test_indices = all_indices[train_len + val_len:]

print(f"Train indices: {train_indices}")
print(f"Validation indices: {val_indices}")
print(f"Test indices: {test_indices}")

# Step 2: Count operations only in training flows
operation_counter = Counter()

for idx in train_indices:
    flow = flows[idx]
    provenance_tree = flow.get("provenance_tree", {})
    for node in provenance_tree.values():
        op = node.get("operation")
        if op:
            operation_counter[op] += 1

# Step 3: Build mapping for top 100 operations
most_common_ops = operation_counter.most_common(100)
ops_mapping = {op: idx for idx, (op, _) in enumerate(most_common_ops)}

# Output
print("ops_mapping =", ops_mapping)