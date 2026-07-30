import os
import itertools
import logging
import torch
import pandas as pd
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from xgboost import XGBClassifier
import sys
from datamodule import ProvenanceGraphDataModule
import argparse

from sklearn.tree import export_graphviz
import torch.nn.functional as F

from torch_geometric.nn import GlobalAttention
from torch_geometric.data import Batch

import numpy as np
import time

import random

# Setup logger
logger = logging.getLogger(__name__)

# Define model
allfeats = ["ops", "sink_t", "vult_t", "tainted"]

pd.set_option('display.max_rows', None)  # Show all rows

def one_hot_encode(input_tensor, num_classes):
    return F.one_hot(input_tensor, num_classes=num_classes).float()

def apply_max_pooling(one_hot_tensor):
    return torch.max(one_hot_tensor, dim=0)[0]  # Apply max pooling and return the values (ignore indices)

def apply_average_pooling(one_hot_tensor):
    return torch.mean(one_hot_tensor, dim=0)

# Define the number of classes for each feature (adjust based on your data)
feature_classes = {
    "ops": 102,        # Number of classes for "ops" feature
    "sink_t": 102,     # Number of classes for "sink_t" feature
    "vult_t": 102,     # Number of classes for "vult_t" feature
    "tainted": 102     # Number of classes for "tainted" feature
}


class FlowGNNGGNNModule(nn.Module):
    def __init__(
        self,
        feat,
        input_dim,
        hidden_dim,
        n_steps,
        num_output_layers,
        label_style="graph",
        concat_all_absdf=False,
        encoder_mode=False,
        *,
        debug=False,              # <-- NEW: turn prints on/off
        sample_size=8,            # <-- NEW: how many elements to print
        **kwargs,
    ):
        super().__init__()
        feat = "_" if "_" in feat else feat
        self.feature_keys = {"feature": feat}
        self.input_dim = input_dim
        self.concat_all_absdf = concat_all_absdf
        self.debug = debug
        self.sample_size = sample_size

        embedding_dim = hidden_dim
        self.allfeats = ["ops", "sink_t", "vult_t", "tainted"]

        if self.concat_all_absdf:
            self.all_embeddings = nn.ModuleDict(
                {of: nn.Embedding(input_dim, embedding_dim) for of in self.allfeats}
            )
            embedding_dim *= len(self.allfeats)
        else:
            self.embedding = nn.Embedding(input_dim, embedding_dim)

        output_in_size = embedding_dim
        self.out_dim = output_in_size

        if label_style == "graph":
            pooling_gate_nn = nn.Linear(output_in_size, 1)
            self.pooling = GlobalAttention(pooling_gate_nn)

    # ------------------------------------------------------------------ #
    # Helper to print a quick, partial view of a tensor                   #
    # ------------------------------------------------------------------ #
    def _peek(self, name: str, tensor: torch.Tensor):
        if not self.debug:
            return
        flat = tensor.detach().view(-1)          # flatten for convenience
        sample = flat[: self.sample_size].cpu()  # take first few elements
        print(f"{name}: shape={tuple(tensor.shape)}, sample={sample.tolist()}")

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(self, graph: Batch, extrafeats=None):
        # ---- 1. raw feature indices ----------------------------------- #
        if self.concat_all_absdf:
            feats = []
            for of in self.allfeats:
                feat_idx = getattr(graph, of).to(graph.ops.device)
                self._peek(f"{of}_idx", feat_idx)
                feats.append(self.all_embeddings[of](feat_idx))
            feat_embed = torch.cat(feats, dim=-1)
            self._peek("concat_embed", feat_embed)
        else:
            feat_embed = self.embedding(graph.x)
            self._peek("feat_embed", feat_embed)

        # ---- 2. pooled graph representation --------------------------- #
        pooled = self.pooling(feat_embed, graph.batch)
        self._peek("pooled", pooled)

        return pooled

        
        


def load_modified_model(ckpt_path, input_dim):
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    hidden_dim = 32
    n_steps = 5

    model = FlowGNNGGNNModule(
        "_",
        input_dim,
        hidden_dim,
        n_steps,
        num_output_layers=1,
        encoder_mode=True,
        concat_all_absdf=True,
        debug=True,  # optional debug mode
    )

    # Strip full prefix: "gnn_model.flowgnn_encoder."
    full_prefix = "gnn_model.flowgnn_encoder."
    checkpoint_state_dict = {
        k[len(full_prefix):]: v
        for k, v in checkpoint.items()
        if k.startswith(full_prefix)
    }

    # Adjust pooling weight shape if needed
    gate_key = "pooling.gate_nn.weight"
    if gate_key in checkpoint_state_dict:
        expected_shape = model.pooling.gate_nn.weight.shape
        actual_shape = checkpoint_state_dict[gate_key].shape
        if actual_shape != expected_shape:
            print(f"Adjusting {gate_key} from {actual_shape} to {expected_shape}")
            checkpoint_state_dict[gate_key] = checkpoint_state_dict[gate_key][:, -expected_shape[1]:]

    # Load and report missing/unexpected keys
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state_dict, strict=False)
    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)

    return model

def feature_masking(X, mask):
    X = X.copy()
    X[:, mask] = 0
    return X

def evaluate_model(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
    }
    return metrics


def main(args):
    
    
    seed = args.seed
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # All GPUs
    torch.backends.cudnn.deterministic = True  # CuDNN
    torch.backends.cudnn.benchmark = False  # CuDNN
    torch.use_deterministic_algorithms(True)

    # Load dataset
    if args.dataset == "full":
        data_module = ProvenanceGraphDataModule(
            data_path="dataset",
            batch_size=256,
            seed=42,
            sample=-1,
            sample_mode=False,
            train_workers=1,
            val_workers=0,
            test_workers=0,  # Default value
            load_features=False,  # Commented out in YAML, so assumed False
            use_random_weighted_sampler=False,  # Default value
            return_dict=False,
            shuffle_train=False, # for reproducibility
        )
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    if args.pooling == "attn":
        if not args.map_random_seed_to_ckpt:
            ckpt_path = "old_saved_models/2025-07-09_10-22-55_PID4075777/checkpoint-600/pytorch_model.bin"
        else:
            # read a folder and order the folders by name
            ckpt_root = "saved_models_gnn"
            ckpt_folders = sorted(os.listdir(ckpt_root))
            ckpt_path = os.path.join(ckpt_root, ckpt_folders[args.seed-2025])
            # load the last checkpoint in the folder
            ckpt_files = sorted(os.listdir(ckpt_path))
            ckpt_path = os.path.join(ckpt_path, ckpt_files[-1], "pytorch_model.bin")
            print(f"Loading model from {ckpt_path} for seed {args.seed}")
        model = load_modified_model(ckpt_path, data_module.input_dim())
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        
    with torch.no_grad():
        # Collect outputs for all splits
        splits = {
            "Train": data_module.train_dataloader(),
            "Validation": data_module.val_dataloader(),
            "Test": data_module.test_dataloader(),
        }
        outputs_map = {split: ([], [], []) for split in splits}

        # Generate embeddings and labels for all splits
        for split_name, dataloader in splits.items():
            outputs_list, labels_list, gidx_list = outputs_map[split_name]
            for batch in dataloader:
                graphs = batch[0].to("cuda" if torch.cuda.is_available() else "cpu")

                if args.pooling == "attn":
                    outputs = model(graphs)
                    outputs_list.append(outputs)
                else:
                    # Each graph in batch has `batch` index in .batch
                    batch_size = graphs.num_graphs
                    for i in range(batch_size):
                        mask = graphs.batch == i
                        feature_embeddings = []

                        for feat_name in ["ops", "sink_t", "vult_t", "tainted"]:
                            node_features = getattr(graphs, f"{feat_name}")[mask]
                            one_hot_features = one_hot_encode(node_features, num_classes=feature_classes[feat_name])
                            pooled_output = (
                                apply_average_pooling(one_hot_features) if args.pooling == "avg"
                                else apply_max_pooling(one_hot_features)
                            )
                            feature_embeddings.append(pooled_output)

                        outputs_list.append(torch.cat(feature_embeddings, dim=0).unsqueeze(0))

                for i in range(graphs.num_graphs):
                    mask = graphs.batch == i
                    label = getattr(graphs, "VULN")[mask].max().item()
                    g_idx = getattr(graphs, "G_IDX")[mask][0].item()
                    labels_list.append(label)
                    gidx_list.append(g_idx)

        outputs_list = [o.cpu() for o in outputs_list]
        # Prepare datasets
        datasets = {
            split_name: (
                torch.cat(outputs_list).cpu().numpy(),
                torch.tensor(labels_list),
                gidx_list,
            )
            for split_name, (
                outputs_list,
                labels_list,
                gidx_list,
            ) in outputs_map.items()
        }


    # Define classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(random_state=args.seed),
        "Logistic Regression": LogisticRegression(random_state=args.seed, max_iter=1000),
        "SVM": SVC(random_state=args.seed, probability=True),
        "XGBoost": XGBClassifier(random_state=args.seed),
    }

        

    # Train and evaluate classifiers

    for name, clf in classifiers.items():
        all_predictions = {}
        gidx_splits = {}
        X_train, Y_train, _ = datasets["Train"]
        if args.mask != -1:
            X_train = feature_masking(X_train, args.mask)
        time_start = time.time()  
        clf.fit(X_train, Y_train)
        print(f"Training time for {name}: {time.time() - time_start}")
        
        if hasattr(clf, "feature_importances_"):
            print(f"Feature Importances for {name}: {clf.feature_importances_}")
            
        if name == "Random Forest" and args.draw_trees:
            # draw trees
            for i, t in enumerate(clf.estimators_):
                with open(f"trees/tree_{i}.dot", 'w') as f:
                    export_graphviz(t, out_file=f)
                os.system(f"dot -Tpng trees/tree_{i}.dot -o trees/tree_{i}.png")

        for split_name, (X, Y, G) in datasets.items():
            if args.mask != -1:
                X_train = feature_masking(X_train, args.mask)
            time_start = time.time()
            pred = clf.predict(X)
            print(f"Prediction time for {name} on {split_name}: {time.time() - time_start}")
            metrics = evaluate_model(Y, pred)
            print(f"{name} on {split_name} - {metrics}")
            pred_prob = clf.predict_proba(X)
            print(f"{name} on {split_name} - Predicted probabilities: {pred_prob}")
            print(f"{name} on {split_name} - True labels: {np.eye(2)[Y]}")

            # Collect predictions and splits
            for gidx, p in zip(G, pred):
                if gidx not in all_predictions:
                    all_predictions[gidx] = {'Train': 'Not Included', 'Validation': 'Not Included', 'Test': 'Not Included'}
                    gidx_splits[gidx] = []
                all_predictions[gidx][split_name] = 'True' if p == 1 else 'False'
                if split_name not in gidx_splits[gidx]:
                    gidx_splits[gidx].append(split_name)

        max_gidx = 1882

        combined_results = [
            (
                gidx,
                " / ".join(
                    all_predictions[gidx][split] 
                    for split in ['Train', 'Validation', 'Test'] 
                    if gidx in all_predictions and all_predictions[gidx][split] != 'Not Included'
                ) or 'Not Included',
                ", ".join(gidx_splits[gidx]) if gidx in gidx_splits and gidx_splits[gidx] else 'Not Included'
            )
            for gidx in range(max_gidx + 1)
        ]

        # Create DataFrame
        df_combined = pd.DataFrame(
            combined_results,
            columns=['G_IDX_', 'Prediction', 'Present In']
        )
        
        if args.verbose:
            # Print 'Prediction' Column with model name
            print(f"=== {name} - Prediction Column ===")
            print(df_combined['Prediction'].to_string(index=False))

            # Print 'Present In' Column with model name
            print(f"\n=== {name} - Present In Column ===")
            print(df_combined['Present In'].to_string(index=False))
            
        print(f"{"=" * 40}")

# for train.py use only
def run_classical_baselines(X_train, y_train, X_test, y_test, seed=42):
    # Set seed
    np.random.seed(seed)
    random.seed(seed)

    # Convert one-hot labels to class indices if needed
    if y_train.ndim == 2:
        y_train = np.argmax(y_train, axis=1)
    if y_test.ndim == 2:
        y_test = np.argmax(y_test, axis=1)

    # Define classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(random_state=seed),
        "Logistic Regression": LogisticRegression(random_state=seed, max_iter=1000),
        "SVM": SVC(random_state=seed),
        "XGBoost": XGBClassifier(random_state=seed),
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"\n{name}")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        results[name] = evaluate_model(y_test, y_pred)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="full",
        choices=["full", "dedup", "dup"],
        help="Dataset to use: full, dedup, or dup",
    )
    
    parser.add_argument(
        "--mask",
        type=int,
        default=-1,
        help="Masking value for input features",
    )
    
    parser.add_argument(
        '--pooling',
        type=str,
        default="attn",
        choices=["attn", "max", 'avg'],
        help="Pooling method to use",
    )
    
    parser.add_argument(
        "--draw-trees",
        action="store_true",
        help="Draw trees for Random Forest",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    parser.add_argument(
        "--map_random_seed_to_ckpt",
        action='store_true',
        default=False,
        help="Map random seed to checkpoint for loading models",
    )
    
    parser.add_argument(
        "--verbose",
        action='store_true',
        default=False,
        help="Enable verbose output",
    )

    args = parser.parse_args()

    main(args)
