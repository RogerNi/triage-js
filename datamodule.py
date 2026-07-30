import pandas as pd
import torch
import os
import numpy as np
import json
import hashlib
import logging

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Subset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class ProvenanceDataset(InMemoryDataset):
    def __init__(self, data_path, load_features=True, **kwargs):
        self.kwargs = kwargs
        self.data_path = data_path
        self.basic_meta = pd.read_csv(os.path.join(data_path, "basic_metadata.csv"))
        self.return_dict = kwargs.get("return_dict", True)
        self.use_chat_template = kwargs.get("use_chat_template", False)
        self.chat_template_no_labels = kwargs.get("chat_template_no_labels", False)
        self.load_features = load_features
        self.undersample = kwargs.get("undersample", None)
        self.oversample = kwargs.get("oversample", None)
        self.balance = kwargs.get("balance", False)
        self.seed = kwargs.get("seed", 0)
        self.return_only_func = kwargs.get("return_only_func", False)
        self.rng = np.random.RandomState(self.seed)

        if load_features:
            feature_file_name = kwargs.get("feature_file_name", "node_information.csv")
            self.node_info = pd.read_csv(os.path.join(data_path, feature_file_name))

        data_file = os.path.join(data_path, "provenance_graphs.pt")
        self.graphs = torch.load(data_file, weights_only=False)

        if kwargs.get("pre_filter", None) is not None:
            with open(kwargs["pre_filter"], "r") as f:
                pre_filter = json.load(f)
            self.graphs = [self.graphs[i] for i in pre_filter]

        partition = {"train": 0, "val": 1, "test": 2, "all": 3}[kwargs["partition"]]
        self.partition = partition

        if partition != 3:
            if kwargs.get("fix_split") is not None:
                with open(kwargs["fix_split"], "r") as file:
                    fixed_split = json.load(file)
                idx_map = [fixed_split["train"], fixed_split["val"], fixed_split["test"]][partition]
                self.graphs = [self.graphs[i] for i in idx_map]
                self.basic_meta = self.basic_meta.iloc[idx_map].reset_index(drop=True)
            else:
                total_len = len(self.graphs)
                train_len = int(0.8 * total_len)
                val_len = int(0.1 * total_len)
                test_len = total_len - train_len - val_len
                partitions = torch.utils.data.random_split(self.graphs, [train_len, val_len, test_len],
                                                           generator=torch.Generator().manual_seed(self.seed))
                self.graphs = list(partitions[partition])
                print(f"Partition {partition} G_IDX: {[int(g.G_IDX[0]) for g in self.graphs]}")
                self.basic_meta = self.basic_meta.iloc[
                    [int(g.G_IDX[0].item()) for g in self.graphs]
                ].reset_index(drop=True)

        feats_init = [dict() for _ in self.graphs]
        self.extrafeats = feats_init

        if self.load_features:
            self.block_size = kwargs["block_size"]
            if kwargs.get("model_name_or_path", None) is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(kwargs.get("model_name_or_path", ""))
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "right"
            self.preprompt = kwargs.get("preprompt", "")
            self.func_list = []
            if kwargs.get("code_feature", "per_node") == "per_node":
                self.node_info.set_index("global_node_idx", inplace=True)
            else:
                self.node_info.set_index("graph_idx", inplace=True)

            for graph in self.graphs:
                graph_idx = int(graph.G_IDX[0])
                label = int(graph.y[0])
                if kwargs.get("code_feature", "per_node") == "per_node":
                    node_func_list = self.node_info[self.node_info["graph_idx"] == graph_idx]["code"].fillna("").tolist()
                    text = self.preprompt + "\n\n\n".join(
                        [f"Block {i} of codes:\n{func}" for i, func in enumerate(node_func_list)])
                    text += f"\n\nThe graph connecting these blocks of code has the following structure:\nEdges: {graph.edge_index.t().tolist()}"
                    self.func_list.append(self.convert_examples_to_features(text, label))
                    raise NotImplementedError("Per-node code feature is not implemented yet.")
                else:
                    node_func = self.node_info.loc[graph_idx, "code"]
                    if not isinstance(node_func, str):
                        node_func = "code missing"
                    self.func_list.append(self.convert_examples_to_features(node_func, label, instruct=self.preprompt))

        logger.info(f"total {partition}: {len(self.graphs)}, confirmed: {self.basic_meta.confirmed.sum()}, unconfirmed: {len(self.graphs)-self.basic_meta.confirmed.sum()}")

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = graph.VULN.max().long()
        label_onehot = torch.nn.functional.one_hot(label.long(), num_classes=2).float()
        result = {"graph": graph, "extrafeats": self.extrafeats[idx], "labels": label_onehot}
        if self.load_features:
            result["func"] = self.func_list[idx]
        
        if self.return_only_func:
            return result["func"]
        elif self.return_dict:
            return result
        else:
            return tuple(result[k] for k in ["graph", "extrafeats", "func"] if k in result)

    def __len__(self):
        return len(self.graphs)

    def get_class_num(self):
        return [self.basic_meta.confirmed.sum(), len(self.basic_meta) - self.basic_meta.confirmed.sum()]

    def get_epoch_indices(self):
        index = self.basic_meta.index
        if self.balance:
            logger.info("Automatic balancing activated. Resampling to balance class distribution.")
            vul = self.basic_meta[self.basic_meta.confirmed == True]
            nonvul = self.basic_meta[self.basic_meta.confirmed == False]
            if len(vul) < len(nonvul):
                vul = vul.sample(len(nonvul), replace=True, random_state=self.rng)
            else:
                nonvul = nonvul.sample(len(vul), replace=True, random_state=self.rng)
            index = pd.concat([vul, nonvul]).index
        elif self.undersample is not None or self.oversample is not None:
            logger.info("Manual undersampling/oversampling activated.")
            vul = self.basic_meta[self.basic_meta.confirmed == True]
            nonvul = self.basic_meta[self.basic_meta.confirmed == False]
            if self.undersample is not None:
                if str(self.undersample).startswith("v"):
                    ratio = float(str(self.undersample)[1:])
                    nonvul = nonvul.sample(int(len(vul) * ratio), replace=False, random_state=self.rng)
                else:
                    nonvul = nonvul.sample(int(len(nonvul) * self.undersample), replace=False, random_state=self.rng)
            if self.oversample is not None:
                if str(self.oversample).startswith("v"):
                    ratio = float(str(self.oversample)[1:])
                    vul = vul.sample(int(len(nonvul) * ratio), replace=True, random_state=self.rng)
                else:
                    vul = vul.sample(int(len(vul) * self.oversample), replace=True, random_state=self.rng)
            index = pd.concat([vul, nonvul]).index
        return index

    def convert_examples_to_features(self, func, label=None, instruct=None):
        if self.use_chat_template:
            messages = []

            # Add system instruction if provided
            if instruct:
                messages.append({"role": "system", "content": instruct})

            if self.partition in [0, 1]:
                messages.append({"role": "user", "content": func})
                if not self.chat_template_no_labels:
                    messages.append({"role": "assistant", "content": "Yes" if label == 1 else "No"})
                func = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    padding="max_length",
                    truncation=True,
                    max_length=self.block_size,
                )
            else:
                messages.append({"role": "user", "content": func})
                func = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    padding="max_length",
                    truncation=True,
                    max_length=self.block_size,
                )

        if self.kwargs.get("no_tokenization", True) and self.partition == 2:
            return func

        return self.tokenizer(
            str(func),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.block_size,
        )

    def dedup_sanity_check(self, graphs):
        hash_set = set()
        unique_gidxs = set()

        for graph in graphs:
            # Access _G_IDX (assumed to be a tensor attribute in each graph)
            _g_idx = int(graph.G_IDX[0].item()) if hasattr(graph, "G_IDX") else int(graph.x_G_IDX[0].item())
            graph_hash = self.hash_graph(graph)
            if graph_hash in hash_set:
                return False
            hash_set.add(graph_hash)
            unique_gidxs.add(_g_idx)

        return True
    
    def hash_graph(self, graph):
        allfeats = ["ops", "sink_t", "vult_t", "tainted"]
        
        # Compute adjacency matrix hash
        adj_matrix = to_dense_adj(graph.edge_index)[0]
        adj_hash = hashlib.md5(adj_matrix.cpu().numpy().tobytes()).hexdigest()

        # Concatenate node feature hashes
        node_features = b''
        for feat_name in allfeats:
            if hasattr(graph, f"_{feat_name}"):
                feat_tensor = getattr(graph, f"_{feat_name}")
            else:
                feat_tensor = graph[f"_{feat_name}"]
            node_features += feat_tensor.cpu().numpy().tobytes()

        combined_data = adj_hash + hashlib.md5(node_features).hexdigest()
        return hashlib.md5(combined_data.encode()).hexdigest()


class ProvenanceGraphDataModule():
    def __init__(
        self,
        data_path,
        batch_size=256,
        seed=0,
        sample=-1,
        sample_mode=False,
        undersample=None,
        oversample=None,
        train_workers=4,
        val_workers=0,
        test_workers=0,
        load_features=False,
        use_random_weighted_sampler=False,
        block_size=1024,
        llm_model_name_or_path=None,
        fix_split=None,
        balance=False,
        pre_filter=None,
        feature_file_name=None,
        preprompt="",
        code_feature="per_node",
        no_tokenization=False,
        return_dict=True,
        use_chat_template=False,
        chat_template_no_labels=False,
        shuffle_train=True,
        return_only_func=False,
    ):
        """Init class from provgraph dataset (PyG version)."""
        dataargs = {
            "sample": sample,
            "sample_mode": sample_mode,
            "undersample": undersample,
            "oversample": oversample,
            "seed": seed,
            "load_features": load_features,
            "block_size": block_size,
            "model_name_or_path": llm_model_name_or_path,
            "fix_split": fix_split,
            "balance": balance,
            "pre_filter": pre_filter,
            "feature_file_name": feature_file_name,
            "preprompt": preprompt,
            "code_feature": code_feature,
            "no_tokenization": no_tokenization,
            "return_dict": return_dict,
            "use_chat_template": use_chat_template,
            "chat_template_no_labels": chat_template_no_labels,
            "return_only_func": return_only_func,
        }
        self.sample_mode = sample_mode

        logger.info("Data args: %s", dataargs)
        self.train = ProvenanceDataset(data_path, partition="train", **dataargs)
        self.val = ProvenanceDataset(data_path, partition="val", **dataargs)
        self.test = ProvenanceDataset(data_path, partition="test", **dataargs)
        
        logger.info(f"SPLIT SIZES: {len(self.train)} {len(self.val)} {len(self.test)}")

        self.batch_size = batch_size
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.test_workers = test_workers
        
        self.use_random_weighted_sampler = use_random_weighted_sampler
        self.shuffle_train = shuffle_train

    def input_dim(self):
        return 102  # adjust as necessary for your feature size

    def positive_weight(self):
        return None

    def train_dataloader(self):
        """Return train dataloader (PyG)."""
        if self.use_random_weighted_sampler:
            # sampler = ImbalancedDatasetSampler(self.train)
            # return PyGDataLoader(
            #     self.train,
            #     batch_size=self.batch_size,
            #     num_workers=self.train_workers,
            #     sampler=sampler,
            # )
            raise NotImplementedError("Random weighted sampler is not implemented.")
        else:
            return PyGDataLoader(
                Subset(self.train, self.train.get_epoch_indices()),
                shuffle=self.shuffle_train,
                batch_size=self.batch_size,
                num_workers=self.train_workers,
            )

    def val_dataloader(self):
        """Return val dataloader (PyG)."""
        return PyGDataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.val_workers
        )

    def test_dataloader(self):
        """Return test dataloader (PyG)."""
        return PyGDataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.test_workers
        )
        
    def get_train_class_num(self):
        return self.train.get_class_num()