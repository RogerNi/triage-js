"""
Gated Graph Neural Network module for graph classification tasks using PyG
"""
import torch
from torch import nn
from torch_geometric.nn import GatedGraphConv, GlobalAttention
import logging

logger = logging.getLogger(__name__)

allfeats = [
    "ops", "sink_t", "vult_t", "tainted"
]

class FlowGNNGGNNModule(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 n_steps,
                 num_output_layers,
                 label_style="graph",
                 concat_all_absdf=True,
                 encoder_mode=False,
                 **kwargs):
        super(FlowGNNGGNNModule, self).__init__()

        self.concat_all_absdf = concat_all_absdf
        self.input_dim = input_dim

        embedding_dim = hidden_dim
        if self.concat_all_absdf:
            self.all_embeddings = nn.ModuleDict({
                of: nn.Embedding(input_dim, embedding_dim) for of in allfeats
            })
            embedding_dim *= len(allfeats)
            hidden_dim *= len(allfeats)
        else:
            self.embedding = nn.Embedding(input_dim, embedding_dim)

        # PyG GatedGraphConv assumes edge_index and edge_weight, no etypes
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=n_steps)

        self.output_in_size = hidden_dim + embedding_dim
        self.out_dim = self.output_in_size

        if label_style == "graph":
            self.pooling = GlobalAttention(gate_nn=nn.Linear(self.output_in_size, 1))

        if not encoder_mode:
            output_layers = []
            for i in range(num_output_layers):
                out_size = 1 if i == num_output_layers - 1 else self.output_in_size
                output_layers.append(nn.Linear(self.output_in_size, out_size))
                if i != num_output_layers - 1:
                    output_layers.append(nn.ReLU())
            self.output_layer = nn.Sequential(*output_layers)

        self.label_style = label_style
        self.encoder_mode = encoder_mode

    def forward(self, data, extra_feats=None):
        # data: PyG Data object with .x and .edge_index
        if self.concat_all_absdf:
            feats = []
            for of in allfeats:
                feat_idx = getattr(data, f"{of}").to(data.ops.device)
                feats.append(self.all_embeddings[of](feat_idx))
            feat_embed = torch.cat(feats, dim=-1)
        else:
            feat_embed = self.embedding(data.x)

        # Gated Graph Conv in PyG does not use edge features
        x = self.ggnn(feat_embed, data.edge_index)

        # Concat with input
        x = torch.cat([x, feat_embed], dim=-1)

        # Graph-level prediction
        if self.label_style == "graph":
            x = self.pooling(x, data.batch)

        if self.encoder_mode:
            return x
        else:
            return self.output_layer(x).squeeze()