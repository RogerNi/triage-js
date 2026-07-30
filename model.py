import torch
import torch.nn as nn
from transformers import (
    LlamaForSequenceClassification,
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig, 
    AutoModel, 
    PreTrainedModel
)

from operator import itemgetter
from torch_geometric.data import Batch
from torch_geometric.utils import to_undirected



class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, extra_dim, dropout=None, args=None):
        super().__init__()
        if config is not None:
            llm_hidden_size = config.hidden_size
        else:
            llm_hidden_size = 0

        
        output_layers = []
        output_in_size = llm_hidden_size + extra_dim
        for i in range(args.num_output_layers):
            if i == args.num_output_layers-1:
                output_size = 2
            else:
                output_size = output_in_size
            output_layers.append(nn.Linear(output_in_size, output_size))
            if i != args.num_output_layers-1:
                output_layers.append(nn.ReLU())
        self.output_layer = nn.Sequential(*output_layers)
        

    def forward(self, features, flowgnn_embed, return_embed=False, **kwargs):
        if features is not None:
            # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
            x = features
            if flowgnn_embed is not None:
                x = torch.cat((x, flowgnn_embed), dim=1)
        else:
            x = flowgnn_embed
            
        if return_embed:
            return x
        
        return self.output_layer(x)


class GNNModel(nn.Module):
    def __init__(self, flowgnn_encoder, config, args):
        super().__init__()
        self.flowgnn_encoder = flowgnn_encoder
        self.classifier = ClassificationHead(
            config, 0 if args.no_gnn or flowgnn_encoder is None else self.flowgnn_encoder.out_dim, args.dropout, args
        ) 
        self.args = args
        

    def forward(
        self,
        graphs=None,
        llm_hidden_states=None,
        return_embed=False,
        **kwargs
    ):
        if self.args.no_gnn:
            flowgnn_embed = None
        elif graphs is not None:
            flowgnn_embed = self.flowgnn_encoder(graphs, {})
            
        logits = self.classifier(llm_hidden_states, flowgnn_embed, return_embed=return_embed)
        return logits
    
    
class MyModelForSequenceClassification(PreTrainedModel):
    config_class = PretrainedConfig
    base_model_prefix = "base_model"
    
    def __init__(self, llm_model, gnn_model_t, gnn_encoder, global_args, class_weights):
        """
        Args:
            config: The model configuration, which should include a field `gnn_model`
            global_args: A namespace or dict with fields like `optimize_llm` and `no_llm`
            class_weights: A list or tensor of class weights for loss computation
        """
        super().__init__(PretrainedConfig())
        self.global_args = global_args
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        normalized_class_weights = class_weights_tensor / class_weights_tensor.mean()
        self.llm_model = llm_model
        self.gnn_model = gnn_model_t(gnn_encoder, self.llm_model.config if self.llm_model is not None else None, global_args)
        self.gnn_model = self.gnn_model.to(next(self.llm_model.parameters()).device if self.llm_model is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.loss_fct = nn.CrossEntropyLoss(weight=normalized_class_weights.to(next(self.gnn_model.parameters()).device))
        self.init_weights()
        
    def load_gnn_model(self):
        # get the device of the GNN model
        gnn_device = next(self.gnn_model.parameters()).device
        flowgnn_state_dict = torch.load(self.global_args.load_gnn_from, map_location=gnn_device)
        print(f"Ckpt keys: {list(flowgnn_state_dict.keys())}")
        flowgnn_state_dict = {
            key.replace("gnn_model.flowgnn_encoder.", ""): value
            for key, value in flowgnn_state_dict.items()
            if key.startswith("gnn_model.flowgnn_encoder.")
        }
        print(f"flowgnn_state_dict keys: {list(self.gnn_model.flowgnn_encoder.state_dict().keys())}")
        self.gnn_model.flowgnn_encoder.load_state_dict(flowgnn_state_dict)
    
    def freeze_gnn(self):
        for param in self.gnn_model.parameters():
            param.requires_grad = False
        for param in self.gnn_model.flowgnn_encoder.parameters():
            param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, graphs=None, indices=None, return_embed=False, **kwargs):
        if graphs is not None and not self.global_args.no_gnn:
            if isinstance(graphs, Batch):
                # Unbatch to individual Data objects
                graph_list = graphs.to_data_list()
            else:
                graph_list = graphs  # already unbatched

            # Select the graphs by given indices
            selector = itemgetter(*indices.tolist())
            graph_list = selector(graph_list)

            if not isinstance(graph_list, Batch):
                if not isinstance(graph_list, (list, tuple)):
                    graph_list = (graph_list,)
                graphs = Batch.from_data_list(list(graph_list)).to(labels.device)
            else:
                graphs = graph_list.to(labels.device)

        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        model_inputs.update(kwargs)
        
        # Get LLM hidden states.
        if self.global_args.optimize_llm == "none":
            self.llm_model.eval()
            with torch.no_grad():
                base_outputs = self.llm_model(**model_inputs)
                llm_hidden_states = base_outputs["last_hidden_state"]
        else:
            if self.global_args.no_llm:
                llm_hidden_states = None
                base_outputs = {}
            else:
                model_inputs_clean = model_inputs.copy()
                model_inputs_clean.pop("extrafeats", None)
                model_inputs_clean.pop("num_items_in_batch", None)

                base_outputs = self.llm_model(**model_inputs_clean)
                llm_hidden_states = base_outputs["last_hidden_state"]
        
        # If using the LLM, pick the last non-padding token from the sequence.
        if not self.global_args.no_llm and llm_hidden_states is not None:
            # Get pad token from the model configuration.
            pad_token = self.llm_model.config.pad_token_id
            # Create pad mask.
            if isinstance(pad_token, (list, tuple)):
                pad_ids = torch.tensor(pad_token, device=input_ids.device)
                pad_mask = (input_ids.unsqueeze(-1) == pad_ids).any(dim=-1)
            else:
                pad_mask = (input_ids == pad_token)
            non_pad_mask = (~pad_mask).to(torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=llm_hidden_states.device)
            # For each sample, get the index of the last non-pad token.
            last_non_pad_token = (token_indices * non_pad_mask).argmax(dim=-1)
            batch_indices = torch.arange(input_ids.shape[0], device=llm_hidden_states.device)
            llm_hidden_states = llm_hidden_states[batch_indices, last_non_pad_token]
        
        # Forward pass through the GNN classification head.
        with torch.amp.autocast('cuda', enabled=False):
            logits = self.gnn_model(graphs=graphs, llm_hidden_states=llm_hidden_states, return_embed=return_embed)
        
        if return_embed:
            return logits
                
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits,
        }
        