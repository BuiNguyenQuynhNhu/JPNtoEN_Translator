"""
models/full_model/baseline.py

This module implements the Baseline NLLB/mBART wrapper for translation.
It exposes the encoder hidden states (Stage 5) to prepare for graph construction.

How to test:
    import torch
    from models.full_model.baseline import BaselineTranslator
    model = BaselineTranslator("facebook/nllb-200-distilled-600M")
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones(2, 10)
    labels = torch.randint(0, 1000, (2, 10))
    out = model(input_ids, attention_mask, labels)
    print(out.loss) # Scalar
    print(out.encoder_last_hidden_state.shape) # Expected: [2, 10, 1024]
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from models.graph.features import NodeFeatureExtractor
from models.graph.transformer import GraphTransformer
from models.graph.adapter import GraphMemoryAdapter, GraphLMHeadWrapper

class BaselineTranslator(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.node_extractor = NodeFeatureExtractor()
        
        # We extract hidden size from the config
        embed_dim = self.model.config.d_model
        self.graph_transformer = GraphTransformer(embed_dim=embed_dim)
        
        # Cross-Attention Memory Adapter
        self.graph_adapter = GraphMemoryAdapter(embed_dim=embed_dim)
        
        # Wrap the original LM Head
        self.model.lm_head = GraphLMHeadWrapper(
            original_lm_head=self.model.lm_head,
            graph_adapter=self.graph_adapter
        )
        
    def forward(self, input_ids, attention_mask, labels=None, graph=None, **kwargs):
        """
        Forward pass for training.
        """
        if graph is not None:
            # 1. Run Encoder first to get hidden states for graph extraction
            encoder_outputs = self.model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Stage 7: Extract contextual node embeddings
            if hasattr(encoder_outputs, 'hidden_states') and encoder_outputs.hidden_states is not None:
                encoder_hs = encoder_outputs.hidden_states[-1]
            else:
                encoder_hs = encoder_outputs.last_hidden_state
                
            node_features = self.node_extractor(
                encoder_hs,
                graph["node_spans"],
                graph["batch_index"]
            )
            
            # Stage 8: Graph Transformer
            batch_size = input_ids.size(0)
            encoded_nodes = self.graph_transformer(
                node_features=node_features,
                edge_index=graph["edge_index"],
                edge_type=graph["edge_type"],
                batch_index=graph["batch_index"],
                local_node_index=graph["local_node_index"],
                batch_size=batch_size
            )
            
            # Stage 9: Store in LM Head wrapper for cross-attention interception
            self.model.lm_head.set_graph_state(
                node_features=encoded_nodes,
                batch_index=graph["batch_index"],
                local_node_index=graph["local_node_index"],
                batch_size=batch_size
            )
            
            # 2. Run Decoder and rest of model, passing encoder_outputs
            kwargs["encoder_outputs"] = encoder_outputs
            outputs = self.model(
                input_ids=input_ids, # NLLB accepts this even with encoder_outputs
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )
            
            # Keep for backward compatibility/logging if needed
            outputs.node_features = encoded_nodes
            
        else:
            # Clear graph state if no graph provided
            if hasattr(self.model.lm_head, "set_graph_state"):
                self.model.lm_head.graph_state = None
                
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )
                
        return outputs
        
    def generate(self, input_ids, attention_mask, graph=None, **kwargs):
        """
        Forward pass for inference/generation.
        """
        if graph is not None:
            # For generation, we must manually run the encoder to get hidden states first
            # because generate() handles the encoder internally if not passed.
            # However, we can simply pass `encoder_outputs` to generate() so it skips the encoder step!
            
            encoder_outputs = self.model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            if hasattr(encoder_outputs, 'hidden_states') and encoder_outputs.hidden_states is not None:
                encoder_hs = encoder_outputs.hidden_states[-1]
            else:
                encoder_hs = encoder_outputs.last_hidden_state
                
            node_features = self.node_extractor(
                encoder_hs,
                graph["node_spans"],
                graph["batch_index"]
            )
            
            batch_size = input_ids.size(0)
            encoded_nodes = self.graph_transformer(
                node_features=node_features,
                edge_index=graph["edge_index"],
                edge_type=graph["edge_type"],
                batch_index=graph["batch_index"],
                local_node_index=graph["local_node_index"],
                batch_size=batch_size
            )
            
            self.model.lm_head.set_graph_state(
                node_features=encoded_nodes,
                batch_index=graph["batch_index"],
                local_node_index=graph["local_node_index"],
                batch_size=batch_size
            )
            
            # Pass the pre-computed encoder_outputs to generate()
            kwargs["encoder_outputs"] = encoder_outputs
        else:
            if hasattr(self.model.lm_head, "set_graph_state"):
                self.model.lm_head.graph_state = None
                
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

# Future improvements:
# - Add LoRA support if requested.
# - The model can later be injected with graph modules.
