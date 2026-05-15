import os
import torch
import torch.nn as nn
from src.models.encoder import GraphBiasAdapter, ALiBiKVMemoryAdapter, ScalarResidualFusion
from transformers.modeling_outputs import BaseModelOutput

class GraphAugmentedNLLB(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        d_model = base_model.config.d_model
        self.graph_adapter = GraphBiasAdapter(d_model, config.NUM_RELATIONS)
        self.memory_adapter = ALiBiKVMemoryAdapter(d_model, config.MEMORY_WINDOW_SIZE)
        self.fusion = ScalarResidualFusion(d_model)
        
    def forward(self, input_ids, attention_mask=None, labels=None, edge_index_head=None, edge_index_dep=None, edge_type=None):
        encoder_outputs = self.base_model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        enc_hidden_states = encoder_outputs.last_hidden_state
        
        if edge_index_head is not None and edge_index_dep is not None and edge_type is not None:
            graph_delta = self.graph_adapter(enc_hidden_states, edge_index_head, edge_index_dep, edge_type)
        else:
            graph_delta = torch.zeros_like(enc_hidden_states)
            
        memory_delta, mem_k, mem_v = self.memory_adapter(enc_hidden_states)
        
        fused_hidden_states = self.fusion(enc_hidden_states, graph_delta, memory_delta)
        
        if not self.training:
            self.memory_adapter.update_memory(mem_k, mem_v)
            
        outputs = self.base_model(
            encoder_outputs=(fused_hidden_states,),
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs

    def generate(self, input_ids, attention_mask=None, edge_index_head=None, edge_index_dep=None, edge_type=None, **kwargs):
        encoder_outputs = self.base_model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        enc_hidden_states = encoder_outputs.last_hidden_state
        
        if edge_index_head is not None and edge_index_dep is not None and edge_type is not None:
            graph_delta = self.graph_adapter(enc_hidden_states, edge_index_head, edge_index_dep, edge_type)
        else:
            graph_delta = torch.zeros_like(enc_hidden_states)
            
        memory_delta, mem_k, mem_v = self.memory_adapter(enc_hidden_states)
        fused_hidden_states = self.fusion(enc_hidden_states, graph_delta, memory_delta)
        
        if not self.training:
            self.memory_adapter.update_memory(mem_k, mem_v)
            
        fused_encoder_outputs = BaseModelOutput(last_hidden_state=fused_hidden_states)
        
        return self.base_model.generate(
            encoder_outputs=fused_encoder_outputs,
            attention_mask=attention_mask,
            **kwargs
        )
