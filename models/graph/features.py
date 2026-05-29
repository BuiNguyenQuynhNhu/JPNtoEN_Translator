"""
models/graph/features.py

Extracts node features from encoder hidden states using vectorized span mean pooling.
"""

import torch
import torch.nn as nn

class NodeFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(
        self, 
        encoder_hidden_states: torch.Tensor, 
        node_spans: torch.Tensor, 
        batch_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Extracts pooled features for each node based on its token span.
        
        Args:
            encoder_hidden_states: [B, T, D] Output from Seq2Seq Encoder
            node_spans: [Total_Nodes, 2] Start (inclusive) and End (exclusive) token indices
            batch_index: [Total_Nodes] Maps each node to its sequence in the batch
            
        Returns:
            node_features: [Total_Nodes, D] Contextual embeddings for the nodes
        """
        num_nodes = node_spans.size(0)
        
        # If there are no nodes in the batch (empty graph), return empty tensor
        if num_nodes == 0:
            return torch.empty((0, encoder_hidden_states.size(-1)), 
                               device=encoder_hidden_states.device, 
                               dtype=encoder_hidden_states.dtype)
            
        B, T, D = encoder_hidden_states.size()
        
        # 1. Gather the sequence for each node
        # node_seqs shape: [Total_Nodes, T, D]
        node_seqs = encoder_hidden_states[batch_index]
        
        # 2. Create a mask for the token spans
        # positions shape: [1, T]
        positions = torch.arange(T, device=encoder_hidden_states.device).unsqueeze(0)
        
        starts = node_spans[:, 0].unsqueeze(1) # [Total_Nodes, 1]
        ends = node_spans[:, 1].unsqueeze(1)   # [Total_Nodes, 1]
        
        # mask shape: [Total_Nodes, T]
        mask = (positions >= starts) & (positions < ends)
        mask = mask.to(encoder_hidden_states.dtype).unsqueeze(-1) # [Total_Nodes, T, 1]
        
        # 3. Sum the tokens within the span
        # summed shape: [Total_Nodes, D]
        summed = (node_seqs * mask).sum(dim=1)
        
        # 4. Divide by the span length to get the mean
        # lengths shape: [Total_Nodes, 1]
        lengths = mask.sum(dim=1).clamp(min=1.0)
        
        # pooled shape: [Total_Nodes, D]
        pooled = summed / lengths
        
        return pooled
