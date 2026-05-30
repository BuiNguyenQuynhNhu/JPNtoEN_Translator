"""
models/graph/adapter.py

Implements Cross-Attention to inject the graph memory back into the decoder,
and a wrapper for the Language Modeling head to intercept the decoder's hidden states.
"""

import torch
import torch.nn as nn
import math

class GraphMemoryAdapter(nn.Module):
    def __init__(self, embed_dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self, 
        decoder_hidden_states: torch.Tensor, 
        node_features: torch.Tensor, 
        batch_index: torch.Tensor,
        local_node_index: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        decoder_hidden_states: [B, T, D]
        node_features: [Total_Nodes, D]
        """
        B, T, D = decoder_hidden_states.size()
        H = self.num_heads
        head_dim = D // H
        
        num_nodes = node_features.size(0)
        if num_nodes == 0:
            return decoder_hidden_states
            
        counts = torch.bincount(batch_index, minlength=B)
        max_nodes = counts.max().item()
        
        # Scatter sparse nodes to dense batch [B, Max_Nodes, D]
        dense_nodes = torch.zeros(B, max_nodes, D, device=node_features.device, dtype=node_features.dtype)
        dense_nodes[batch_index, local_node_index] = node_features
        
        # Create key padding mask [B, Max_Nodes]
        # True where padded (i.e. invalid)
        node_mask = torch.arange(max_nodes, device=node_features.device).unsqueeze(0) < counts.unsqueeze(1)
        key_padding_mask = ~node_mask 
        
        # Prevent NaN in softmax for completely empty sequences by unmasking their first node
        empty_seqs = (counts == 0)
        if empty_seqs.any():
            key_padding_mask[empty_seqs, 0] = False
        
        residual = decoder_hidden_states
        x = self.norm(decoder_hidden_states)
        
        # Cross Attention
        # Q: [B, H, T, head_dim]
        q = self.q_proj(x).view(B, T, H, head_dim).transpose(1, 2)
        # K, V: [B, H, Max_Nodes, head_dim]
        k = self.k_proj(dense_nodes).view(B, max_nodes, H, head_dim).transpose(1, 2)
        v = self.v_proj(dense_nodes).view(B, max_nodes, H, head_dim).transpose(1, 2)
        
        # [B, H, T, Max_Nodes]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply padding mask to keys
        # scores: [B, H, T, Max_Nodes]
        # key_padding_mask: [B, Max_Nodes] -> [B, 1, 1, Max_Nodes]
        scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        # [B, H, T, Max_Nodes] x [B, H, Max_Nodes, head_dim] -> [B, H, T, head_dim]
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        
        if empty_seqs.any():
            out[empty_seqs] = 0.0
        
        return residual + out


class GraphLMHeadWrapper(nn.Module):
    """
    Wraps the baseline's LM Head. 
    Intercepts the final decoder hidden states, runs them through the GraphMemoryAdapter,
    and then passes them to the original LM Head.
    """
    def __init__(self, original_lm_head: nn.Module, graph_adapter: GraphMemoryAdapter):
        super().__init__()
        self.original_lm_head = original_lm_head
        self.graph_adapter = graph_adapter
        
        # Stored per-forward/generate pass
        self.graph_state = None
        
    def set_graph_state(self, node_features, batch_index, local_node_index, batch_size):
        """
        Called by BaselineTranslator before the LM Head is invoked.
        """
        self.graph_state = {
            "node_features": node_features,
            "batch_index": batch_index,
            "local_node_index": local_node_index,
            "batch_size": batch_size
        }
        
    def forward(self, decoder_hidden_states: torch.Tensor, *args, **kwargs):
        if self.graph_state is not None:
            decoder_hidden_states = self.graph_adapter(
                decoder_hidden_states=decoder_hidden_states,
                **self.graph_state
            )
            
        return self.original_lm_head(decoder_hidden_states, *args, **kwargs)
