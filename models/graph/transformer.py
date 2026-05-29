"""
models/graph/transformer.py

Implements a custom Graph Transformer module that:
1. Converts flattened sparse node features into a dense padded batch.
2. Embeds edge types to create an attention bias matrix.
3. Runs Multi-Head Attention over the graph.
4. Flattens the processed nodes back to sparse format.
"""

import torch
import torch.nn as nn
import math

class GraphAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]
        attn_bias: [B, N, N] - pre-scaled structural bias matrix
        """
        B, N, D = x.size()
        H = self.num_heads
        head_dim = D // H
        
        residual = x
        x = self.norm1(x)
        
        q = self.q_proj(x).view(B, N, H, head_dim).transpose(1, 2) # [B, H, N, head_dim]
        k = self.k_proj(x).view(B, N, H, head_dim).transpose(1, 2) # [B, H, N, head_dim]
        v = self.v_proj(x).view(B, N, H, head_dim).transpose(1, 2) # [B, H, N, head_dim]
        
        # [B, H, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Add the structural bias to the attention scores
        # attn_bias is [B, N, N], broadcast to [B, 1, N, N]
        scores = scores + attn_bias.unsqueeze(1)
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        # [B, H, N, head_dim] -> [B, N, H*head_dim]
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        x = residual + out
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x

class GraphTransformer(nn.Module):
    def __init__(self, embed_dim: int = 1024, num_heads: int = 8, num_layers: int = 2, num_edge_types: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Edge type embeddings to inject structural bias into attention
        # We add 1 for "No Edge" which will be heavily masked or 0.
        self.edge_embeddings = nn.Embedding(num_edge_types + 1, 1)
        
        self.layers = nn.ModuleList([
            GraphAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
    def forward(
        self, 
        node_features: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_type: torch.Tensor,
        batch_index: torch.Tensor,
        local_node_index: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        node_features: [Total_Nodes, D]
        edge_index: [2, Total_Edges]
        edge_type: [Total_Edges]
        batch_index: [Total_Nodes]
        local_node_index: [Total_Nodes]
        """
        num_nodes = node_features.size(0)
        if num_nodes == 0:
            return node_features
            
        B = batch_size
        counts = torch.bincount(batch_index, minlength=B)
        max_nodes = counts.max().item()
        
        # 1. Scatter to dense [B, Max_Nodes, D]
        dense_nodes = torch.zeros(B, max_nodes, self.embed_dim, device=node_features.device, dtype=node_features.dtype)
        dense_nodes[batch_index, local_node_index] = node_features
        
        # 2. Build Dense Attention Bias [B, Max_Nodes, Max_Nodes]
        # Default is 0 (which means "No Edge"), so padding mask will be handled separately
        attn_bias = torch.zeros(B, max_nodes, max_nodes, device=node_features.device, dtype=node_features.dtype)
        
        if edge_index.size(1) > 0:
            # Map global edge indices to batch and local indices
            src_global = edge_index[0]
            tgt_global = edge_index[1]
            
            # Since edges only exist within the same graph, batch_index[src] == batch_index[tgt]
            edge_batch = batch_index[src_global]
            src_local = local_node_index[src_global]
            tgt_local = local_node_index[tgt_global]
            
            # Embed the edge types. edge_type is 1-indexed (1=DEP, 2=TEMP, 3=SAME)
            # Embedding gives shape [Total_Edges, 1]
            e_emb = self.edge_embeddings(edge_type).squeeze(-1)
            
            # Scatter edge bias
            attn_bias[edge_batch, src_local, tgt_local] = e_emb
            
        # 3. Apply Padding Mask to Attention Bias
        # Nodes that do not exist (padding) should have -inf bias so they are never attended to
        node_mask = torch.arange(max_nodes, device=node_features.device).unsqueeze(0) < counts.unsqueeze(1) # [B, Max_Nodes]
        padding_mask = ~node_mask # True where padded
        
        # We need to mask out attending TO padded nodes, and FROM padded nodes
        # [B, 1, Max_Nodes]
        attn_bias.masked_fill_(padding_mask.unsqueeze(1), -1e9)
        attn_bias.masked_fill_(padding_mask.unsqueeze(2), -1e9)
        
        # 4. Pass through Graph Transformer layers
        x = dense_nodes
        for layer in self.layers:
            x = layer(x, attn_bias)
            
        # 5. Gather back to sparse [Total_Nodes, D]
        out_features = x[batch_index, local_node_index]
        
        return out_features
