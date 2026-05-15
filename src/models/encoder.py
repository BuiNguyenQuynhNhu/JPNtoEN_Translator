import torch
import torch.nn as nn
import math
import collections

def get_alibi_slopes(num_heads):
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    slopes = [math.pow(base, i) for i in range(1, closest_power_of_2 + 1)]
    if closest_power_of_2 < num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        slopes += [math.pow(extra_base, i) for i in range(1, 2 * (num_heads - closest_power_of_2) + 1, 2)]
    return slopes

class GraphBiasAdapter(nn.Module):
    def __init__(self, d_model, num_relations, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Relation-Aware bias embedding
        self.edge_bias_emb = nn.Embedding(num_relations, num_heads, padding_idx=0)
        
    def forward(self, x, edge_index_head, edge_index_dep, edge_type):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # O(S^2) dense scores matrix, required by standard self-attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if edge_index_head is not None and edge_index_dep is not None and edge_type is not None:
            # edge_type: (Total_Edges,)
            bias_emb = self.edge_bias_emb(edge_type) # (Total_Edges, num_heads)
            
            # Since scores is (B, H, S, S), we need to know which batch each edge belongs to.
            # In our custom collate_fn, we will pad edge_index so it's (B, Max_Edges).
            # If so, edge_type is (B, Max_Edges).
            # bias_emb is (B, Max_Edges, num_heads).
            if bias_emb.dim() == 3:
                # We scatter directly into the scores matrix.
                # bias_emb: (B, E, H) -> transpose -> (B, H, E)
                bias_emb = bias_emb.transpose(1, 2)
                
                # To scatter_add_, we need to flatten the S x S dimensions or use a loop.
                # Actually, an easy way is just to iterate over batches or use index_put_.
                for b in range(batch_size):
                    # Get valid edges for this batch (ignore padding -1 or 0)
                    valid_mask = edge_type[b] > 0
                    heads = edge_index_head[b][valid_mask]
                    deps = edge_index_dep[b][valid_mask]
                    embs = bias_emb[b, :, valid_mask] # (H, V_E)
                    
                    if heads.numel() > 0:
                        # In-place scatter addition
                        # scores[b, :, heads, deps] += embs
                        scores[b].index_put_((slice(None), heads, deps), embs, accumulate=True)
            
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)

class ALiBiKVMemoryAdapter(nn.Module):
    def __init__(self, d_model, window_size, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Bounded DEQUE Memory (OOM-free)
        self.memory_k = collections.deque(maxlen=window_size)
        self.memory_v = collections.deque(maxlen=window_size)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.register_buffer("alibi_slopes", torch.tensor(get_alibi_slopes(num_heads)).view(1, num_heads, 1, 1))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if len(self.memory_k) > 0:
            # Because deque maxlen is fixed, this will never OOM.
            past_k = torch.cat(list(self.memory_k), dim=2)
            past_v = torch.cat(list(self.memory_v), dim=2)
            
            scores = torch.matmul(q, past_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            M_S = past_k.size(2)
            q_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)
            k_idx = torch.arange(M_S, device=x.device).unsqueeze(0)
            distances = M_S - k_idx + q_idx
            alibi_bias = distances.unsqueeze(0).unsqueeze(0) * self.alibi_slopes * -1.0
            
            scores = scores + alibi_bias
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, past_v)
            
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            out = self.out_proj(out)
            return out, k, v
        else:
            return torch.zeros_like(x), k, v
        
    def update_memory(self, k, v):
        # Strict detach to prevent gradient accumulation
        self.memory_k.append(k.detach())
        self.memory_v.append(v.detach())

class ScalarResidualFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.beta = nn.Parameter(torch.tensor(0.01))
        
        self.norm_graph = nn.LayerNorm(d_model)
        self.norm_memory = nn.LayerNorm(d_model)
        
    def forward(self, enc_out, graph_delta, memory_delta):
        graph_delta_norm = self.norm_graph(graph_delta)
        memory_delta_norm = self.norm_memory(memory_delta)
        
        final_hidden = enc_out + self.alpha * graph_delta_norm + self.beta * memory_delta_norm
        return final_hidden
