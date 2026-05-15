import os
import torch
import torch.nn as nn
import spacy
import numpy as np
import math


class GraphBuilder:
    def __init__(self, tokenizer):
        self.nlp = spacy.load("ja_ginza")
        self.tokenizer = tokenizer
        self.rel2id = {"<pad>": 0, "ROOT": 1}
        
    def build_dense_graph(self, text, max_length):
        doc = self.nlp(text)
        
        encoded = self.tokenizer(
            text, 
            return_tensors="pt", 
            return_offsets_mapping=True,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True
        )
        
        input_ids = encoded["input_ids"][0]
        offsets = encoded["offset_mapping"][0]
        special_token_mask = [id in self.tokenizer.all_special_ids for id in input_ids.tolist()]
        
        word_to_subwords = {token.i: [] for token in doc}
        subword_idx = 0
        num_subwords = len(offsets)
        
        for token in doc:
            token_start = token.idx
            token_end = token.idx + len(token.text)
            
            while subword_idx < num_subwords:
                start_char, end_char = offsets[subword_idx]
                
                if special_token_mask[subword_idx] or start_char == end_char:
                    subword_idx += 1
                    continue
                    
                if start_char >= token_end:
                    break 
                    
                if end_char <= token_start:
                    subword_idx += 1 
                    continue
                    
                word_to_subwords[token.i].append(subword_idx)
                subword_idx += 1
                
        edge_bias_matrix = torch.zeros((max_length, max_length), dtype=torch.long)
        
        for token in doc:
            if token.dep_ not in self.rel2id:
                self.rel2id[token.dep_] = len(self.rel2id)
            rel_id = self.rel2id[token.dep_]
            
            head_subwords = word_to_subwords.get(token.head.i, [])
            dep_subwords = word_to_subwords.get(token.i, [])
            
            for head_idx in head_subwords:
                for dep_idx in dep_subwords:
                    if head_idx < max_length and dep_idx < max_length:
                        edge_bias_matrix[head_idx, dep_idx] = rel_id
                        
        return edge_bias_matrix


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
        
    def forward(self, x, edge_bias_matrix):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if edge_bias_matrix is not None:
            bias_emb = self.edge_bias_emb(edge_bias_matrix).permute(0, 3, 1, 2)
            scores = scores + bias_emb
            
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
        
        self.memory_k = []
        self.memory_v = []
        
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
            past_k = torch.cat(self.memory_k, dim=2)
            past_v = torch.cat(self.memory_v, dim=2)
            
            # Cross-Attention Querying Past Context
            scores = torch.matmul(q, past_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            # Apply ALiBi distance penalty
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
        self.memory_k.append(k.detach())
        self.memory_v.append(v.detach())
        if len(self.memory_k) > self.window_size:
            self.memory_k.pop(0)
            self.memory_v.pop(0)

class ScalarResidualFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Alpha and Beta initialized strictly small to preserve pretrained NLLB knowledge
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.beta = nn.Parameter(torch.tensor(0.01))
        
        self.norm_graph = nn.LayerNorm(d_model)
        self.norm_memory = nn.LayerNorm(d_model)
        
    def forward(self, enc_out, graph_delta, memory_delta):
        graph_delta_norm = self.norm_graph(graph_delta)
        memory_delta_norm = self.norm_memory(memory_delta)
        
        final_hidden = enc_out + self.alpha * graph_delta_norm + self.beta * memory_delta_norm
        return final_hidden

