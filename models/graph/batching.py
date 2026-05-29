"""
models/graph/batching.py

Utility to convert a list of graph dictionaries into flattened PyTorch tensors
compatible with message passing and custom scatter attention.
"""

import torch
from typing import List, Dict, Any, Tuple

def align_char_spans_to_tokens(char_span: Tuple[int, int], offset_mapping: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Given a character span (start, end) and the tokenizer's offset mapping,
    returns the token span (token_start_idx, token_end_idx).
    The token span is inclusive for start, exclusive for end: [start, end)
    """
    start_char, end_char = char_span
    token_start = -1
    token_end = -1
    
    for idx, (t_start, t_end) in enumerate(offset_mapping):
        # Ignore special tokens with span (0, 0)
        if t_start == 0 and t_end == 0:
            continue
            
        if token_start == -1 and t_end > start_char:
            token_start = idx
            
        if t_start < end_char:
            token_end = idx + 1
            
    if token_start == -1:
        token_start = 0
    if token_end == -1 or token_end <= token_start:
        token_end = token_start + 1
        
    return token_start, token_end

def batch_graphs(graphs: List[Dict[str, Any]], offset_mappings: List[List[Tuple[int, int]]]) -> Dict[str, torch.Tensor]:
    """
    Takes a list of graph dictionaries and their corresponding offset mappings,
    and returns batched flattened tensors.
    
    Returns:
    - node_spans: Tensor of shape [Total_Nodes, 2] containing token (start, end)
    - node_types: Tensor of shape [Total_Nodes] containing 0 for EVENT, 1 for ENTITY
    - edge_index: Tensor of shape [2, Total_Edges]
    - edge_type: Tensor of shape [Total_Edges]
    - batch_index: Tensor of shape [Total_Nodes] mapping each node to its batch.
    """
    all_node_spans = []
    all_node_types = []
    all_edge_indices = [[], []]
    all_edge_types = []
    all_batch_indices = []
    
    node_offset = 0
    
    for b, (graph, offsets) in enumerate(zip(graphs, offset_mappings)):
        nodes = graph["nodes"]
        edges = graph["edges"]
        
        for node in nodes:
            # Align character span to token indices
            token_start, token_end = align_char_spans_to_tokens(node["char_span"], offsets)
            all_node_spans.append([token_start, token_end])
            
            node_type = 0 if node["type"] == "EVENT" else 1
            all_node_types.append(node_type)
            
            all_batch_indices.append(b)
            
        for edge in edges:
            all_edge_indices[0].append(edge["source"] + node_offset)
            all_edge_indices[1].append(edge["target"] + node_offset)
            all_edge_types.append(edge["type"])
            
        node_offset += len(nodes)
        
    # If there are no nodes/edges, we must return empty tensors of correct dimensions
    if len(all_node_spans) == 0:
        return {
            "node_spans": torch.empty((0, 2), dtype=torch.long),
            "node_types": torch.empty((0,), dtype=torch.long),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_type":  torch.empty((0,), dtype=torch.long),
            "batch_index": torch.empty((0,), dtype=torch.long)
        }
        
    return {
        "node_spans": torch.tensor(all_node_spans, dtype=torch.long),
        "node_types": torch.tensor(all_node_types, dtype=torch.long),
        "edge_index": torch.tensor(all_edge_indices, dtype=torch.long),
        "edge_type": torch.tensor(all_edge_types, dtype=torch.long),
        "batch_index": torch.tensor(all_batch_indices, dtype=torch.long)
    }
