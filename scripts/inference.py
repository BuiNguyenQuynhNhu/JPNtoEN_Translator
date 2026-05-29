"""
scripts/inference.py

Standalone inference script for translating a single Japanese sentence 
using the Graph-Augmented NLLB model and visualizing its semantic graph.
"""

import argparse
import yaml
import torch
import networkx as nx
import matplotlib.pyplot as plt
from preprocessing.tokenizer import TranslationTokenizer
from models.graph.builder import GraphBuilder
from models.full_model.baseline import BaselineTranslator

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def visualize_graph(graph: dict, text: str, output_path: str = "graph_visualization.png"):
    """
    Visualizes the parsed semantic graph using networkx.
    """
    if "node_spans" not in graph or not graph["node_spans"]:
        print("No graph nodes extracted.")
        return

    G = nx.DiGraph()
    
    # 1. Add nodes
    for i, span in enumerate(graph["node_spans"]):
        # Extract text for the node
        node_text = text[span[0]:span[1]]
        G.add_node(i, label=node_text)
        
    # 2. Add edges
    edge_types = {1: "Dependency", 2: "Temporal", 3: "Coreference"}
    colors = {1: "gray", 2: "blue", 3: "green"}
    
    if "edge_index" in graph and graph["edge_index"].size(1) > 0:
        src = graph["edge_index"][0].tolist()
        tgt = graph["edge_index"][1].tolist()
        types = graph["edge_type"].tolist()
        
        for s, t, ty in zip(src, tgt, types):
            G.add_edge(s, t, label=edge_types.get(ty, "Unknown"), color=colors.get(ty, "black"))
            
    # 3. Draw
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=1.0)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.8)
    
    # Draw labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_family="sans-serif")
    
    # Draw edges
    edges = G.edges()
    edge_colors = [G[u][v]['color'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrowsize=20, arrowstyle='->')
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title("Semantic Discourse Graph Visualization")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Graph visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single sentence")
    parser.add_argument("--text", type=str, required=True, help="Japanese sentence to translate")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline/checkpoint-best.pt", help="Path to model checkpoint")
    parser.add_argument("--output_image", type=str, default="graph_visualization.png", help="Path to save the graph visualization")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading tokenizer...")
    tokenizer = TranslationTokenizer(
        model_name=config["model"]["model_name"],
        src_lang=config["model"]["src_lang"],
        tgt_lang=config["model"]["tgt_lang"],
        max_length=config["model"]["max_length"]
    )
    
    print("Loading Graph Builder...")
    graph_builder = GraphBuilder()
    
    print("Initializing Model...")
    model = BaselineTranslator(model_name=config["model"]["model_name"])
    
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}. Using raw model.")
        
    model = model.to(device)
    model.eval()
    
    # Process text
    print(f"Input: {args.text}")
    encoded = tokenizer.tokenize_source(args.text)
    input_ids = encoded["input_ids"].unsqueeze(0).to(device)
    attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)
    
    graph = graph_builder.build_graph(args.text)
    
    # Format graph for batch of size 1
    if graph["node_spans"]:
        # Prepare mock offset_mapping (not heavily used in generate but needed for compatibility if graph adapter requires it)
        graph["batch_index"] = torch.zeros(len(graph["node_spans"]), dtype=torch.long, device=device)
        graph["local_node_index"] = torch.arange(len(graph["node_spans"]), dtype=torch.long, device=device)
        
        if len(graph["edge_index"]) > 0:
            graph["edge_index"] = torch.tensor(graph["edge_index"], dtype=torch.long, device=device)
            graph["edge_type"] = torch.tensor(graph["edge_type"], dtype=torch.long, device=device)
        else:
            graph["edge_index"] = torch.empty((2, 0), dtype=torch.long, device=device)
            graph["edge_type"] = torch.empty((0,), dtype=torch.long, device=device)
            
        graph["node_spans"] = [graph["node_spans"]] # Wrap in batch list
    else:
        graph = None
        
    # Generate
    with torch.no_grad():
        if config["training"].get("mixed_precision", True):
            with torch.autocast(device_type=device if device != "cpu" else "cpu"):
                generated_tokens = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    graph=graph,
                    max_length=config["model"]["max_length"]
                )
        else:
            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph=graph,
                max_length=config["model"]["max_length"]
            )
            
    translation = tokenizer.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print(f"\nTranslation: {translation}\n")
    
    if graph is not None:
        visualize_graph(
            {"node_spans": graph["node_spans"][0], "edge_index": graph["edge_index"], "edge_type": graph["edge_type"]}, 
            args.text, 
            args.output_image
        )

if __name__ == "__main__":
    main()
