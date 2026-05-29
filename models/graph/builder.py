"""
models/graph/builder.py

Builds a semantic graph (Entities and Events) from Japanese text using spaCy (`ja_ginza`).
Ensures the graph is sparse and focuses on discourse structure rather than token-level dependencies.
"""

import spacy
from typing import List, Dict, Tuple, Any

class GraphBuilder:
    def __init__(self, spacy_model: str = "ja_core_news_sm"):
        # Load spaCy model for Japanese
        self.nlp = spacy.load(spacy_model)
        
        # Define edge type mapping to integers
        self.edge_type_map = {
            "DEPENDENCY": 1,
            "TEMPORAL": 2,
            "SAME_ENTITY": 3
        }
        
    def build_graph(self, text: str) -> Dict[str, Any]:
        """
        Parses text and returns a list of nodes and edges.
        Node: {"id": int, "type": str, "text": str, "char_span": (start, end)}
        Edge: {"source": int, "target": int, "type": int}
        """
        doc = self.nlp(text)
        
        nodes = []
        edges = []
        node_id_counter = 0
        
        # Token to Node mapping to easily build edges later
        token2node = {}
        
        # 1. Extract Event Nodes (Verbs)
        last_event_id = None
        for token in doc:
            if token.pos_ == "VERB":
                nodes.append({
                    "id": node_id_counter,
                    "type": "EVENT",
                    "text": token.text,
                    "char_span": (token.idx, token.idx + len(token.text))
                })
                token2node[token.i] = node_id_counter
                
                # Add TEMPORAL edge to previous event (sequential continuity)
                if last_event_id is not None:
                    edges.append({
                        "source": last_event_id,
                        "target": node_id_counter,
                        "type": self.edge_type_map["TEMPORAL"]
                    })
                last_event_id = node_id_counter
                node_id_counter += 1
                
        # 2. Extract Entity Nodes (Nouns, PROPN, Entities)
        # We use a set to avoid duplicate node spans if `doc.ents` overlaps with `token.pos_ == "NOUN"`
        extracted_spans = set()
        
        # First, named entities
        for ent in doc.ents:
            nodes.append({
                "id": node_id_counter,
                "type": "ENTITY",
                "text": ent.text,
                "char_span": (ent.start_char, ent.end_char)
            })
            for i in range(ent.start, ent.end):
                token2node[i] = node_id_counter
            extracted_spans.add((ent.start_char, ent.end_char))
            node_id_counter += 1
            
        # Then, significant nouns not covered by entities
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                char_span = (token.idx, token.idx + len(token.text))
                
                # Check for overlap
                overlap = False
                for start, end in extracted_spans:
                    if not (char_span[1] <= start or char_span[0] >= end):
                        overlap = True
                        break
                        
                if not overlap:
                    nodes.append({
                        "id": node_id_counter,
                        "type": "ENTITY",
                        "text": token.text,
                        "char_span": char_span
                    })
                    token2node[token.i] = node_id_counter
                    extracted_spans.add(char_span)
                    node_id_counter += 1
                    
        # 3. Extract Dependency Edges
        # We trace paths from Entity nodes to their syntactic head Event nodes
        for token in doc:
            if token.i in token2node and token.pos_ in ["NOUN", "PROPN", "PRON"]:
                entity_node_id = token2node[token.i]
                
                # Trace up to find a verb
                head = token.head
                while head.i != head.head.i and head.pos_ != "VERB":
                    head = head.head
                    
                if head.pos_ == "VERB" and head.i in token2node:
                    event_node_id = token2node[head.i]
                    edges.append({
                        "source": entity_node_id,
                        "target": event_node_id,
                        "type": self.edge_type_map["DEPENDENCY"]
                    })
                    # Bidirectional dependency
                    edges.append({
                        "source": event_node_id,
                        "target": entity_node_id,
                        "type": self.edge_type_map["DEPENDENCY"]
                    })
                    
        # 4. Extract Coreference (Same-Entity) Edges
        # Group entity nodes by exact text match
        entity_groups = {}
        for node in nodes:
            if node["type"] == "ENTITY":
                entity_groups.setdefault(node["text"], []).append(node["id"])
                
        for text, ids in entity_groups.items():
            if len(ids) > 1:
                # Fully connect identical entities (clique)
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        edges.append({
                            "source": ids[i],
                            "target": ids[j],
                            "type": self.edge_type_map["SAME_ENTITY"]
                        })
                        edges.append({
                            "source": ids[j],
                            "target": ids[i],
                            "type": self.edge_type_map["SAME_ENTITY"]
                        })
                        
        return {
            "nodes": nodes,
            "edges": edges
        }
