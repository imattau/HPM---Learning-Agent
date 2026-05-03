from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import networkx as nx
import numpy as np
from ..core import State
from .base import PolygraphView, PolygraphGenerator

@dataclass(slots=True)
class GraphPolygraphGenerator(PolygraphGenerator):
    """Generate views from a graph represented as adjacency dict or edge list."""
    def generate(self, raw: Any, *, context: dict[str, Any] | None = None) -> list[PolygraphView]:
        if isinstance(raw, dict):
            G = nx.from_dict_of_lists(raw)
        elif isinstance(raw, tuple) and len(raw) == 2:
            nodes, edges = raw
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
        else:
            raise TypeError("GraphPolygraphGenerator expects adjacency dict or (nodes, edges)")
        context = dict(context or {})
        context.setdefault("domain", "graph")
        views = []
        
        # 1. Adjacency matrix (flattened)
        adj = nx.to_numpy_array(G)
        adj_flat = tuple(map(float, adj.flatten()))
        views.append(PolygraphView(
            name="adjacency",
            state=State(value=adj_flat, context={**context, "view": "adj"}),
            context={**context, "view": "adj"},
        ))
        
        # 2. Degree sequence
        degrees = [d for n, d in G.degree()]
        views.append(PolygraphView(
            name="degrees",
            state=State(value=tuple(map(int, degrees)), context={**context, "view": "degrees"}),
            context={**context, "view": "degrees"},
        ))
        
        # 3. Betweenness centrality (flattened vector)
        bc = nx.betweenness_centrality(G)
        bc_list = [bc[n] for n in sorted(G.nodes())]
        views.append(PolygraphView(
            name="betweenness",
            state=State(value=tuple(map(float, bc_list)), context={**context, "view": "betweenness"}),
            context={**context, "view": "betweenness"},
        ))
        return views
