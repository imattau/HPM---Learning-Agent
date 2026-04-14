"""
GraphOracle — computes empirical state vector for NetworkX graphs.
"""
from __future__ import annotations

import numpy as np
from typing import Any, List, Optional, TYPE_CHECKING
from .base import BaseOracle

if TYPE_CHECKING:
    from hpm_ai_v2.domains.base import DomainConfig

class GraphOracle(BaseOracle):
    """Computes empirical state vector for NetworkX graphs."""

    def __init__(self, config: "DomainConfig"):
        self.config = config

    def compute_state(
        self,
        outputs: List[Any],
        errors: List[Optional[str]],
        code: str = "",
    ) -> np.ndarray:
        s_dim = self.config.S_DIM
        s = np.zeros(s_dim)
        valid_outputs = [o for o, e in zip(outputs, errors) if e is None]
        if not valid_outputs:
            s[0] = 0.0
            return s
        s[0] = 1.0
        
        import networkx as nx
        n_nodes = []
        n_edges = []
        avg_degs = []
        node_labels = []
        for G in valid_outputs:
            if isinstance(G, nx.Graph):
                try:
                    n = G.number_of_nodes()
                    e = G.number_of_edges()
                    n_nodes.append(n)
                    n_edges.append(e)
                    avg_degs.append(2.0 * e / n if n > 0 else 0.0)
                    
                    # Extract numeric labels if possible
                    labels = [n for n in G.nodes() if isinstance(n, (int, float))]
                    if labels:
                        node_labels.extend(labels)
                except Exception:
                    continue
        
        if n_nodes:
            # Normalized stats
            s[3] = float(np.mean(n_nodes)) / 100.0
            s[4] = float(np.mean(n_edges)) / 100.0
            s[5] = float(np.mean(avg_degs)) / 10.0
            
            if node_labels:
                s[6] = float(np.mean(node_labels)) / 100.0
                s[7] = float(np.min(node_labels)) / 100.0
                s[8] = float(np.max(node_labels)) / 100.0
            
            # binary flags for simple structural features
            s[10] = 1.0 if any(n > 0 for n in n_nodes) else 0.0
            s[11] = 1.0 if any(e > 0 for e in n_edges) else 0.0
            
        return s
