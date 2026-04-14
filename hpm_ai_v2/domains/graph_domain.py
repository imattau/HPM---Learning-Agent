"""
GraphDomainConfig — Configuration for NetworkX-based graph transformations.
"""
from __future__ import annotations
import numpy as np
from typing import List
from hpm_ai_v2.domains.base import DomainConfig
from hfn.hfn import HFN

class GraphDomainConfig(DomainConfig):
    """
    Manifold structure for graph domain.
    """
    def __init__(self):
        concepts = [
            "ADD_NODE", "REMOVE_NODE",
            "ADD_EDGE", "REMOVE_EDGE",
            "CLEAR_GRAPH", "COPY_GRAPH"
        ]
        # S_DIM=20, DIM=len(concepts), m_dim = S_DIM + DIM + S_DIM
        super().__init__(concepts, s_dim=20)

def get_graph_primitive_nodes(config: GraphDomainConfig) -> List[HFN]:
    """
    Instantiate one HFN node per primitive concept defined in the config.
    """
    nodes = []
    for i, concept in enumerate(config.concepts):
        mu = np.zeros(config.m_dim)
        # concept one-hot in middle slice
        mu[config.S_DIM + i] = 1.0
        node = HFN(
            mu=mu,
            sigma=np.ones(config.m_dim),
            id=f"graph_op_{concept}",
            use_diag=True
        )
        nodes.append(node)
    return nodes
