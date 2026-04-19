"""
KnowledgeGraphDomainConfig — Configuration for factual reasoning via external Knowledge Graphs (Wikidata, ConceptNet).
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional
from hpm_ai_v2.domains.base import DomainConfig
from hfn.hfn import HFN

class KnowledgeGraphDomainConfig(DomainConfig):
    """
    Manifold structure for Knowledge Graph domain.
    Focuses on entities, relations, and structured factual lookup.
    """
    def __init__(self, concepts: Optional[List[str]] = None, s_dim: int = 4):
        if concepts is None:
            concepts = [
                "KG_ENTITY", "KG_PROPERTY", "KG_STATEMENT",
                "OP_SPARQL_QUERY", "OP_TRAVERSE", "OP_ENRICH",
                "REL_INSTANCE_OF", "REL_SUBCLASS_OF", "REL_PART_OF",
                "REL_CREATED_BY", "REL_DISCOVERED_BY"
            ]
        super().__init__(concepts, s_dim=s_dim)

    @property
    def domain_type(self) -> str:
        return "knowledge_graph"

    def encode_entity(self, label: str, description: str = "") -> np.ndarray:
        """
        Produce a vector representation for a KG entity.
        Combines label-based encoding with generic KG_ENTITY markers.
        """
        # For now, simple marker-based encoding
        mu = np.zeros(self.m_dim)
        mu[self.S_DIM + self.concepts.index("KG_ENTITY")] = 1.0
        # If we had a pre-trained embedder, we'd use it here to fill the rest of DIM
        return mu
