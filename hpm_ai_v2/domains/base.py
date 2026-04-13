"""
Base DomainConfig for HPM agents.
Encapsulates domain-specific semantics, dimensions and concept mappings.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict

class DomainConfig:
    """
    Configuration object defining the manifold structure and semantics for a domain.
    """
    def __init__(self, concepts: List[str], s_dim: int = 20):
        self.concepts = concepts
        self.concept_idx = {c: i for i, c in enumerate(concepts)}
        self.DIM = len(concepts)
        self.S_DIM = s_dim
        # m_dim = [S_DIM (state) | DIM (concept) | S_DIM (delta)]
        self.m_dim = self.S_DIM + self.DIM + self.S_DIM
        
        # Standard structural indices (legacy compatibility)
        self.STRUCT_DIMS = [0] + list(range(10, 17))
        self.STRUCTURE_SLICE = slice(10, 17)

    def get_concept_vector(self, concept: str) -> np.ndarray:
        """Return a one-hot vector for the given concept."""
        vec = np.zeros(self.DIM)
        if concept in self.concept_idx:
            vec[self.concept_idx[concept]] = 1.0
        return vec

    def get_concept_name(self, index: int) -> str | None:
        """Return the name of the concept at the given index."""
        if 0 <= index < self.DIM:
            return self.concepts[index]
        return None
