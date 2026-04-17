"""
HighDimDomainConfig — Configuration for high-dimensional scaling experiments (SP100).
"""
from __future__ import annotations
from typing import List, Dict
from hpm_ai_v2.domains.base import DomainConfig

class HighDimDomainConfig(DomainConfig):
    """
    Configuration for high-dimensional input spaces.
    Generates SELECT_i concepts for each dimension.
    """
    def __init__(self, D: int, s_dim: Optional[int] = None):
        self.D = D
        # Default s_dim to D + 20 to allow encoding correlations for all dimensions
        if s_dim is None:
            s_dim = D + 20
        # Dynamic concepts: SELECT(0)...SELECT(D-1)
        select_concepts = [f"SELECT_{i}" for i in range(D)]
        base_concepts = [
            "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", 
            "OP_SIN", "OP_SQUARE", "OP_CONST"
        ]
        super().__init__(concepts=select_concepts + base_concepts, s_dim=s_dim)
        self.concept_idx = {c: i for i, c in enumerate(self.concepts)}
