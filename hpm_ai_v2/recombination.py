from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Any, Set
from .patterns import CompositePattern, SubstrateID

class StructuralRecombinator:
    """
    Operator R: H x H -> H subject to constraints C.
    Implements structural merging of constituent features.
    """
    def __init__(self, innovation_rate: float = 0.1):
        self.innovation_rate = innovation_rate
        
    def recombine(self, h_a: CompositePattern, h_b: CompositePattern) -> Optional[CompositePattern]:
        """
        Creates a new pattern h* by merging constituents of h_a and h_b.
        """
        # 1. Structural Constraint Check C(h*)
        # Simplified: Merged pattern must not exceed a certain complexity at Level 1
        new_features = h_a.constituent_features.union(h_b.constituent_features)
        
        if len(new_features) > 10: # Example constraint
            return None
            
        # 2. Create New Pattern
        new_id = f"recomb_{h_a.id}_{h_b.id}"
        new_level = max(h_a.level, h_b.level)
        
        # New pattern starts in flexible internal substrate
        h_star = CompositePattern(
            id=new_id,
            level=new_level,
            substrate_id=SubstrateID.INTERNAL_FLEX,
            constituent_features=new_features
        )
        
        # 3. Insight Evaluation (Appendix E4)
        insight_score = self.calculate_insight(h_a, h_b, h_star)
        h_star.affective_score = insight_score
        
        # Level 5 Patterns gain Generative Utility for successful contributions
        if h_a.level >= 5:
            h_a.generative_utility += insight_score * 0.1
        if h_b.level >= 5:
            h_b.generative_utility += insight_score * 0.1
            
        # Initial weight is proportional to insight
        h_star.weight = 0.01 * insight_score
        
        return h_star

    def calculate_insight(self, h_a: CompositePattern, h_b: CompositePattern, h_star: CompositePattern) -> float:
        """
        I(h*) = beta_orig * (alpha_nov * Nov(h*) + alpha_eff * Eff(h*))
        """
        # Novelty: Inverse of common features
        common = h_a.constituent_features.intersection(h_b.constituent_features)
        total = h_star.constituent_features
        novelty = 1.0 - (len(common) / len(total)) if len(total) > 0 else 0.0
        
        # Effectiveness: Placeholder (assumed 1.0 for this mock)
        effectiveness = 1.0
        
        beta_orig = 1.0
        alpha_nov = 0.5
        alpha_eff = 0.5
        
        return beta_orig * (alpha_nov * novelty + alpha_eff * effectiveness)
