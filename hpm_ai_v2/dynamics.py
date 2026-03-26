from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Any
from .patterns import CompositePattern, SubstrateID, SUBSTRATE_REGISTRY

class MetaPatternRule:
    """
    HPM v1.25 Meta Pattern Rule Dynamics.
    Implements replicator style weight updates with conflict inhibition
    and substrate-specific decay.
    """
    def __init__(self, 
                 learning_rate: float = 0.5, 
                 conflict_scale: float = 0.5,
                 density_threshold: float = 0.8,
                 stability_kappa: float = 0.1):
        self.learning_rate = learning_rate
        self.conflict_scale = conflict_scale
        self.density_threshold = density_threshold
        self.stability_kappa = stability_kappa
        
    def update_weights(self, 
                       patterns: List[CompositePattern], 
                       kappa_matrix: np.ndarray) -> None:
        """
        w_i(t+1) = w_i(t) + eta*(Total_i - Total_avg)*w_i - beta_c * sum(kappa_ij * w_i * w_j)
        Plus Stability Bias: Total_i = Total_i + kappa_D * D(h_i)
        """
        if not patterns:
            return
            
        weights = np.array([p.weight for p in patterns])
        
        # Calculate scores with maturation penalty and stability bias
        scores = []
        for p in patterns:
            penalty = self.calculate_maturation_gate(p.level, patterns)
            
            # D(h) = alpha * C(h) + beta * E(h) + gamma * F(h)
            density = p.calculate_density(connectivity=1.0, social_frequency=0.5)
            
            # Apply subtractive penalty and additive stability bias
            final_score = p.total_score - penalty + self.stability_kappa * density
            
            scores.append(final_score)
        scores = np.array(scores)
        
        # Calculate population average total score
        total_avg = np.sum(weights * scores) / np.sum(weights) if np.sum(weights) > 0 else 0
        
        new_weights = np.copy(weights)
        
        # Calculate exponents for softmax
        # w_i(t+1) = w_i(t) * exp(learning_rate * score_i) / sum(w_j * exp(learning_rate * score_j))
        exponents = np.exp(self.learning_rate * scores)
        new_weights = weights * exponents
        
        # Conflict inhibition (subtractive on the result)
        if kappa_matrix.size > 0:
            inhibition = self.conflict_scale * (kappa_matrix @ weights)
            new_weights *= np.maximum(0.1, 1.0 - inhibition)
            
        # Apply substrate-specific decay
        for i, p in enumerate(patterns):
            props = SUBSTRATE_REGISTRY[p.substrate_id]
            new_weights[i] *= (1.0 - props.decay_rate)
            
        # Ensure non-negative weights
        new_weights = np.maximum(new_weights, 0.0)
        total_w = np.sum(new_weights)
        if total_w > 0:
            new_weights /= total_w
        
        for i, p in enumerate(patterns):
            p.weight = new_weights[i]

    def check_substrate_shift(self, p: CompositePattern, connectivity: float, social_freq: float) -> bool:
        """
        A pattern shifts substrate when Density D(h) exceeds a threshold.
        INTERNAL_FLEX -> INTERNAL_PROC (Internalization)
        INTERNAL_PROC -> EXTERNAL_SYM (Externalization)
        """
        density = p.calculate_density(connectivity, social_freq)
        
        if p.substrate_id == SubstrateID.INTERNAL_FLEX:
            if density >= self.density_threshold:
                p.substrate_id = SubstrateID.INTERNAL_PROC
                return True
        elif p.substrate_id == SubstrateID.INTERNAL_PROC:
            # Externalization requires high social evaluator
            if p.social_score >= self.density_threshold and density >= self.density_threshold:
                p.substrate_id = SubstrateID.EXTERNAL_SYM
                return True
        return False
        
    def calculate_maturation_gate(self, level: int, population: List[CompositePattern]) -> float:
        """
        Returns a penalty value based on lower-level density foundation.
        Returns 0.0 (no penalty) if lower level foundation is solid.
        Returns a positive value (subtractive penalty) if foundation is weak.
        """
        if level <= 1:
            return 0.0
            
        # Check density of patterns at level - 1
        lower_level_patterns = [p for p in population if p.level == level - 1]
        if not lower_level_patterns:
            return 5.0 # High subtractive penalty if no lower level patterns exist
            
        # Calculate average density of lower level patterns
        avg_density = np.mean([p.calculate_density(1.0, 0.0) for p in lower_level_patterns])
        
        # If avg_density < threshold, apply subtractive penalty
        if avg_density < self.density_threshold:
            return 5.0 * (1.0 - (avg_density / self.density_threshold))
            
        return 0.0

    def detect_conflict(self, h_i: CompositePattern, h_j: CompositePattern) -> float:
        """
        Calculates incompatibility kappa_ij based on Hellinger Distance 
        between predictive distributions.
        """
        # If levels are different, they are naturally less incompatible (different niches)
        if h_i.level != h_j.level:
            return 0.1
            
        # In a full system, we would get the predictive distribution object from params
        # For this simulation, we check if they have different constituent features
        if h_i.constituent_features != h_j.constituent_features:
            # High Hellinger distance placeholder
            return 0.8
            
        return 0.0
