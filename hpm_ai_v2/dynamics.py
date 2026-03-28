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
            gate_penalty = self.calculate_maturation_gate(p.level, patterns)
            gate_multiplier = 1.0 / (1.0 + gate_penalty)
            
            # D(h) = alpha * C(h) + beta * E(h) + gamma * F(h)
            density = p.calculate_density(connectivity=1.0, social_frequency=0.5)
            
            # Apply Stability Bias to the total score
            biased_score = p.total_score + self.stability_kappa * density
            
            # Maturation Gate scales the score (0.0 to 1.0)
            scores.append(biased_score * gate_multiplier)
        scores = np.array(scores)
        
        # Calculate population average total score
        total_avg = np.sum(weights * scores) / np.sum(weights) if np.sum(weights) > 0 else 0
        
        adjusted_scores = np.copy(scores)
        
        for i, p in enumerate(patterns):
            # Conflict inhibition term lowers the effective score of mutually incompatible patterns.
            if kappa_matrix.size > 0:
                adjusted_scores[i] -= self.conflict_scale * np.sum(kappa_matrix[i, :] * weights)
            
            # Apply substrate-specific decay as a score penalty.
            props = SUBSTRATE_REGISTRY[p.substrate_id]
            adjusted_scores[i] -= props.decay_rate
        
        # Softmax-style competition over adjusted scores.
        competition_temperature = 8.0
        exponents = np.exp(self.learning_rate * competition_temperature * adjusted_scores)
        new_weights = exponents / np.sum(exponents)
        
        # Maintain a minimal exploration floor to avoid permanent extinction.
        new_weights = np.maximum(new_weights, 1e-3)
        new_weights /= np.sum(new_weights)
        
        for i, p in enumerate(patterns):
            p.weight = new_weights[i]

    def check_substrate_shift(self, p: CompositePattern, connectivity: float, social_freq: float) -> bool:
        """
        A pattern shifts substrate when Density D(h) exceeds a threshold.
        INTERNAL_FLEX -> INTERNAL_PROC (Internalization)
        INTERNAL_PROC -> EXTERNAL_SYM (Externalization)
        Only one shift per call to maintain developmental sequence.
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
        Returns a developmental penalty based on lower-level density foundation.
        Returns 0.0 when the lower-level foundation is solid.
        Returns a positive value when the foundation is weak or missing.
        """
        if level <= 1:
            return 0.0
            
        # Check density across all lower-level patterns.
        # This allows a strong Level 1 foundation to support later levels
        # even if an intermediate level is not yet instantiated.
        lower_level_patterns = [p for p in population if p.level < level]
        if not lower_level_patterns:
            return 1.0 # Hard lock if no lower level patterns exist
            
        # Calculate the strongest lower-level density.
        # This treats maturation as requiring at least one solid foundation,
        # rather than averaging in weak but irrelevant patterns.
        foundation_density = max(p.calculate_density(1.0, 0.0) for p in lower_level_patterns)
        
        # Weak foundations incur a penalty that decays as density approaches the threshold.
        if foundation_density < self.density_threshold:
            return self.density_threshold / max(foundation_density, 1e-6)
            
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
