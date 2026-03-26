from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional, Any, Set

class SubstrateID(Enum):
    INTERNAL_FLEX = auto()  # Neural-like. High decay, high recombination flexibility.
    INTERNAL_PROC = auto()  # Motor-like. Low decay, zero flexibility.
    EXTERNAL_SYM = auto()   # Symbolic/Shared. Zero decay, high verification requirement.

@dataclass
class SubstrateProperties:
    decay_rate: float
    resource_cost: float
    recombination_flexibility: float
    stability_gain: float

SUBSTRATE_REGISTRY: Dict[SubstrateID, SubstrateProperties] = {
    SubstrateID.INTERNAL_FLEX: SubstrateProperties(
        decay_rate=0.01,
        resource_cost=1.0,
        recombination_flexibility=1.0,
        stability_gain=0.1
    ),
    SubstrateID.INTERNAL_PROC: SubstrateProperties(
        decay_rate=0.001,
        resource_cost=0.1,  # 10x cheaper
        recombination_flexibility=0.0,
        stability_gain=0.5
    ),
    SubstrateID.EXTERNAL_SYM: SubstrateProperties(
        decay_rate=0.0,
        resource_cost=0.5,
        recombination_flexibility=0.5,
        stability_gain=1.0
    )
}

@dataclass
class CompositePattern:
    """
    HPM v1.25 Composite Pattern (h).
    Factorizes as p(x, z_1, z_2) = p(z_2) * p(z_1 | z_2) * p(x | z_1).
    """
    id: str
    level: int  # Developmental level [1-5]
    substrate_id: SubstrateID
    
    # Generative model parameters (theta)
    # In a real implementation, these would be probability distributions
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Constituent Features (S)
    constituent_features: Set[str] = field(default_factory=set)
    
    # Weight in the population (w_i)
    weight: float = 0.01
    
    # Moving average metrics
    ema_loss: float = 0.0
    ema_hier_loss: float = 0.0
    ema_compression: float = 0.0
    
    # Evaluator scores (J_i)
    affective_score: float = 0.0
    social_score: float = 0.0
    generative_utility: float = 0.0 # Only for Level 5
    
    @property
    def resource_cost(self) -> float:
        base_cost = SUBSTRATE_REGISTRY[self.substrate_id].resource_cost
        # Level-based penalty (Working memory load) - Reduced multiplier
        return base_cost * (1.0 + (self.level - 1) * 0.1)

    def __post_init__(self):
        pass

    @property
    def sensitivity(self) -> tuple[float, float]:
        """
        Returns (alpha_L, beta_L): Surface vs Structural sensitivity.
        Level 1: Surface dominance.
        Level 4: Internalized Schemas.
        Level 5: Pure Generative Structure.
        """
        if self.level == 1:
            return (0.9, 0.1)
        if self.level == 2:
            return (0.6, 0.4)
        if self.level == 3:
            return (0.2, 0.8)
        if self.level == 4:
            return (0.05, 0.95)
        return (0.01, 0.99) # Level 5

    @property
    def accuracy(self) -> float:
        return -self.ema_loss

    @property
    def total_score(self) -> float:
        """
        Total_i(t) = A_i(t) + J_i(t)
        Where A_i(t) = alpha * Surface_Accuracy + beta * Structural_Fitness
        """
        alpha, beta = self.sensitivity
        
        # Surface component
        surface_fitness = self.accuracy # -ema_loss
        
        # Structural component (Hierarchical fit)
        structural_fitness = 0.0
        if self.level > 1:
            # Equation A.3 & D7: -L_hier + Comp
            structural_fitness = -self.ema_hier_loss + self.ema_compression
            
        epistemic_total = alpha * surface_fitness + beta * structural_fitness
            
        # J_i = E_aff + E_soc - E_res + (Generative_Utility if L5)
        j_i = self.affective_score + self.social_score - self.resource_cost
        if self.level >= 5:
            j_i += self.generative_utility
            
        return epistemic_total + j_i


    def calculate_connectivity(self, dependency_matrix: np.ndarray) -> float:
        """
        C(h): Graph density of the parameter dependency matrix.
        """
        if dependency_matrix.size == 0:
            return 0.0
        n = len(self.constituent_features)
        if n <= 1:
            return 1.0 # Single feature is fully connected to itself
        
        edges = np.count_nonzero(dependency_matrix)
        max_edges = n * (n - 1) # Assuming directed graph
        return edges / max_edges if max_edges > 0 else 0.0

    def calculate_density(self, connectivity: float, social_frequency: float) -> float:
        """
        D(h) = alpha * C(h) + beta * E(h) + gamma * F(h)
        """
        # Placeholders for alpha, beta, gamma weights
        alpha, beta, gamma = 0.4, 0.3, 0.3
        evaluator_reinforcement = self.affective_score + self.social_score
        return (alpha * connectivity + 
                beta * evaluator_reinforcement + 
                gamma * social_frequency)
