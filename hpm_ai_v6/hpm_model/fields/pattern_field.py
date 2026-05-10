import numpy as np
import torch
from typing import List, Dict, Any
from hpm_ai_v6.hpm_model.core.cell import Cell

class DynamicPatternField:
    """
    Implements a dynamic pattern field (Section 2.5.4).
    The field amplifies patterns that are socially rewarded or gain high population weight.
    This acts as a 'cultural' substrate that biases learning.
    """
    def __init__(self, 
                 initial_bias: float = 0.1, 
                 decay: float = 0.95, 
                 influence_rate: float = 0.05):
        self.amplification: Dict[str, float] = {}
        self.initial_bias = initial_bias
        self.decay = decay
        self.influence_rate = influence_rate

    def update(self, patterns: List[Cell], weights: np.ndarray, social_scores: np.ndarray):
        """
        Update field amplifications based on current population state.
        Field(h, t+1) = decay * Field(h, t) + influence * (social_score + population_weight)
        """
        for i, p in enumerate(patterns):
            current = self.amplification.get(p.name, self.initial_bias)
            # Field grows with social approval and current adoption (weight)
            inc = self.influence_rate * (social_scores[i] + weights[i])
            self.amplification[p.name] = current * self.decay + inc

    def update_tensor(self, patterns: List[Cell], weights: torch.Tensor, social_scores: torch.Tensor):
        weights = weights.detach().cpu()
        social_scores = social_scores.detach().cpu()
        for i, p in enumerate(patterns):
            current = self.amplification.get(p.name, self.initial_bias)
            inc = self.influence_rate * (float(social_scores[i]) + float(weights[i]))
            self.amplification[p.name] = current * self.decay + inc

    def get_amplifications(self, patterns: List[Cell]) -> Dict[str, float]:
        """Returns the current amplification factors for the given patterns."""
        return {p.name: float(self.amplification.get(p.name, 0.0)) for p in patterns}

    def get_amplification_vector(self, patterns: List[Cell]) -> np.ndarray:
        """Returns the current amplification factors as a numpy array."""
        return np.array([self.amplification.get(p.name, 0.0) for p in patterns])

    def get_amplification_tensor(self, patterns: List[Cell], *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.tensor([self.amplification.get(p.name, 0.0) for p in patterns], dtype=dtype)
