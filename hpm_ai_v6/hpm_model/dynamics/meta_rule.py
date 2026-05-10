import numpy as np
import torch
from typing import List, Optional, Dict
from hpm_ai_v6.hpm_model.core.cell import Cell

class MetaPatternRule:
    """
    Implements the Meta Pattern Rule for pattern population dynamics.
    Covers replicator dynamics (Appendix D.5), conflict inhibition, 
    and exponential forgetting (decay).
    """
    def __init__(self, 
                 patterns: List[Cell], 
                 learning_rate: float = 0.2, 
                 conflict_scale: float = 0.05, 
                 forget_decay: float = 1.0,
                 field_strength: float = 0.2):
        self.patterns = patterns
        self.n = len(patterns)
        self.eta = learning_rate
        self.beta_c = conflict_scale
        self.decay = forget_decay
        self.field_strength = field_strength
        self._weights = torch.ones(self.n, dtype=torch.float32) / (self.n + 1e-9)
        self._kappa = torch.zeros((self.n, self.n), dtype=torch.float32)

    @property
    def weights(self) -> np.ndarray:
        return self._weights.detach().cpu().numpy().copy()

    @weights.setter
    def weights(self, value):
        tensor = torch.as_tensor(value, dtype=torch.float32)
        self._weights = tensor.clone()

    @property
    def kappa(self) -> np.ndarray:
        return self._kappa.detach().cpu().numpy().copy()

    def get_weights_tensor(self) -> torch.Tensor:
        return self._weights

    def set_weights_tensor(self, value: torch.Tensor):
        self._weights = value.detach().clone().to(dtype=torch.float32)

    def get_kappa_tensor(self) -> torch.Tensor:
        return self._kappa

    def set_incompatibility(self, i: int, j: int, value: float):
        """Sets the conflict level between pattern i and j."""
        self._kappa[i, j] = value
        self._kappa[j, i] = value

    def update_weights(self, scores: np.ndarray, field_amplifications: np.ndarray):
        """
        Updates pattern weights using replicator dynamics with conflict and field influence.
        Section D.5, D.6.
        """
        # Augmented scores: Utility + Field Support
        score_t = torch.as_tensor(scores, dtype=torch.float32)
        field_t = torch.as_tensor(field_amplifications, dtype=torch.float32)
        augmented_scores = score_t + self.field_strength * field_t

        avg_score = torch.dot(self._weights, augmented_scores)
        conflict = torch.matmul(self._kappa, self._weights)

        new_weights = (
            self.decay * self._weights
            + self.eta * (augmented_scores - avg_score) * self._weights
            - self.beta_c * conflict * self._weights
        )

        new_weights = torch.clamp(new_weights, min=0.0)
        total = torch.sum(new_weights)
        if float(total) > 0.0:
            self._weights = new_weights / total
        else:
            self._weights = torch.ones(self.n, dtype=torch.float32) / max(self.n, 1)

    def update_weights_tensor(self, scores: torch.Tensor, field_amplifications: torch.Tensor):
        augmented_scores = scores.to(dtype=torch.float32) + self.field_strength * field_amplifications.to(dtype=torch.float32)

        avg_score = torch.dot(self._weights, augmented_scores)
        conflict = torch.matmul(self._kappa, self._weights)

        new_weights = (
            self.decay * self._weights
            + self.eta * (augmented_scores - avg_score) * self._weights
            - self.beta_c * conflict * self._weights
        )

        new_weights = torch.clamp(new_weights, min=0.0)
        total = torch.sum(new_weights)
        if float(total) > 0.0:
            self._weights = new_weights / total
        else:
            self._weights = torch.ones(self.n, dtype=torch.float32) / max(self.n, 1)

    def get_best_pattern(self) -> Cell:
        """Returns the pattern with the highest current weight."""
        return self.patterns[int(torch.argmax(self._weights).item())]

    def get_weights_dict(self) -> Dict[str, float]:
        """Returns a mapping of pattern names to their weights."""
        return {p.name: float(self.weights[i]) for i, p in enumerate(self.patterns)}
