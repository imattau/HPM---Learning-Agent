from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.dynamics.meta_rule import MetaPatternRule
from hpm_ai_v6.hpm_model.dynamics.learning import HPMLearner

class BaseHPMAgent(ABC):
    """
    Abstract Base Class for HPM Agents.
    Defines the standard interface for perceiving and learning from pattern sequences.
    """
    def __init__(self, 
                 patterns: List[Cell],
                 learning_rate: float = 0.2,
                 conflict_scale: float = 0.05,
                 forget_decay: float = 1.0,
                 beta_e: float = 1.0,
                 beta_a: float = 0.5,
                 beta_s: float = 0.5):
        self.patterns = patterns
        self.meta_rule = MetaPatternRule(
            patterns=self.patterns,
            learning_rate=learning_rate,
            conflict_scale=conflict_scale,
            forget_decay=forget_decay
        )
        self.learner = HPMLearner(
            meta_rule=self.meta_rule,
            beta_e=beta_e,
            beta_a=beta_a,
            beta_s=beta_s
        )

    @abstractmethod
    def perceive(self, observation_seq: List[Cell], population: List[Cell], context: Dict[str, Any]):
        """Agent processes a sequence of observations and updates its internal state."""
        pass

    def get_best_pattern(self) -> Cell:
        return self.meta_rule.get_best_pattern()

    def get_weights_tensor(self) -> torch.Tensor:
        return self.meta_rule.get_weights_tensor().clone()

    def get_weights(self) -> np.ndarray:
        return self.meta_rule.weights.copy()
