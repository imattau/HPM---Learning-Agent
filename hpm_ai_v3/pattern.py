from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, Any, Optional, List
import uuid
import time
import networkx as nx

from .device_utils import get_device, get_optimal_dtype

class HPMPattern(ABC):
    """Abstract base for all HPM patterns."""
    _global_device: Optional[torch.device] = None
    _global_dtype: Optional[torch.dtype] = None

    @classmethod
    def set_device(cls, device: torch.device):
        cls._global_device = device
        cls._global_dtype = get_optimal_dtype(device)

    def __init__(self, pattern_id: Optional[str] = None):
        self.id = pattern_id or str(uuid.uuid4())[:8]
        self._device = self._global_device or get_device(verbose=False)
        self._dtype = self._global_dtype or torch.float32
        self.weight = 0.01
        self.substrate_type: str = "neural"
        self.required_observation_keys: List[str] = []
        
        self.loss_ema: Optional[float] = None
        self.accuracy: float = -10.0  # Penalty until first update
        self.affective_score: float = 0.0
        self.social_score: float = 0.0
        self.curiosity_reward: float = 0.0
        self.coherence_score: float = 0.0
        self.insight_boost: float = 0.0
        self.invariance_score: float = 0.0
        
        self.structural_connectivity: float = 0.0
        self.evaluator_reinforcement: float = 0.0
        self.field_amplification: float = 0.0
        self.density_weight: float = 0.2
        self.stickiness: float = 0.0

        self.birth_time: float = time.time()
        self.last_used: float = self.birth_time
        self.use_count: int = 0
        self.compilation_count: int = 0
        
    def mark_used(self):
        self.last_used = time.time()
        self.use_count += 1

    def to(self, device: torch.device):
        """Move pattern parameters to device (override in subclasses)."""
        self._device = device
        return self

    @property
    def pattern_density(self) -> float:
        return (0.4 * self.structural_connectivity + 
                0.3 * self.evaluator_reinforcement + 
                0.3 * self.field_amplification)
    
    def compute_stickiness(self, base_loss: float) -> float:
        from scipy.special import expit
        eta, delta = 2.0, 0.5
        safe_loss = base_loss if np.isfinite(base_loss) else 1.0
        x = eta * self.pattern_density - delta * safe_loss
        self.stickiness = float(expit(x))
        return self.stickiness

    @abstractmethod
    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor: pass
    @abstractmethod
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]: pass
    @abstractmethod
    def intervene(self, intervention: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, torch.Tensor]: pass
    @abstractmethod
    def update_parameters(self, observations: Dict[str, torch.Tensor], learning_rate: float = 0.01): pass
    @abstractmethod
    def structural_distance(self, other: 'HPMPattern') -> float: pass
    @abstractmethod
    def extract_causal_graph(self) -> nx.DiGraph: pass

    def batch_update(self, observations_batch: List[Dict[str, torch.Tensor]], learning_rate: float = 0.01):
        for obs in observations_batch:
            self.update_parameters(obs, learning_rate)

    def total_score(self, beta_aff=0.3, gamma_soc=0.1, delta_cur=0.2, eta_coh=0.2, zeta_ins=0.5, zeta_inv=0.3) -> float:
        return (self.accuracy + beta_aff * self.affective_score + 
                gamma_soc * self.social_score + delta_cur * self.curiosity_reward +
                eta_coh * self.coherence_score + zeta_ins * self.insight_boost +
                zeta_inv * self.invariance_score)

    def filter_observations(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in observations.items() if k in self.required_observation_keys}
