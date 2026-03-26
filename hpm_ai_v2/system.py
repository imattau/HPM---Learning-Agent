from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Any
from .patterns import CompositePattern, SubstrateID
from .dynamics import MetaPatternRule

class HPMSystem:
    """
    Main HPM System coordinating patterns, evaluators, and dynamics.
    """
    def __init__(self, 
                 patterns: List[CompositePattern], 
                 rule: MetaPatternRule,
                 ema_lambda: float = 0.1):
        self.patterns = patterns
        self.rule = rule
        self.ema_lambda = ema_lambda
        self.t = 0
        self.kappa_matrix = np.zeros((len(patterns), len(patterns)))
        
    def step(self, 
             surface_loss: np.ndarray, 
             structural_loss: Optional[np.ndarray] = None,
             compression: Optional[np.ndarray] = None,
             social_freq: Optional[np.ndarray] = None) -> None:
        """
        Processes a single observation x.
        """
        self.t += 1
        
        # 1. Update Epistemic metrics
        for i, p in enumerate(self.patterns):
            # Update surface ema
            p.ema_loss = (1 - self.ema_lambda) * p.ema_loss + self.ema_lambda * surface_loss[i]
            
            # Update structural ema (hierarchical)
            if structural_loss is not None and p.level > 1:
                p.ema_hier_loss = (1 - self.ema_lambda) * p.ema_hier_loss + self.ema_lambda * structural_loss[i]
            
            # Update compression ema
            if compression is not None and p.level > 1:
                p.ema_compression = (1 - self.ema_lambda) * p.ema_compression + self.ema_lambda * compression[i]
            
        # 2. Update Conflict Matrix (Discovery)
        n = len(self.patterns)
        new_kappa = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                k_ij = self.rule.detect_conflict(self.patterns[i], self.patterns[j])
                new_kappa[i, j] = k_ij
                new_kappa[j, i] = k_ij
        self.kappa_matrix = new_kappa
            
        # 3. Update Weights
        self.rule.update_weights(self.patterns, self.kappa_matrix)
        
        # 4. Substrate Management
        if social_freq is None:
            social_freq = np.zeros(len(self.patterns))
            
        for i, p in enumerate(self.patterns):
            # C(h) placeholder
            connectivity = 1.0 if len(p.constituent_features) > 1 else 0.5
            self.rule.check_substrate_shift(p, connectivity, social_freq[i])

    def update_ema_loss(self, pattern_idx: int, instantaneous_loss: float) -> None:
        """
        Li(t) = (1 - lambda)Li(t-1) + lambda * ell_i(t)
        """
        p = self.patterns[pattern_idx]
        p.ema_loss = (1 - self.ema_lambda) * p.ema_loss + self.ema_lambda * instantaneous_loss
