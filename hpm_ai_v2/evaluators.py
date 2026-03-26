from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Any
from .patterns import CompositePattern, SubstrateID

class InstitutionalField:
    """
    A specialized pattern field (e.g., Science) that applies a Replication Filter.
    """
    def __init__(self, replication_threshold: float = 0.5):
        self.replication_threshold = replication_threshold
        self.variance_tracker: Dict[str, List[float]] = {}
        
    def apply_filter(self, p: CompositePattern, current_loss: float) -> float:
        """
        Calculates amplification for pattern p.
        Returns a multiplier for the social score / field amplification.
        If accuracy is high variance or inconsistent, the multiplier is low.
        """
        if p.id not in self.variance_tracker:
            self.variance_tracker[p.id] = []
            
        self.variance_tracker[p.id].append(current_loss)
        
        # Keep only recent history
        if len(self.variance_tracker[p.id]) > 10:
            self.variance_tracker[p.id].pop(0)
            
        if len(self.variance_tracker[p.id]) < 3:
            return 0.5 # Early discovery penalty
            
        # Calculate variance and mean of loss
        losses = np.array(self.variance_tracker[p.id])
        loss_variance = np.var(losses)
        mean_loss = np.mean(losses)
        
        # High variance (inconsistency) or high mean loss (low accuracy)
        # results in poor replication score
        replication_score = np.exp(-loss_variance) * np.exp(-mean_loss)
        
        return 1.0 if replication_score >= self.replication_threshold else 0.1
