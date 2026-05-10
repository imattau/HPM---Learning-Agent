import numpy as np
import torch
from typing import List, Dict, Any, Optional
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.evaluators.epistemic import EpistemicEvaluator

class ReplicationInstitution:
    """
    Implements a Replication Institution (Section 2.5.4).
    Periodically filters the pattern population by evaluating them on 
    a 'gold standard' validation sequence and pruning low performers.
    """
    def __init__(self, 
                 prune_ratio: float = 0.2, 
                 validation_seq: Optional[List[Cell]] = None):
        self.prune_ratio = prune_ratio
        self.validation_seq = validation_seq
        self.epistemic = EpistemicEvaluator()

    def filter_population(self, patterns: List[Cell], weights: np.ndarray, 
                          population: List[Cell], 
                          custom_validation_seq: Optional[List[Cell]] = None) -> np.ndarray:
        """
        Evaluates patterns and returns a new weight vector where poorly 
        performing patterns are zeroed out or heavily penalized.
        """
        val_seq = custom_validation_seq or self.validation_seq
        if not val_seq:
            return weights # No filtering without validation data
            
        losses = []
        for pat in patterns:
            # We use epistemic evaluation on the validation sequence
            context = {"observation_seq": val_seq, "population": population}
            res = self.epistemic.evaluate(pat, context)
            # Epistemic score is -NLL, so we want to minimize NLL (maximize score)
            losses.append(-res.score) # loss = NLL
            
        losses = np.array(losses)
        
        # Determine pruning threshold
        threshold = np.percentile(losses, (1.0 - self.prune_ratio) * 100)
        
        new_weights = weights.copy()
        for i, loss in enumerate(losses):
            if loss > threshold:
                new_weights[i] = 0.0 # Prune
                
        # Renormalize
        total = np.sum(new_weights)
        if total > 0:
            new_weights /= total
        else:
            # If all were pruned (edge case), return original
            return weights
            
        return new_weights

    def filter_population_tensor(self, patterns: List[Cell], weights: torch.Tensor,
                          population: List[Cell],
                          custom_validation_seq: Optional[List[Cell]] = None) -> torch.Tensor:
        filtered = self.filter_population(
            patterns,
            weights.detach().cpu().numpy(),
            population,
            custom_validation_seq=custom_validation_seq,
        )
        return torch.as_tensor(filtered, dtype=torch.float32)

class NormativeInstitution:
    """
    Filters patterns based on alignment with institutional norms 
    (e.g., minimum social consensus threshold).
    """
    def __init__(self, min_consensus: float = 0.3):
        self.min_consensus = min_consensus

    def filter_population(self, patterns: List[Cell], weights: np.ndarray, 
                          consensus_vec: np.ndarray) -> np.ndarray:
        new_weights = weights.copy()
        for i, p in enumerate(patterns):
            if p.similarity(consensus_vec) < self.min_consensus:
                new_weights[i] = 0.0
                
        total = np.sum(new_weights)
        if total > 0:
            new_weights /= total
        return new_weights

    def filter_population_tensor(self, patterns: List[Cell], weights: torch.Tensor,
                          consensus_vec: torch.Tensor) -> torch.Tensor:
        new_weights = weights.detach().clone().to(dtype=torch.float32)
        for i, p in enumerate(patterns):
            if float(p.similarity_tensor(consensus_vec)) < self.min_consensus:
                new_weights[i] = 0.0

        total = torch.sum(new_weights)
        if float(total) > 0.0:
            new_weights = new_weights / total
        return new_weights
