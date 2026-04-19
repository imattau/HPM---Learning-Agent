import torch
import numpy as np
from typing import List, Dict, Any
from pattern import HPMPattern
from causal_pattern import CausalPattern
from symbolic_pattern import SymbolicPattern

class EvaluatorManager:
    def __init__(self, lambda_L: float = 0.1):
        self.lambda_L = lambda_L
        
    def update_structural_connectivity(self, pattern: HPMPattern):
        """
        Estimate internal connectivity of pattern.
        """
        if isinstance(pattern, CausalPattern):
            # Use average absolute weight as proxy for connectivity
            total_param_norm = 0.0
            count = 0
            for param in pattern.parameters():
                total_param_norm += torch.norm(param.data).item()
                count += param.data.numel()
            pattern.structural_connectivity = min(1.0, total_param_norm / max(1, count * 10))
        elif isinstance(pattern, SymbolicPattern):
            # Symbolic complexity
            try:
                import inspect
                src = inspect.getsource(pattern.forward_fn)
                ops = ['+', '-', '*', '/', '**', 'sin', 'cos', 'exp', 'log']
                count = sum(src.count(op) for op in ops)
                pattern.structural_connectivity = min(1.0, count / 50.0)
            except:
                pattern.structural_connectivity = 0.5
                
    def update_evaluator_reinforcement(self, pattern: HPMPattern):
        """
        Evaluator saturation: sum of all non-epistemic evaluator scores.
        """
        pattern.evaluator_reinforcement = (
            pattern.affective_score + 
            pattern.curiosity_reward + 
            pattern.coherence_score + 
            pattern.social_score
        ) / 4.0  # normalize
        
    def update_epistemic(self, pattern: HPMPattern, observations: Dict[str, torch.Tensor]):
        """Running loss and accuracy."""
        with torch.no_grad():
            logp = pattern.log_prob(observations)
            instant_loss = -logp.item()
        
        if pattern.loss_ema is None:
            pattern.loss_ema = instant_loss
        else:
            pattern.loss_ema = (1 - self.lambda_L) * pattern.loss_ema + self.lambda_L * instant_loss
        pattern.accuracy = -pattern.loss_ema
        
    def update_curiosity(self, pattern: HPMPattern, learning_progress: float):
        """Curiosity as learning progress (improvement in compression)."""
        pattern.curiosity_reward = 0.9 * pattern.curiosity_reward + 0.1 * learning_progress
        
    def update_coherence(self, pattern: HPMPattern, population: List[HPMPattern]):
        """Coherence: how well pattern's causal graph aligns with majority."""
        if not population:
            return
        distances = [pattern.structural_distance(other) for other in population if other != pattern]
        if distances:
            avg_dist = np.mean(distances)
            pattern.coherence_score = 1.0 - avg_dist  # High coherence = low distance
        else:
            pattern.coherence_score = 1.0
            
    def update_affective(self, pattern: HPMPattern, reward: float):
        """Direct affective signal."""
        pattern.affective_score = 0.9 * pattern.affective_score + 0.1 * reward
        
    def update_social(self, pattern: HPMPattern, social_signal: float):
        """Social feedback from pattern field."""
        pattern.social_score = 0.9 * pattern.social_score + 0.1 * social_signal
        
    def compute_insight(self, new_pattern: HPMPattern, parent_a: HPMPattern, parent_b: HPMPattern) -> float:
        """Insight boost for recombined pattern."""
        nov_a = new_pattern.structural_distance(parent_a)
        nov_b = new_pattern.structural_distance(parent_b)
        novelty = (nov_a + nov_b) / 2.0
        effectiveness = new_pattern.accuracy if new_pattern.accuracy > 0 else 0.0
        insight = novelty * 0.6 + effectiveness * 0.4
        return insight
