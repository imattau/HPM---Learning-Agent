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
        if isinstance(pattern, CausalPattern):
            total_norm = sum(torch.norm(p.data).item() for p in pattern.parameters())
            count = sum(p.data.numel() for p in pattern.parameters())
            pattern.structural_connectivity = min(1.0, total_norm / max(1, count * 10))
        elif isinstance(pattern, SymbolicPattern):
            try:
                import inspect
                src = inspect.getsource(pattern.forward_fn)
                ops = ['+', '-', '*', '/', '**', 'sin', 'cos', 'exp', 'log', 'if', 'for']
                count = sum(src.count(op) for op in ops)
                pattern.structural_connectivity = min(1.0, count / 50.0)
            except:
                pattern.structural_connectivity = 0.5
        else:
            pattern.structural_connectivity = 0.5
            
    def update_evaluator_reinforcement(self, pattern: HPMPattern):
        pattern.evaluator_reinforcement = (
            pattern.affective_score + 
            pattern.curiosity_reward + 
            pattern.coherence_score + 
            pattern.social_score
        ) / 4.0
        
    def update_epistemic(self, pattern: HPMPattern, observations: Dict[str, torch.Tensor]):
        obs = {k: (v.to(pattern._device) if isinstance(v, torch.Tensor) else v) for k, v in observations.items()}
        with torch.no_grad():
            logp = pattern.log_prob(obs)
            instant_loss = -logp.item()
        if pattern.loss_ema is None:
            pattern.loss_ema = instant_loss
        else:
            pattern.loss_ema = (1 - self.lambda_L) * pattern.loss_ema + self.lambda_L * instant_loss
        pattern.accuracy = -pattern.loss_ema
        
    def update_curiosity(self, pattern: HPMPattern, compression: float):
        if not hasattr(pattern, '_prev_compression'):
            pattern._prev_compression = compression
            progress = 0.0
        else:
            progress = max(0.0, compression - pattern._prev_compression)
            pattern._prev_compression = compression
        pattern.curiosity_reward = 0.9 * pattern.curiosity_reward + 0.1 * progress
        
    def update_coherence(self, pattern: HPMPattern, population: List[HPMPattern], observations: Dict[str, torch.Tensor] = None):
        if not population or len(population) < 2:
            base_coherence = 1.0
        else:
            distances = [pattern.structural_distance(other) for other in population if other != pattern]
            base_coherence = 1.0 - np.mean(distances) if distances else 1.0
        
        if observations is not None and hasattr(pattern, 'surface_dependence'):
            x = observations.get("input")
            if x is not None:
                surface_dep = pattern.surface_dependence(x)
                pattern.coherence_score = base_coherence * (1.0 - 0.5 * surface_dep)
            else:
                pattern.coherence_score = base_coherence
        else:
            pattern.coherence_score = base_coherence
            
    def update_affective(self, pattern: HPMPattern, reward: float):
        pattern.affective_score = 0.9 * pattern.affective_score + 0.1 * reward
        
    def update_social(self, pattern: HPMPattern, social_signal: float):
        pattern.social_score = 0.9 * pattern.social_score + 0.1 * social_signal
        
    def update_invariance(self, pattern: HPMPattern, observations: Dict[str, torch.Tensor]):
        # Find a suitable input key for this pattern to perturb
        input_key = "input"
        if input_key not in observations or (hasattr(pattern, 'required_observation_keys') and input_key not in pattern.required_observation_keys):
            for k in getattr(pattern, 'required_observation_keys', []):
                if k in observations:
                    input_key = k
                    break

        x = observations.get(input_key)
        if x is None or not isinstance(x, torch.Tensor): return

        x = x.to(pattern._device)
        if x.dim() == 1: x = x.unsqueeze(0)
        x_pert = x.clone()
        if x.shape[1] > 6:
            perm = torch.randperm(x.shape[0], device=pattern._device)
            x_pert[:, 2:6] = x[perm][:, 2:6]
        else:
            # Small random noise for low-dim inputs
            x_pert = x + torch.randn_like(x) * 0.1

        # Build full context for patterns that need multiple inputs
        context_orig = {k: observations[k] for k in getattr(pattern, 'required_observation_keys', []) if k in observations}
        context_pert = context_orig.copy()
        context_orig[input_key] = x
        context_pert[input_key] = x_pert

        with torch.no_grad():
            out_orig = pattern.sample(context_orig)
            out_pert = pattern.sample(context_pert)

            
            # Use probabilities if available (more stable for classification)
            if "probs" in out_orig:
                p_orig = out_orig["probs"]
                p_pert = out_pert["probs"]
                dist_val = ((p_orig - p_pert) ** 2).mean().item()
            elif "y" in out_orig:
                pred_orig = out_orig["y"]
                pred_pert = out_pert["y"]
                if isinstance(pred_orig, torch.Tensor) and isinstance(pred_pert, torch.Tensor):
                    dist_val = ((pred_orig.float() - pred_pert.float()) ** 2).mean().item()
                else:
                    dist_val = 0.0
            else:
                dist_val = 0.0
        
        pattern.invariance_score = 0.9 * pattern.invariance_score + 0.1 * np.exp(-dist_val)
        
    def compute_insight(self, new_pattern: HPMPattern, parent_a: HPMPattern, parent_b: HPMPattern) -> float:
        nov_a = new_pattern.structural_distance(parent_a)
        nov_b = new_pattern.structural_distance(parent_b)
        novelty = (nov_a + nov_b) / 2.0
        effectiveness = max(0.0, new_pattern.accuracy) if new_pattern.accuracy > -1 else 0.0
        return novelty * 0.6 + effectiveness * 0.4
