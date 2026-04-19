import numpy as np
import time
import torch
import networkx as nx
from typing import List, Tuple, Optional, Dict
from pattern import HPMPattern
from causal_pattern import CausalPattern
from symbolic_pattern import SymbolicPattern
from motor_pattern import MotorPattern
from evaluators import EvaluatorManager
from compiler import SubstrateCompiler
from device_utils import get_device

class PatternPopulation:
    def __init__(self, 
                 initial_patterns: List[HPMPattern],
                 eta: float = 0.1,
                 beta_c: float = 0.05,
                 recombination_prob: float = 0.01,
                 lambda_s: float = 0.3,
                 decay_rate: float = 0.005,
                 interference_strength: float = 0.02,
                 age_decay_rate: float = 0.01):
        self.patterns = initial_patterns
        self.eta = eta
        self.beta_c = beta_c
        self.recombination_prob = recombination_prob
        self.lambda_s = lambda_s
        self.decay_rate = decay_rate
        self.interference_strength = interference_strength
        self.age_decay_rate = age_decay_rate
        self._device = get_device(verbose=False)
        
        self._update_kappa_matrix()
        
    def _update_kappa_matrix(self):
        n = len(self.patterns)
        if n == 0: return
        self.kappa = np.ones((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = self.patterns[i].structural_distance(self.patterns[j])
                self.kappa[i, j] = d
                self.kappa[j, i] = d
                    
    def step(self, 
             evaluator_mgr: EvaluatorManager,
             observations: Dict,
             compiler: SubstrateCompiler,
             pattern_field_signal: Optional[List[float]] = None):
        # 1. Update evaluators
        for i, p in enumerate(self.patterns):
            evaluator_mgr.update_epistemic(p, observations)
            evaluator_mgr.update_invariance(p, observations)
            evaluator_mgr.update_coherence(p, self.patterns, observations)
            
            if hasattr(p, 'compression_score'):
                comp = p.compression_score(observations)
                evaluator_mgr.update_curiosity(p, comp)
            
            if pattern_field_signal is not None:
                evaluator_mgr.update_social(p, pattern_field_signal[i])
            
            p.update_parameters(observations)
            
        # Update density and stickiness
        for i, p in enumerate(self.patterns):
            evaluator_mgr.update_structural_connectivity(p)
            evaluator_mgr.update_evaluator_reinforcement(p)
            p.field_amplification = pattern_field_signal[i] if pattern_field_signal else 0.0
            base_loss = p.loss_ema if p.loss_ema is not None else 1.0
            p.compute_stickiness(base_loss)
            
        # 2. Check for substrate shifting
        for i, p in enumerate(self.patterns):
            if compiler.should_compile(p):
                sym_p = None
                if isinstance(p, CausalPattern):
                    sym_p = compiler.compile_to_symbolic(p)
                elif isinstance(p, MotorPattern):
                    sym_p = compiler.compile_motor_to_symbolic(p)
                if sym_p is not None:
                    sym_p.weight = p.weight
                    sym_p.accuracy = p.accuracy
                    self.patterns[i] = sym_p
                    
        # 3. Structural Recombination
        if np.random.random() < self.recombination_prob and len(self.patterns) >= 2:
            weights = np.array([p.weight for p in self.patterns])
            if weights.sum() > 0:
                parents = np.random.choice(self.patterns, size=2, replace=False, p=weights/weights.sum())
                new_pattern = self._recombine(parents[0], parents[1])
                if new_pattern is not None:
                    insight = evaluator_mgr.compute_insight(new_pattern, parents[0], parents[1])
                    new_pattern.insight_boost = insight
                    new_pattern.weight = 0.01
                    self.patterns.append(new_pattern)
                    self._update_kappa_matrix()
                
        # 4. Compute scores
        n = len(self.patterns) # re-compute if patterns added
        totals = np.array([p.total_score() for p in self.patterns])
        weights = np.array([p.weight for p in self.patterns])
        
        # Mark used patterns (top 3 by weight)
        if n > 0:
            top_k_idx = np.argsort(weights)[-min(3, n):]
            for idx in top_k_idx:
                if weights[idx] > 0.01:
                    self.patterns[idx].mark_used()
        
        avg_total = np.dot(weights, totals) / max(weights.sum(), 1e-6)
        stickiness_bonus = np.array([p.stickiness for p in self.patterns])
        avg_stickiness = np.dot(weights, stickiness_bonus) / max(weights.sum(), 1e-6)
        
        # 5. Replicator dynamics
        new_weights = np.zeros(n)
        current_time = time.time()
        
        for i in range(n):
            advantage = (totals[i] - avg_total) + self.lambda_s * (stickiness_bonus[i] - avg_stickiness)
            growth = self.eta * advantage * weights[i]
            
            inhibition = 0
            for j in range(n):
                if i != j:
                    inhibition += self.kappa[i, j] * weights[i] * weights[j]
            inhibition *= self.beta_c
            
            decay = self.decay_rate * weights[i] * (1.0 - self.patterns[i].affective_score)
            
            interference = 0.0
            for j in range(n):
                if i != j:
                    similarity = 1.0 - self.kappa[i, j]
                    interference += similarity * weights[j]
            interference *= self.interference_strength * weights[i]
            
            idle_time = current_time - self.patterns[i].last_used
            age_decay = self.age_decay_rate * idle_time / 3600.0
            
            new_weights[i] = weights[i] + growth - inhibition - decay - interference - age_decay
            new_weights[i] = max(0.0, new_weights[i])
            
        total_w = new_weights.sum()
        if total_w > 0:
            for i, p in enumerate(self.patterns):
                p.weight = new_weights[i] / total_w
        else:
            for p in self.patterns:
                p.weight = 1.0 / n
                
        self.patterns = [p for p in self.patterns if p.weight > 1e-4]
        self._update_kappa_matrix()
        
    def get_top_patterns(self, k: int = 3) -> List[HPMPattern]:
        sorted_pats = sorted(self.patterns, key=lambda p: p.weight, reverse=True)
        return sorted_pats[:k]
    
    def _recombine(self, p1: HPMPattern, p2: HPMPattern) -> Optional[HPMPattern]:
        if isinstance(p1, CausalPattern) and isinstance(p2, CausalPattern):
            new_p = CausalPattern(p1.input_dim, p1.z1_dim, p1.z2_dim)
            with torch.no_grad():
                for param1, param2, new_param in zip(p1.parameters(), p2.parameters(), new_p.parameters()):
                    new_param.data = 0.5 * (param1.data + param2.data)
            new_p.causal_graph = nx.compose(p1.causal_graph, p2.causal_graph)
            return new_p
        elif isinstance(p1, SymbolicPattern) and isinstance(p2, SymbolicPattern):
            def combined_fn(ctx):
                out1 = p1.forward_fn(ctx)
                out2 = p2.forward_fn(ctx)
                if isinstance(out1, torch.Tensor):
                    return (out1 + out2) / 2.0
                return (torch.tensor(out1) + torch.tensor(out2)) / 2.0
            return SymbolicPattern(combined_fn, p1.required_observation_keys)
        return None
