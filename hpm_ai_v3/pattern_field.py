import os
import numpy as np
import torch
import multiprocessing as mp
from typing import List, Dict, Any, Optional
from .population import PatternPopulation
from .pattern import HPMPattern
from .evaluators import EvaluatorManager
from compiler import SubstrateCompiler
from .device_utils import get_parallel_context
from collections import deque

class HPMAgent:
    """Local HPM agent, Ray-free."""
    def __init__(self, agent_id: str, initial_patterns: List[HPMPattern], buffer_size: int = 100):
        self.agent_id = agent_id
        self.population = PatternPopulation(initial_patterns)
        self.evaluator_mgr = EvaluatorManager()
        self.compiler = SubstrateCompiler()
        self.buffer = deque(maxlen=buffer_size)
        
    def step(self, observations: Dict, social_signals: Optional[Dict[str, float]] = None):
        self.buffer.append(observations)
        signal_list = [social_signals.get(p.id, 0.0) for p in self.population.patterns] if social_signals else None
        self.population.step(self.evaluator_mgr, observations, self.compiler, signal_list)
        
    def get_top_patterns(self, k: int = 3) -> List[HPMPattern]:
        return self.population.get_top_patterns(k)
    
    def test_patterns(self, patterns: List[HPMPattern]) -> Dict[str, float]:
        if not self.buffer: return {p.id: 0.0 for p in patterns}
        scores = {}
        for pat in patterns:
            losses = []
            for obs in self.buffer:
                obs_filt = pat.filter_observations(obs)
                if obs_filt:
                    with torch.no_grad():
                        losses.append(-pat.log_prob(obs_filt).item())
            scores[pat.id] = -np.mean(losses) if losses else -10.0
        return scores
    
    def apply_weight_penalty(self, pattern_id: str, penalty: float):
        for p in self.population.patterns:
            if p.id == pattern_id:
                p.weight = max(1e-6, p.weight - penalty)
                total = sum(pat.weight for pat in self.population.patterns)
                for pat in self.population.patterns: pat.weight /= total
                break
    
    def receive_social_signal(self, pattern_id: str, signal: float):
        for p in self.population.patterns:
            if p.id == pattern_id:
                self.evaluator_mgr.update_social(p, signal)
                break

class PatternField:
    """Institutional layer managing local multi-agent replication."""
    def __init__(self, num_agents=5, pattern_factory=None, replication_threshold=0.3,
                 field_amplification_bias=0.2, social_signal_magnitude=1.5,
                 replication_frequency=10, consensus_required=0.5, penalty_persistence=0.8):
        self.agents = [HPMAgent(f"agent_{i}", [pattern_factory() for _ in range(3)]) for i in range(num_agents)]
        self.replication_threshold = replication_threshold
        self.field_amplification_bias = field_amplification_bias
        self.social_signal_magnitude = social_signal_magnitude
        self.replication_frequency = replication_frequency
        self.consensus_required = consensus_required
        self.penalty_persistence = penalty_persistence
        self.global_pattern_frequency = {}
        self.penalty_memory = {i: {} for i in range(num_agents)}
        self._step_counter = 0
        
    def step_field(self, observations_batch: List[Dict]):
        self._step_counter += 1
        for i, agent in enumerate(self.agents):
            agent.step(observations_batch[i % len(observations_batch)])
        
        if self._step_counter % self.replication_frequency != 0: return
        
        # Collect top patterns
        all_published = []
        for agent_idx, agent in enumerate(self.agents):
            for pat in agent.get_top_patterns(k=1):
                all_published.append((agent_idx, pat))
                self.global_pattern_frequency[pat.id] = self.global_pattern_frequency.get(pat.id, 0) + 1
        
        # Replication tests
        for pub_agent_idx, pattern in all_published:
            scores = []
            for agent_idx, agent in enumerate(self.agents):
                if agent_idx == pub_agent_idx: continue
                scores.append(agent.test_patterns([pattern])[pattern.id])
                
            succ = sum(1 for s in scores if s > self.replication_threshold)
            ratio = succ / len(scores) if scores else 0
            
            base_sig = self.social_signal_magnitude if ratio >= self.consensus_required else -self.social_signal_magnitude
            sig = base_sig + self.field_amplification_bias * np.log1p(self.global_pattern_frequency.get(pattern.id, 0))
            
            if sig < 0:
                self.penalty_memory[pub_agent_idx][pattern.id] = self.penalty_memory[pub_agent_idx].get(pattern.id, 0) * self.penalty_persistence + abs(sig)
                self.agents[pub_agent_idx].apply_weight_penalty(pattern.id, self.penalty_memory[pub_agent_idx][pattern.id] * 0.05)
            else:
                self.penalty_memory[pub_agent_idx].pop(pattern.id, None)
                self.agents[pub_agent_idx].receive_social_signal(pattern.id, sig)
            
            if sig > 0:
                for i in range(len(self.agents)):
                    if i != pub_agent_idx: self.agents[i].receive_social_signal(pattern.id, sig * 0.5)

    def get_field_convergence(self) -> float:
        all_p = [p for agent in self.agents for p in agent.get_top_patterns(k=1)]
        if len(all_p) < 2: return 1.0
        dists = [all_p[i].structural_distance(all_p[j]) for i in range(len(all_p)) for j in range(i+1, len(all_p))]
        return 1.0 - np.mean(dists)

class ParallelPatternField(PatternField):
    """PatternField with multiprocessing support for CPU."""
    def __init__(self, num_agents=5, pattern_factory=None, n_workers=None, **kwargs):
        super().__init__(num_agents, pattern_factory, **kwargs)
        self.n_workers = n_workers or min(num_agents, os.cpu_count())
        self._pool = None

    def step_field(self, observations_batch: List[Dict]):
        self._step_counter += 1
        
        if self._pool is None:
            ctx = get_parallel_context()
            self._pool = ctx.Pool(self.n_workers)
        
        # Parallel agent steps
        step_args = [(agent, obs) for agent, obs in zip(self.agents, observations_batch)]
        self.agents = self._pool.starmap(_agent_step_wrapper, step_args)
        
        if self._step_counter % self.replication_frequency != 0: return
        
        # Collect top patterns
        all_published = []
        for agent_idx, agent in enumerate(self.agents):
            for pat in agent.get_top_patterns(k=1):
                all_published.append((agent_idx, pat))
                self.global_pattern_frequency[pat.id] = self.global_pattern_frequency.get(pat.id, 0) + 1
        
        # Replication tests (kept sequential for simplicity, but could be parallelized)
        for pub_agent_idx, pattern in all_published:
            scores = []
            for agent_idx, agent in enumerate(self.agents):
                if agent_idx == pub_agent_idx: continue
                scores.append(agent.test_patterns([pattern])[pattern.id])
                
            succ = sum(1 for s in scores if s > self.replication_threshold)
            ratio = succ / len(scores) if scores else 0
            
            base_sig = self.social_signal_magnitude if ratio >= self.consensus_required else -self.social_signal_magnitude
            sig = base_sig + self.field_amplification_bias * np.log1p(self.global_pattern_frequency.get(pattern.id, 0))
            
            if sig < 0:
                self.penalty_memory[pub_agent_idx][pattern.id] = self.penalty_memory[pub_agent_idx].get(pattern.id, 0) * self.penalty_persistence + abs(sig)
                self.agents[pub_agent_idx].apply_weight_penalty(pattern.id, self.penalty_memory[pub_agent_idx][pattern.id] * 0.05)
            else:
                self.penalty_memory[pub_agent_idx].pop(pattern.id, None)
                self.agents[pub_agent_idx].receive_social_signal(pattern.id, sig)
            
            if sig > 0:
                for i in range(len(self.agents)):
                    if i != pub_agent_idx: self.agents[i].receive_social_signal(pattern.id, sig * 0.5)

    def __del__(self):
        if hasattr(self, '_pool') and self._pool:
            self._pool.close()
            self._pool.terminate()

def _agent_step_wrapper(agent, obs):
    agent.step(obs)
    return agent
