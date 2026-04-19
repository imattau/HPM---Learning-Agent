import numpy as np
import torch
from typing import List, Dict, Any, Optional
from population import PatternPopulation
from pattern import HPMPattern
from evaluators import EvaluatorManager
from compiler import SubstrateCompiler
from collections import deque

class HPMAgent:
    """Local HPM agent, replaces ray.remote HPMAgent."""
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
    
    def receive_social_signal(self, pattern_id: str, signal: float):
        for p in self.population.patterns:
            if p.id == pattern_id:
                self.evaluator_mgr.update_social(p, signal)
                break

class PatternField:
    """Institutional layer managing local multi-agent population."""
    def __init__(self, num_agents: int = 5, pattern_factory: callable = None, replication_threshold: float = 0.1):
        self.agents = [HPMAgent(f"agent_{i}", [pattern_factory() for _ in range(3)]) for i in range(num_agents)]
        self.replication_threshold = replication_threshold
        self.shared_ledger = {}
        self.global_pattern_frequency = {}
        
    def step_field(self, observations_batch: List[Dict]):
        # 1. Agents learn
        for i, agent in enumerate(self.agents):
            agent.step(observations_batch[i % len(observations_batch)])
        
        # 2. Collect patterns
        published = []
        for agent_idx, agent in enumerate(self.agents):
            for pat in agent.get_top_patterns(k=1):
                published.append((agent_idx, pat))
                self.global_pattern_frequency[pat.id] = self.global_pattern_frequency.get(pat.id, 0) + 1
        
        # 3. Replication tests
        unique_patterns = {pat.id: pat for _, pat in published}
        scores = {pat_id: [] for pat_id in unique_patterns}
        
        for agent in self.agents:
            results = agent.test_patterns(list(unique_patterns.values()))
            for pat_id, score in results.items():
                scores[pat_id].append(score)
        
        # 4. Feedback
        feedback = {i: {} for i in range(len(self.agents))}
        for pub_agent_idx, pat in published:
            avg_score = np.mean(scores[pat.id])
            signal = (1.0 if avg_score > self.replication_threshold else -0.5) + (0.2 * np.log1p(self.global_pattern_frequency.get(pat.id, 0)))
            feedback[pub_agent_idx][pat.id] = signal
            if signal > 0:
                for idx in range(len(self.agents)):
                    if idx != pub_agent_idx: feedback[idx][pat.id] = signal * 0.5
        
        for agent_idx, signals in feedback.items():
            for pat_id, sig in signals.items():
                self.agents[agent_idx].receive_social_signal(pat_id, sig)

    def get_field_convergence(self) -> float:
        all_pats = [p for agent in self.agents for p in agent.get_top_patterns(k=1)]
        if len(all_pats) < 2: return 1.0
        dists = [all_pats[i].structural_distance(all_pats[j]) for i in range(len(all_pats)) for j in range(i+1, len(all_pats))]
        return 1.0 - np.mean(dists)
