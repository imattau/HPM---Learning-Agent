import numpy as np
import copy
from typing import List, Tuple
from hpm_ai_v4.pattern import HierarchicalPattern

class PatternRepository:
    """
    Stores patterns that have achieved high density across agents.
    Allows patterns to be injected into agents (transfer learning).
    """
    def __init__(self, density_threshold: float = 0.5):
        # List of (pattern, global_frequency)
        self.stored_patterns: List[Tuple[HierarchicalPattern, int]] = []
        self.density_threshold = density_threshold

    def update(self, agent_pool: 'AgentPool'):
        """Harvest high-density patterns from all agents in the pool."""
        from hpm_ai_v4.evaluators.metrics import epistemic_score
        
        for agent in agent_pool.agents:
            for p in agent.patterns:
                # Density proxy: epistemic score + compression bonus
                ep = epistemic_score(p)
                comp = p.compression(agent.obs_buffer) if hasattr(p, 'compression') else 0
                density = ep + comp
                
                if density > self.density_threshold:
                    sim_idx = self._find_similar_idx(p)
                    if sim_idx == -1:
                        # New unique high-density pattern
                        self.stored_patterns.append((copy.deepcopy(p), 1))
                    else:
                        # Increment frequency of existing pattern
                        stored, freq = self.stored_patterns[sim_idx]
                        self.stored_patterns[sim_idx] = (stored, freq + 1)

    def _find_similar_idx(self, p: HierarchicalPattern, threshold: float = 0.8) -> int:
        for i, (stored, _) in enumerate(self.stored_patterns):
            if self._structural_similarity(stored, p) > threshold:
                return i
        return -1

    def _structural_similarity(self, p: HierarchicalPattern, q: HierarchicalPattern) -> float:
        """Compute cosine similarity of flattened parameter vectors."""
        if p.complexity != q.complexity:
            return 0.0
            
        def get_params(pat):
            if pat.complexity >= 2:
                return np.concatenate([
                    pat.A3.flatten(), pat.A32.flatten(), 
                    pat.A21.flatten(), pat.B.flatten()
                ])
            else:
                return np.array([getattr(pat, 'theta', 0.5)])
                
        p_i = get_params(p)
        p_j = get_params(q)
        
        if len(p_i) != len(p_j): return 0.0
        
        return np.dot(p_i, p_j) / (np.linalg.norm(p_i)*np.linalg.norm(p_j) + 1e-12)

    def inject_best_patterns(self, agent: 'HPMAgent', num_patterns: int = 1):
        """Inject the most frequent/proven patterns into an agent's population."""
        if not self.stored_patterns:
            return
            
        # Sort by frequency (provenance)
        sorted_repo = sorted(self.stored_patterns, key=lambda x: x[1], reverse=True)
        
        current_ids = [p.id for p in agent.patterns]
        max_id = max(current_ids) if current_ids else 0
        
        for i in range(min(num_patterns, len(sorted_repo))):
            new_p = copy.deepcopy(sorted_repo[i][0])
            new_p.id = max_id + i + 1
            new_p.weight = 0.05 # Start with a modest experimental weight
            agent.patterns.append(new_p)
