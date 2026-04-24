import numpy as np
from collections import defaultdict

class SocialNetwork:
    """
    Global pattern field over multiple agents: prevalent patterns gain collective weight.
    This creates the cultural convergence predicted in Section 9.5.
    """
    def __init__(self, influence_strength: float = 0.1):
        self.influence_strength = influence_strength
        self.global_frequencies = defaultdict(float)

    def update(self, agent_pool: 'AgentPool'):
        """Count pattern occurrences across the population and propagate field influence."""
        freq_counts = defaultdict(int)
        
        all_patterns = []
        for agent in agent_pool.agents:
            for p in agent.patterns:
                all_patterns.append(p)
                sig = self._pattern_signature(p)
                freq_counts[sig] += 1
                
        total = sum(freq_counts.values()) + 1e-12
        self.global_frequencies = {sig: count / total for sig, count in freq_counts.items()}
        
        # Propagate field influence back to individual pattern weights
        for agent in agent_pool.agents:
            for p in agent.patterns:
                sig = self._pattern_signature(p)
                field_boost = self.influence_strength * self.global_frequencies.get(sig, 0.0)
                # Influence pattern weights by their global prevalence
                p.weight *= (1 + field_boost)

    def _pattern_signature(self, p: 'HierarchicalPattern') -> str:
        """Create a structural signature for grouping similar patterns."""
        if p.complexity < 2:
            return "flat"
        # Discretize multiple parameters to create a robust structural hash
        # Use mean of diagonals to capture 'stability' vs 'alternating' dynamics
        key = (
            np.round(np.diag(p.A3).mean(), 1),
            np.round(np.diag(p.A32).mean(), 1),
            np.round(p.B.mean(axis=0)[0], 1), # average emission bias
            p.complexity
        )
        return str(key)
