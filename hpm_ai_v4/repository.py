import numpy as np
import copy
from typing import List, Tuple
from hpm_ai_v4.pattern import HierarchicalPattern
from hpm_ai_v4.tools.pattern_equivalence import PatternEquivalenceIndex

class PatternRepository:
    """
    Stores patterns that have achieved high density across agents.
    Allows patterns to be injected into agents (transfer learning).
    """
    def __init__(self, density_threshold: float = 0.5, equivalence_index: PatternEquivalenceIndex | None = None):
        # List of (pattern, global_frequency)
        self.stored_patterns: List[Tuple[HierarchicalPattern, int]] = []
        self.density_threshold = density_threshold
        self.equivalence_index = equivalence_index or PatternEquivalenceIndex()

    def update(self, agent_pool: 'AgentPool'):
        """Harvest high-density patterns from all agents in the pool."""
        from hpm_ai_v4.evaluators.metrics import epistemic_score
        
        for agent in agent_pool.agents:
            for p in agent.patterns:
                # Density proxy: epistemic score + compression bonus
                ep = epistemic_score(p)
                if hasattr(p, 'compression'):
                    try:
                        comp = p.compression()
                    except TypeError:
                        # Fallback for old interface if any objects persist
                        comp = p.compression(agent.obs_buffer)
                else:
                    comp = 0
                density = ep + comp
                
                if density > self.density_threshold:
                    match = self.equivalence_index.match_pattern(
                        p,
                        [stored for stored, _ in self.stored_patterns],
                        threshold=0.8,
                    )
                    if match.matched_index == -1:
                        # New unique high-density pattern
                        self.stored_patterns.append((copy.deepcopy(p), 1))
                    else:
                        # Increment frequency of existing pattern
                        stored, freq = self.stored_patterns[match.matched_index]
                        self.stored_patterns[match.matched_index] = (stored, freq + 1)

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
