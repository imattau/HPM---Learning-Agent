import numpy as np
from typing import Any

class CurriculumScheduler:
    """
    Adapts environment complexity based on agent population development.
    Implements developmental trajectory as defined in Section 7.4.
    """
    def __init__(self, env: Any):
        self.env = env
        self.stage = 0   # 0: surface, 1: local, 2: relational, 3: abstract, 4: generative

    def update(self, agent_pool: 'AgentPool'):
        """Monitor average population complexity and advance environment stage."""
        all_patterns = []
        for agent in agent_pool.agents:
            all_patterns.extend(agent.patterns)
            
        if not all_patterns:
            return
            
        # Compute weighted average complexity across all agents
        avg_complexity = np.mean([p.complexity for p in all_patterns])
        
        # Advance stage when agents prove mastery of current level
        if avg_complexity > 2.0 and self.stage < 1:
            self.stage = 1
            self._increase_complexity()
        elif avg_complexity > 2.5 and self.stage < 2:
            self.stage = 2
            self._increase_complexity()
        elif avg_complexity > 3.0 and self.stage < 3:
            self.stage = 3
            self._increase_complexity()
        elif avg_complexity > 3.5 and self.stage < 4:
            self.stage = 4
            self._increase_complexity()

    def _increase_complexity(self):
        """Increase the structure or difficulty of the environmental data source."""
        if hasattr(self.env, 'increase_complexity'):
            self.env.increase_complexity()
        elif hasattr(self.env, 'B_true'):
            # Fallback: make the emission matrix more complex or noisy
            # to challenge the agent's discrimination/categorization
            self.env.B_true = self.env.B_true * 0.9 + np.ones_like(self.env.B_true) * 0.05
            print(f"Curriculum: Environment complexity increased to stage {self.stage}")
