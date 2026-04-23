import numpy as np
import random
from hpm_ai_v4.operators.dynamics import recombine

class ReflectionEngine:
    """
    Monitors progress and triggers meta-interventions:
    - Increase exploration on stagnation.
    - Perform cross-agent recombination.
    - Inject proven repository patterns.
    """
    def __init__(self, agent_pool, repository):
        self.agent_pool = agent_pool
        self.repository = repository
        self.performance_history = []
        self.stagnation_counter = 0

    def step(self, step_idx: int):
        """Analyze current pool performance and intervene if necessary."""
        all_patterns = []
        for agent in self.agent_pool.agents:
            all_patterns.extend(agent.patterns)
            
        if not all_patterns: return
        
        # Performance proxy: average epistemic score (negative loss)
        avg_epistemic = np.mean([-p.running_loss for p in all_patterns])
        self.performance_history.append(avg_epistemic)
        
        # Keep window for stagnation detection
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)
            
        # Detect stagnation
        if len(self.performance_history) >= 10:
            recent = self.performance_history[-10:]
            if (max(recent) - min(recent)) < 0.02: # No meaningful improvement
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                
            if self.stagnation_counter >= 3:
                self._intervene()

    def _intervene(self):
        """Execute escape-from-local-optimum strategies."""
        print(f"Reflection: Stagnation detected. Intervening across agent pool.")
        
        # 1. Broaden curiosity: Increase affective weights globally
        for agent in self.agent_pool.agents:
            agent.beta_aff = min(0.9, agent.beta_aff * 1.2)
            
        # 2. Cross-pollination: Recombine patterns from different agents
        if len(self.agent_pool.agents) >= 2:
            a1, a2 = random.sample(self.agent_pool.agents, 2)
            if a1.patterns and a2.patterns:
                p1 = random.choice(a1.patterns)
                p2 = random.choice(a2.patterns)
                child = recombine(p1, p2)
                if child:
                    # Inject child into both agents as a high-potential hypothesis
                    child.id = max([p.id for p in a1.patterns] + [0]) + 1
                    child.weight = 0.1
                    a1.patterns.append(child)
                    
        # 3. Injectproven patterns into the agent with the highest average loss
        worst_agent = min(self.agent_pool.agents, key=lambda a: np.mean([-p.running_loss for p in a.patterns] if a.patterns else [-1e9]))
        self.repository.inject_best_patterns(worst_agent, num_patterns=1)
        
        self.stagnation_counter = 0
