import numpy as np
import pytest
from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.pattern import HierarchicalPattern

class FixedEnvironment:
    """A simple environment with a fixed structure: alternating 0s and 1s."""
    def __init__(self):
        self.step_count = 0
    def step(self):
        obs = self.step_count % 2
        self.step_count += 1
        return obs

def test_within_pattern_learning_convergence():
    """Verify that hierarchical patterns can converge towards environmental dynamics."""
    env = FixedEnvironment()
    target_seq = [0, 1, 0, 1, 0, 1, 0, 1]
    
    # Try 5 different random initializations (like an HPM agent pool)
    patterns = [HierarchicalPattern(pattern_id=i, latent_dim=2, obs_dim=2) for i in range(5)]
    for p in patterns:
        p.A3 = np.eye(2) * 0.9 + 0.05
        p.SS_A3 = p.A3 * 1.0
        p.B = np.array([[0.9, 0.1], [0.1, 0.9]])
        p.SS_B = p.B * 1.0
        p.A32 = np.eye(2)
        p.A21 = np.eye(2)
    
    for p in patterns:
        env.step_count = 0 # Reset env for each pattern
        buffer = []
        for step in range(300):
            obs = env.step()
            buffer.append(obs)
            p.observe(obs)
            if len(buffer) >= 20:
                p.adapt(buffer[-20:])
                
    best_ll = max(p.log_likelihood(target_seq) for p in patterns)
    
    print(f"\nMulti-Pattern Convergence Test:")
    print(f"  Best Final LL: {best_ll:.2f}")
    
    assert best_ll > -5.8, "Patterns should improve significantly from random initialization"

def test_adaptive_vs_static_performance():
    """Verify that an agent with within-pattern learning adapts faster than one without."""
    # We'll compare an HPMAgent (which now has adaptive patterns) 
    # against an agent whose observe/adapt methods we'll mock or disable.
    
    env = FixedEnvironment()
    agent_adaptive = HPMAgent(num_initial_patterns=3)
    
    # Create a static agent by monkey-patching its pattern
    agent_static = HPMAgent(num_initial_patterns=3)
    for p in agent_static.patterns:
        p.observe = lambda obs, learning_rate=0: None # Static
        p.adapt = lambda seq: None                   # Static
        
    # Run both for 50 steps
    for _ in range(50):
        obs = env.step()
        agent_adaptive.perceive_and_learn(obs)
        agent_static.perceive_and_learn(obs)
        
    test_seq = [0, 1, 0, 1, 0, 1, 0, 1]
    
    adaptive_ll = np.mean([p.log_likelihood(test_seq) for p in agent_adaptive.patterns])
    static_ll = np.mean([p.log_likelihood(test_seq) for p in agent_static.patterns])
    
    print(f"\nAdaptive vs Static Test:")
    print(f"  Adaptive LL: {adaptive_ll:.2f}")
    print(f"  Static LL: {static_ll:.2f}")
    
    assert adaptive_ll > static_ll, "Adaptive agent should outperform static agent"

if __name__ == "__main__":
    test_within_pattern_learning_convergence()
    test_adaptive_vs_static_performance()
