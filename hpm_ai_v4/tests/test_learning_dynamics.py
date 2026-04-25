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
        env.step_count = 0 # Reset env for each pattern
        buffer = []
        for step in range(300):
            obs = env.step()
            buffer.append(obs)
            if len(buffer) >= 5:
                p.update_parameters_online(buffer)
                
    best_ll = max(p.log_likelihood(target_seq) for p in patterns)
    
    print(f"\nMulti-Pattern Convergence Test:")
    print(f"  Best Final LL: {best_ll:.2f}")
    
    # For a perfect model of [0, 1, 0, 1...], LL would be 0.
    # Random model LL would be ~ -0.69 * 8 = -5.5
    assert best_ll > -5.0, "Patterns should improve significantly from random initialization"

def test_adaptive_vs_static_performance():
    """Verify that an agent with within-pattern learning adapts faster than one without."""
    env = FixedEnvironment()
    agent_adaptive = HPMAgent(num_initial_patterns=3)
    
    agent_static = HPMAgent(num_initial_patterns=3)
    # Mock update_parameters_online to be static
    for p in agent_static.patterns:
        if hasattr(p, 'update_parameters_online'):
            p.update_parameters_online = lambda obs, window_size=0: None
        if hasattr(p, 'observe'):
            p.observe = lambda obs, learning_rate=0: None
        
    # Run both for 100 steps (more steps for better signal)
    for _ in range(100):
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
