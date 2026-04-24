import numpy as np
import pytest
import copy
from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.tools.substrate import ExternalSubstrate
from hpm_ai_v4.social import SocialNetwork
from hpm_ai_v4.meta import HPMMetaLayer
from hpm_ai_v4.pattern import HierarchicalPattern

def test_gossip_mechanism():
    """Verify that patterns can transfer between agents via the substrate."""
    substrate = ExternalSubstrate()
    agent_a = HPMAgent(num_initial_patterns=0, external_substrate=substrate)
    agent_b = HPMAgent(num_initial_patterns=0, external_substrate=substrate)
    
    # Give agent_a a high-weight unique pattern
    p_unique = HierarchicalPattern(pattern_id=99, latent_dim=2, obs_dim=2)
    p_unique.weight = 0.8
    agent_a.patterns.append(p_unique)
    
    # Ensure agent_a's other patterns (flat) are below threshold or removed for this test
    for p in agent_a.patterns:
        if p.id != 99: p.weight = 0.0
    
    # Broadcast from agent_a
    agent_a.external.broadcast(agent_a.patterns)
    assert 99 in substrate.storage
    
    # Gossip to agent_b - try a few times to ensure we hit the 99
    for _ in range(5):
        agent_b.gossip_with_substrate(substrate)
    
    # Verify agent_b now has pattern 99
    agent_b_ids = [p.id for p in agent_b.patterns]
    assert 99 in agent_b_ids
    
def test_social_convergence_signature():
    """Verify that SocialNetwork influence boosts patterns with shared signatures."""
    substrate = ExternalSubstrate()
    
    # Two agents in separate pools but same social network
    agent_a = HPMAgent(num_initial_patterns=0, external_substrate=substrate)
    agent_b = HPMAgent(num_initial_patterns=0, external_substrate=substrate)
    
    # Identical pattern in both
    p1 = HierarchicalPattern(pattern_id=1)
    p1.weight = 0.1
    p1.A3 = np.eye(2) # Stabilize signature
    
    p2 = HierarchicalPattern(pattern_id=2)
    p2.weight = 0.1
    p2.A3 = np.eye(2)
    
    agent_a.patterns.append(p1)
    agent_b.patterns.append(p2)
    
    social = SocialNetwork(influence_strength=1.0)
    
    class MockPool:
        def __init__(self, agents): self.agents = agents
    
    pool = MockPool([agent_a, agent_b])
    
    # Initial weights
    w1_init = p1.weight
    
    social.update(pool)
    
    # Weights should increase due to shared signature
    assert p1.weight > w1_init
    assert p2.weight > w1_init

def test_reflection_stagnation_intervention():
    """Verify that ReflectionEngine triggers meta-interventions on performance stagnation."""
    from hpm_ai_v4.meta import AgentPool
    from hpm_ai_v4.repository import PatternRepository
    from hpm_ai_v4.reflection import ReflectionEngine
    
    substrate = ExternalSubstrate()
    repo = PatternRepository()
    pool = AgentPool(num_agents=1, external_substrate=substrate)
    reflection = ReflectionEngine(pool, repo)
    
    agent = pool.agents[0]
    initial_beta = agent.beta_aff
    
    # Simulate stagnation by calling step multiple times
    # By default, patterns have running_loss = 0.0, so performance is stagnant
    for _ in range(50):
        reflection.step(0)
        
        if agent.beta_aff > initial_beta:
            break
            
    assert agent.beta_aff > initial_beta, "Curiosity (beta_aff) should increase on stagnation"

if __name__ == "__main__":
    pytest.main([__file__])
