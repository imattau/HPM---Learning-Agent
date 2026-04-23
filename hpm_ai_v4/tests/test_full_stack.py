import numpy as np
import pytest
from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.meta import AgentPool

def test_agent_reasoning_integration():
    """Verify that an agent can use its reasoner to act and plan."""
    agent = HPMAgent(num_initial_patterns=2)
    
    # 1. Give the agent some history
    for _ in range(20):
        agent.perceive_and_learn(1, None) # Obs 1 is common
        
    # 2. Test standard action selection (compositional inference)
    action = agent.act()
    assert action in [0, 1]
    
    # 3. Test planning towards a goal
    # We force a goal of 0 (different from history of 1s)
    plan_action = agent.act(goal=0)
    assert plan_action in [0, 1]
    
    # 4. Test explanation
    best_pattern = max(agent.patterns, key=lambda p: p.weight)
    explanation = agent.reasoner.explain(best_pattern)
    assert "ID" in explanation
    assert len(explanation) > 10

def test_meta_agent_pool_reflection():
    """Verify that the meta-layer can coordinate and reflect on multiple agents."""
    from hpm_ai_v4.meta import MetaReasoner
    pool = AgentPool(num_agents=2)
    meta_reasoner = MetaReasoner(pool)
    
    # 1. Provide a common data stream
    for _ in range(50):
        pool.step(0)
        pool.step(1)
        
    # 2. Check if shared substrate has patterns
    pool.substrate.broadcast(pool.agents[0].patterns, threshold=0.001)
    assert len(pool.substrate.storage) > 0
    
    # 3. Trigger reflection
    # Reflection detects issues in a specific agent
    issues = meta_reasoner.detect_inconsistencies(pool.agents[0])
    print(f"\nMeta Reflection Results for Agent 0: {issues}")
    assert isinstance(issues, list)

if __name__ == "__main__":
    test_agent_reasoning_integration()
    test_meta_agent_pool_reflection()
    print("Full HPM Stack Verification Complete!")
