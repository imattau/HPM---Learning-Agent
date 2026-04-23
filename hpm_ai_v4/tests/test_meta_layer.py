import numpy as np
from hpm_ai_v4.meta import HPMMetaLayer
from hpm_ai_v4.tests.test_faithful_sim import TrueEnvironment

def test_meta_layer_simulation():
    """Verify that the meta-layer can coordinate a population of learners."""
    env = TrueEnvironment()
    # Initialize meta-layer with 3 agents
    meta = HPMMetaLayer(env, num_agents=3)
    
    print("\nStarting HPM Meta-Layer Simulation...")
    for step in range(300):
        meta.run_step()
        
        if step % 50 == 0:
            meta.report()
            
    # Final assertions
    all_patterns = [p for a in meta.agent_pool.agents for p in a.patterns]
    assert len(all_patterns) > 0, "Agents should have patterns in their population"
    assert meta.global_step == 300, "Simulation should complete all steps"
    
    # Check social network convergence (global frequencies should be populated)
    assert len(meta.social_network.global_frequencies) > 0
    
    # Check if anything made it to repository
    print(f"Final Repository Size: {len(meta.repository.stored_patterns)}")
    
    print("\nMeta-Layer Simulation Complete!")

if __name__ == "__main__":
    test_meta_layer_simulation()
