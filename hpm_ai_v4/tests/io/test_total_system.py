import numpy as np
import pytest
from hpm_ai_v4.system import TotalHPMSystem
from hpm_ai_v4.io.adapters import TextAdapter, MotorAdapter
from hpm_ai_v4.io.inspection import PatternInspector
from hpm_ai_v4.tests.test_faithful_sim import TrueEnvironment

def test_total_system_end_to_end():
    """Verify the full cognitive loop from text input to motor action."""
    env = TrueEnvironment()
    text_in = TextAdapter()
    motor_out = MotorAdapter()
    
    # Initialize the total system
    system = TotalHPMSystem(text_in, motor_out, env, num_agents=2)
    
    print("\nRunning Total HPM System loop...")
    # Stream of raw text inputs
    raw_inputs = ["Hello world", "HPM learning", "Hierarchy rules"]
    
    for raw in raw_inputs:
        system.step(raw)
        
    system.get_summary()
    
    # Verify learning happened
    all_patterns = [p for a in system.meta_layer.agent_pool.agents for p in a.patterns]
    assert len(all_patterns) > 0
    
    # Verify inspection
    inspector = PatternInspector(all_patterns[0])
    entropies = inspector.transition_entropy()
    assert isinstance(entropies, dict)
    print(f"Sample Pattern Entropies: {entropies}")
    
    print("\nTotal System Verification Successful!")

if __name__ == "__main__":
    test_total_system_end_to_end()
