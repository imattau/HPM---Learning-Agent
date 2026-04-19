"""
test_agent_spawning.py - Demonstrate autonomous agent creation from pipeline.
"""

import sys, os
sys.path.append(os.path.abspath("hpm_ai_v3"))

import torch
import numpy as np
from hpm_ai_v3.agents.registry import AgentRegistry
from hpm_ai_v3.agents.base import AgentPattern
from hpm_ai_v3.agents.composite import CompositeAgentPattern
from hpm_ai_v3.compiler.base_compiler import SubstrateCompiler
from hpm_ai_v3.augmented_agent import AugmentedHPMAgent

# ----------------------------------------------------------------------
# Mock Specialists
# ----------------------------------------------------------------------
class AddOneAgent:
    required_observation_keys = ["x"]
    output_key = "y"
    def invoke(self, ctx):
        x = ctx.get("x", 0)
        return {"y": x + 1}

class MultiplyTwoAgent:
    required_observation_keys = ["y"]
    output_key = "z"
    def invoke(self, ctx):
        y = ctx.get("y", 0)
        return {"z": y * 2}

def main():
    print("=== Autonomous Agent Creation Test ===\n")
    
    AgentRegistry.clear()
    AgentRegistry.register("add_one", AddOneAgent())
    AgentRegistry.register("multiply_by_two", MultiplyTwoAgent())
    
    # Create patterns
    p1 = AgentRegistry.create_pattern("add_one")
    p2 = AgentRegistry.create_pattern("multiply_by_two")
    
    # Create composite: add_one -> multiply_by_two
    composite = CompositeAgentPattern([p1, p2])
    composite.accuracy = 0.99
    composite.loss_ema = 0.01
    composite.weight = 0.8
    
    print(f"Composite agent sequence: {' -> '.join(composite.get_sequence())}")
    
    # Compile and spawn new agent
    compiler = SubstrateCompiler()
    new_agent = compiler.spawn_agent_from_composite(composite, "add_then_multiply")
    
    # Test the spawned agent
    test_context = {"x": 5.0}
    # AugmentedHPMAgent.process returns working context
    result = new_agent.invoke(test_context)
    print(f"\nSpawned agent '{new_agent.agent_id}' processing x=5: {result}")
    
    # Expected: (5+1)*2 = 12
    assert result["z"] == 12.0
    
    # Verify it's registered
    print(f"\nAvailable agents: {AgentRegistry.list_agents()}")
    
    # Test via AgentRegistry.call()
    output = AgentRegistry.call("add_then_multiply", x=10.0)
    print(f"Direct call with x=10: {output}")
    assert output["z"] == 22.0
    
    print("\nAutonomous agent creation test passed.")

if __name__ == "__main__":
    main()
