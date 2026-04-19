"""
test_agent_workflow_compilation.py - Verify multi-agent workflow compilation.
"""

import sys, os
sys.path.append(os.path.abspath("hpm_ai_v3"))

import torch
import numpy as np
from hpm_ai_v3.agents.registry import AgentRegistry
from hpm_ai_v3.agents.base import AgentPattern
from hpm_ai_v3.agents.composite import CompositeAgentPattern
from hpm_ai_v3.compiler.base_compiler import SubstrateCompiler

# ----------------------------------------------------------------------
# Mock Specialists
# ----------------------------------------------------------------------
class VisionSpecialist:
    required_observation_keys = ["image"]
    output_key = "label"
    def invoke(self, ctx):
        return {"label": "cat"}

class LanguageSpecialist:
    required_observation_keys = ["label"]
    output_key = "embedding"
    def invoke(self, ctx):
        return {"embedding": torch.randn(384)}

class MemorySpecialist:
    required_observation_keys = ["embedding"]
    output_key = "docs"
    def invoke(self, ctx):
        return {"docs": ["Doc1", "Doc2"]}

def main():
    print("=== Testing Agent Workflow Compilation ===\n")
    
    AgentRegistry.clear()
    AgentRegistry.register("vision_specialist", VisionSpecialist(), "Classifies images.")
    AgentRegistry.register("language_specialist", LanguageSpecialist(), "Embeds labels.")
    AgentRegistry.register("memory_specialist", MemorySpecialist(), "Retrieves documents.")
    
    # Create patterns
    p_vision = AgentRegistry.create_pattern("vision_specialist")
    p_lang = AgentRegistry.create_pattern("language_specialist")
    p_mem = AgentRegistry.create_pattern("memory_specialist")
    
    # Create composite: vision -> language -> memory
    composite = CompositeAgentPattern([p_vision, p_lang, p_mem])
    composite.accuracy = 0.95
    composite.loss_ema = 0.05
    composite.weight = 0.6
    
    print(f"Composite ID: {composite.id}")
    print(f"Required Inputs: {composite.required_observation_keys}")
    print(f"Output Key: {composite.output_key}")
    
    compiler = SubstrateCompiler()
    
    print("\nChecking should_compile...")
    assert compiler.should_compile(composite)
    
    print("\nCompiling Agent Workflow...")
    sym_pat = compiler.compile_to_symbolic(composite)
    
    assert sym_pat is not None
    print(f"Compiled Symbolic Pattern ID: {sym_pat.id}")
    print(f"Source Code:\n{sym_pat.source_code}")
    
    # Test execution
    print("\nTesting Compiled Execution...")
    context = {"image": torch.randn(3, 224, 224)}
    result = sym_pat.sample(context)
    
    print(f"Result Output: {result}")
    assert "docs" in result
    assert result["docs"] == ["Doc1", "Doc2"]
    
    print("\nAgent workflow compilation test passed.")

if __name__ == "__main__":
    main()
