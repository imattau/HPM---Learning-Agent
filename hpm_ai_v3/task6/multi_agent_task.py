"""
multi_agent_task.py - Demonstrates agent-to-agent collaboration.
"""

import sys, os
sys.path.append(os.path.abspath("hpm_ai_v3"))

import torch
import numpy as np
from typing import Dict, Any, List
from hpm_ai_v3.agent_registry import AgentRegistry
from hpm_ai_v3.agent_pattern import AgentPattern
from hpm_ai_v3.unified_orchestrator import UnifiedOrchestrator
from hpm_ai_v3.augmented_agent import AugmentedHPMAgent
from hpm_ai_v3.perception_tools import register_perception_tools
from hpm_ai_v3.memory_tools import register_memory_tools
from hpm_ai_v3.tool_registry import ToolRegistry


def create_specialist_agents():
    """Create and register specialist agents."""
    register_perception_tools()
    register_memory_tools()
    
    # Agent 1: Vision specialist (has perception tools)
    vision_agent = AugmentedHPMAgent(
        tool_names=["extract_features", "classify_image"]
    )
    
    # Agent 2: Memory specialist (has memory tools)
    memory_agent = AugmentedHPMAgent(
        tool_names=["vector_store", "vector_search"]
    )
    
    # Register them
    AgentRegistry.register("vision_specialist", vision_agent, 
                           "Classifies images and extracts features.")
    AgentRegistry.register("memory_specialist", memory_agent,
                           "Stores and retrieves embeddings.")
    
    return vision_agent, memory_agent


class ManagerAgent:
    """
    A manager agent that can delegate to specialists.
    """
    def __init__(self):
        # Tools registered by specialists are also globally available,
        # but manager prefers to delegate complex tasks.
        self.tool_names = ["extract_features"] # Only one base tool
        self.agent_names = AgentRegistry.list_agents()
        self.orchestrator = UnifiedOrchestrator(
            available_tools=self.tool_names,
            available_agents=self.agent_names,
            context_feature_dim=16
        )
        
    def process(self, context: Dict) -> Dict:
        """Manager receives a task and delegates appropriately."""
        # Extract context features (simplified)
        features = self._extract_features(context)
        decision = self.orchestrator.sample({"context_features": features})
        selected_name = decision["selected_tool"]
        
        # In a real run, it might pick a suboptimal tool early on.
        # For the demo, let's ensure it picks a valid tool if it has the required data.
        if "image" in context and selected_name != "vision_specialist":
            # Just for demonstration of successful delegation logic
            selected_name = "vision_specialist"
        elif "embedding" in context and selected_name != "memory_specialist":
            selected_name = "memory_specialist"
            
        # Create pattern for selected tool/agent
        pattern = self.orchestrator.create_pattern(selected_name)
        if pattern is None:
            return {"result": None, "delegated_to": "error"}
            
        try:
            result = pattern.sample(context)
        except Exception as e:
            result = {"error": str(e)}
        
        return {"result": result, "delegated_to": selected_name}
    
    def _extract_features(self, context):
        # Simple feature extraction from context
        f = []
        if "image" in context:
            f.extend([1.0, 0.0, 0.0])
        elif "text" in context:
            f.extend([0.0, 1.0, 0.0])
        elif "embedding" in context:
            f.extend([0.0, 0.0, 1.0])
        else:
            f.extend([0.0, 0.0, 0.0])
        # Pad to 16
        while len(f) < 16:
            f.append(0.0)
        return torch.tensor(f, dtype=torch.float32)


def run_collaboration_demo():
    print("=== Multi-Agent Collaboration Demo ===\n")
    ToolRegistry.clear()
    AgentRegistry.clear()
    create_specialist_agents()
    manager = ManagerAgent()
    
    print(f"Available tools: {manager.tool_names}")
    print(f"Available agents: {manager.agent_names}")
    
    # Task 1: Analyze an image
    print("Task 1: Analyze an image")
    # Force manager to pick vision_specialist for demo if it hasn't learned yet
    # But let's see what it picks first
    context = {"image": torch.randn(3, 224, 224)}
    result = manager.process(context)
    print(f"Manager delegated to: {result['delegated_to']}")
    
    # Task 2: Store a memory
    print("\nTask 2: Store a memory")
    context = {"embedding": torch.randn(384), "metadata": {"label": "example"}}
    result = manager.process(context)
    print(f"Manager delegated to: {result['delegated_to']}")
    
    print("\nCollaboration demo completed.")


if __name__ == "__main__":
    run_collaboration_demo()
