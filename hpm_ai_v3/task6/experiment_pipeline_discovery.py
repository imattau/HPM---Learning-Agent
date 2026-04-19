"""
experiment_pipeline_discovery.py - Verify automatic discovery of tool pipelines.
"""

import sys, os
sys.path.append(os.path.abspath("hpm_ai_v3"))

import torch
import numpy as np
from collections import deque
from typing import Dict, List, Any

from hpm_ai_v3.tools.registry import ToolRegistry
from hpm_ai_v3.tools.base import ToolPattern
from hpm_ai_v3.tools.composite import CompositeToolPattern
from hpm_ai_v3.operators.pipeline_recombination import PipelineRecombinationOperator
from hpm_ai_v3.meta_tool_orchestrator import MetaToolOrchestrator
from hpm_ai_v3.population import PatternPopulation
from hpm_ai_v3.evaluators import EvaluatorManager
from hpm_ai_v3.compiler import SubstrateCompiler

# ----------------------------------------------------------------------
# Register Tools
# ----------------------------------------------------------------------
def add_one_fn(x):
    return x + 1

def multiply_by_two_fn(y):
    return y * 2

ToolRegistry.register("add_one", add_one_fn, ["x"], "y", cost=0.01)
ToolRegistry.register("multiply_by_two", multiply_by_two_fn, ["y"], "z", cost=0.01)


# ----------------------------------------------------------------------
# Pipeline Discovery Agent
# ----------------------------------------------------------------------
class PipelineDiscoveryAgent:
    def __init__(self, tool_names: List[str]):
        self.tool_names = tool_names
        self.tool_patterns = [ToolRegistry.create_pattern(name) for name in tool_names]
        self.population = PatternPopulation(
            self.tool_patterns,
            eta=0.1,
            beta_c=0.0, # Disable inhibition for demo
            decay_rate=0.0, # Disable decay for demo
            age_decay_rate=0.0,
            pruning_threshold=0.0
        )
        self.eval_mgr = EvaluatorManager()
        self.compiler = SubstrateCompiler(use_gp=False)
        self.meta = MetaToolOrchestrator(available_tools=tool_names, context_feature_dim=4)
        self.pipeline_recomb = PipelineRecombinationOperator(
            co_occurrence_threshold=0.1,
            min_weight=-10.0 # Ignore weight for discovery in this demo
        )
        self.recent_tools = deque(maxlen=2)
        
    def step(self, sample: Dict):
        # The agent picks a tool based on the current context
        # If context has 'y', maybe it picks 'multiply_by_two'
        features = torch.zeros(4)
        features[0] = sample.get("x", 0) / 10.0
        features[1] = 1.0 if "y" in sample else 0.0
        
        meta_out = self.meta.sample({"context_features": features})
        selected_tool_name = meta_out["selected_tool"]
        tool_idx = meta_out["tool_idx"]
        
        self.recent_tools.append(selected_tool_name)
        if len(self.recent_tools) == 2:
            self.pipeline_recomb.record_sequence(list(self.recent_tools))
            
        pat = next((p for p in self.population.patterns if getattr(p, 'tool_name', None) == selected_tool_name or p.id == selected_tool_name), self.population.patterns[0])
        
        try:
            out = pat.sample(sample)
            sample.update(out)
            success = True
        except Exception as e:
            success = False
            
        # Target: z = (x + 1) * 2
        target_z = (sample.get("x", 0) + 1) * 2
        
        reward = 0.0
        if "z" in sample:
            error = abs(sample["z"] - target_z)
            if isinstance(error, torch.Tensor): error = error.item()
            reward = 1.0 - error - pat.cost
        elif success:
            reward = 0.5 # More reward for success
        else:
            reward = -0.1 # Less penalty
        ...
        n_sessions = 1000

        self.meta.update_parameters({
            "context_features": features.unsqueeze(0),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "tool_idx_taken": tool_idx.unsqueeze(0) if tool_idx.dim() == 0 else tool_idx
        })
        
        # Update population
        obs = sample.copy()
        if "x" in obs:
            obs["input"] = torch.tensor([obs["x"]], dtype=torch.float32)
            if "z" not in obs:
                obs["z"] = torch.tensor([target_z], dtype=torch.float32)
            # Ensure all values are tensors for evaluators
            for k in ["x", "y", "z"]:
                if k in obs and not isinstance(obs[k], torch.Tensor):
                    obs[k] = torch.tensor([obs[k]], dtype=torch.float32)
            self.population.step(self.eval_mgr, obs, self.compiler)
        
        return selected_tool_name, reward

def main():
    print("=== HPM Pipeline Discovery Experiment ===\n")
    agent = PipelineDiscoveryAgent(tool_names=["add_one", "multiply_by_two"])
    
    n_sessions = 500
    print("Phase 1: Exploration in Sessions...")
    for sess in range(n_sessions):
        context = {"x": np.random.uniform(0, 10)}
        # Each session allows 2 steps to try chaining
        for t in range(2):
            tool, reward = agent.step(context)
        
        # Force weights for discovery demo
        for p in agent.population.patterns:
            if isinstance(p, ToolPattern):
                p.weight = max(p.weight, 0.1)
        agent.population._update_kappa_matrix()
            
        if sess > 0 and sess % 50 == 0:
            new_comp = agent.pipeline_recomb.should_recombine(agent.population)
            if new_comp:
                print(f"\n[Session {sess}] Discovery! Found frequent pipeline: {' -> '.join(p.tool_name for p in new_comp.patterns)}")
                agent.population.patterns.append(new_comp)
                agent.population._update_kappa_matrix()
                agent.meta.available_tools.append(new_comp.id)
                agent.meta.num_tools += 1
                
                old_weight = agent.meta.policy_net[-1].weight.data
                old_bias = agent.meta.policy_net[-1].bias.data
                new_layer = torch.nn.Linear(agent.meta.hidden_dim, agent.meta.num_tools)
                new_layer.weight.data[:old_weight.shape[0]] = old_weight
                new_layer.bias.data[:old_bias.shape[0]] = old_bias
                agent.meta.policy_net[-1] = new_layer
                agent.meta.torch_optimizer = torch.optim.Adam(agent.meta.parameters(), lr=0.001)
                # Clear stats to avoid duplicate creation
                agent.pipeline_recomb.pair_counts.clear()
                
        if sess % 100 == 0:
            print(f"  Session {sess}: last tool={tool}, reward={reward:.3f}")

    print("\nFinal Population Weights:")
    for p in agent.population.patterns:
        name = getattr(p, 'tool_name', p.id)
        print(f"  {name}: {p.weight:.4f}")

if __name__ == "__main__":
    main()
