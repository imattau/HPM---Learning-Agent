import sys, os
sys.path.append(os.path.abspath("hpm_ai_v3"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from sklearn.tree import DecisionTreeClassifier

from hpm_ai_v3.tool_registry import ToolRegistry
from hpm_ai_v3.tool_pattern import ToolPattern
from hpm_ai_v3.meta_tool_orchestrator import MetaToolOrchestrator
from hpm_ai_v3.symbolic_pattern import SymbolicPattern
from hpm_ai_v3.population import PatternPopulation
from hpm_ai_v3.evaluators import EvaluatorManager
from hpm_ai_v3.compiler import SubstrateCompiler
from hpm_ai_v3.neural_pattern import RegressionPattern

# ----------------------------------------------------------------------
# Register arithmetic tools
# ----------------------------------------------------------------------
def add_fn(a, b):
    return a + b

def multiply_fn(a, b):
    return a * b

def universal_fn(expression):
    return eval(str(expression), {"__builtins__": None}, {})

ToolRegistry.register("add_tool", add_fn, ["a", "b"], "result", cost=0.1)
ToolRegistry.register("multiply_tool", multiply_fn, ["a", "b"], "result", cost=0.1)
ToolRegistry.register("universal_tool", universal_fn, ["expression"], "result", cost=0.5)


# ----------------------------------------------------------------------
# Data Generator
# ----------------------------------------------------------------------
def generate_arithmetic_sample(op_type: str = None):
    if op_type is None:
        op_type = np.random.choice(["sum", "product"])
    a = np.random.uniform(-10, 10)
    b = np.random.uniform(-10, 10)
    if op_type == "sum":
        true_result = a + b
        expression = f"{a} + {b}"
    else:
        true_result = a * b
        expression = f"{a} * {b}"
    return {
        "a": a,
        "b": b,
        "op_hint": op_type,
        "expression": expression,
        "true_result": true_result
    }


# ----------------------------------------------------------------------
# Custom Agent
# ----------------------------------------------------------------------
class ArithmeticHPMAgent:
    def __init__(self, tool_names: List[str]):
        self.tool_names = tool_names
        self.tool_patterns = [ToolRegistry.create_pattern(name) for name in tool_names]
        self.base_patterns = [RegressionPattern(input_dim=4, output_dim=1) for _ in range(2)]
        self.population = PatternPopulation(self.base_patterns + self.tool_patterns)
        self.eval_mgr = EvaluatorManager()
        self.compiler = SubstrateCompiler(use_gp=False)
        self.meta = MetaToolOrchestrator(available_tools=tool_names, context_feature_dim=8)
        self.loss_history = deque(maxlen=20)
        self.tool_history = deque(maxlen=100)
        self.correct_selections = 0
        self.total_selections = 0
        
    def _compute_features(self, sample: Dict) -> torch.Tensor:
        features = []
        features.append(sample["a"] / 10.0)
        features.append(sample["b"] / 10.0)
        features.append(1.0 if sample["op_hint"] == "sum" else 0.0)
        features.append(1.0 if sample["op_hint"] == "product" else 0.0)
        if len(self.loss_history) > 0:
            features.append(np.mean(self.loss_history) / 5.0)
            features.append(np.std(self.loss_history) / 5.0)
        else:
            features.extend([0.0, 0.0])
        if len(self.tool_history) > 0:
            features.append(self.tool_history.count("add_tool") / len(self.tool_history))
            features.append(self.tool_history.count("multiply_tool") / len(self.tool_history))
        else:
            features.extend([0.0, 0.0])
        while len(features) < 8: features.append(0.0)
        return torch.tensor(features[:8], dtype=torch.float32)
    
    def step(self, sample: Dict):
        features = self._compute_features(sample)
        meta_out = self.meta.sample({"context_features": features})
        selected_tool = meta_out["selected_tool"]
        tool_idx = meta_out["tool_idx"]
        self.tool_history.append(selected_tool)
        self.total_selections += 1
        optimal_tool = "add_tool" if sample["op_hint"] == "sum" else "multiply_tool"
        if selected_tool == optimal_tool: self.correct_selections += 1
        
        tool_pat = next(p for p in self.tool_patterns if p.tool_name == selected_tool)
        inputs = {k: sample[k] for k in tool_pat.input_keys if k in sample}
        tool_result = tool_pat.sample(inputs)
        pred_result = tool_result["result"]
        if isinstance(pred_result, torch.Tensor): pred_result = pred_result.item()
        
        error = min(5.0, abs(pred_result - sample["true_result"]))
        reward = -error - tool_pat.cost
        self.loss_history.append(error)
        
        meta_obs = {
            "context_features": features.unsqueeze(0),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "tool_idx_taken": tool_idx.unsqueeze(0) if tool_idx.dim() == 0 else tool_idx
        }
        self.meta.update_parameters(meta_obs)
        
        obs = {
            "a": torch.tensor([sample["a"]], dtype=torch.float32),
            "b": torch.tensor([sample["b"]], dtype=torch.float32),
            "expression": sample["expression"], 
            "input": torch.tensor([sample["a"]/10.0, sample["b"]/10.0, 1.0 if sample["op_hint"]=="sum" else 0.0, 1.0 if sample["op_hint"]=="product" else 0.0], dtype=torch.float32),
            "target": torch.tensor([sample["true_result"]], dtype=torch.float32)
        }
        self.population.step(self.eval_mgr, obs, self.compiler)
        return selected_tool, reward

def compile_meta_to_symbolic(agent: ArithmeticHPMAgent, n_samples: int = 500) -> SymbolicPattern:
    X, y = [], []
    for _ in range(n_samples):
        sample = generate_arithmetic_sample()
        features = agent._compute_features(sample).numpy()
        with torch.no_grad():
            tool_idx = agent.meta.sample({"context_features": torch.tensor(features)})["tool_idx"].item()
        X.append(features); y.append(tool_idx)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(np.array(X), np.array(y))
    def symbolic_policy(context):
        features = agent._compute_features(context).numpy()
        return agent.tool_names[clf.predict([features])[0]]
    sym_pat = SymbolicPattern(forward_fn=symbolic_policy, required_keys=["a", "b", "op_hint", "expression"], pattern_id="symbolic_tool_policy")
    sym_pat.tree_ = clf
    return sym_pat

def main():
    print("=== HPM Tool Compilation Task ===\n")
    agent = ArithmeticHPMAgent(tool_names=["add_tool", "multiply_tool", "universal_tool"])
    n_steps = 2000
    print("Phase 1: Training neural tool selection policy...")
    for step in range(n_steps):
        tool, reward = agent.step(generate_arithmetic_sample())
        if step % 200 == 0:
            acc = agent.correct_selections / agent.total_selections
            recent = list(agent.tool_history)
            dist = {t: recent.count(t) for t in agent.tool_names}
            print(f"  Step {step}: acc={acc:.3f}, dist={dist}")
            agent.correct_selections = 0; agent.total_selections = 0
            
    print("\nPhase 2: Compiling neural policy to symbolic...")
    sym_pat = compile_meta_to_symbolic(agent)
    print("\nPhase 3: Verification...")
    test_samples = [generate_arithmetic_sample() for _ in range(500)]
    neural_correct = 0; symbolic_correct = 0
    for sample in test_samples:
        features = agent._compute_features(sample)
        with torch.no_grad():
            neural_tool = agent.meta.sample({"context_features": features})["selected_tool"]
        sym_tool = sym_pat.forward_fn(sample)
        optimal = "add_tool" if sample["op_hint"] == "sum" else "multiply_tool"
        if neural_tool == optimal: neural_correct += 1
        if sym_tool == optimal: symbolic_correct += 1
    print(f"  Neural accuracy:   {neural_correct/500:.3f}")
    print(f"  Symbolic accuracy: {symbolic_correct/500:.3f}")
    print("\n=== Extracted Symbolic Rules ===")
    from sklearn.tree import export_text
    print(export_text(sym_pat.tree_, feature_names=["a", "b", "is_sum", "is_product", "loss_mean", "loss_std", "add_freq", "mul_freq"]))

if __name__ == "__main__":
    main()
