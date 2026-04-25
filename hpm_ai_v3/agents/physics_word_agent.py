"""
physics_word_agent.py - Agnostic agent for physics word problems.
"""

import torch
import numpy as np
import re
from typing import Dict, Any, List, Tuple, Optional

from .base_discovery import AgnosticDiscoveryAgent
from hpm_ai_v3.tools.registry import ToolRegistry


class PhysicsWordAgent(AgnosticDiscoveryAgent):
    """
    Agnostic agent for solving physics word problems.
    Learns to use NLP and physics libraries via HPM dynamics.
    """
    def __init__(self):
        from hpm_ai_v3.tools.python_substrate import register_python_substrate
        register_python_substrate()
        super().__init__(population_size=3, context_feature_dim=64)

    def initialize_task(self, task: Dict) -> Dict:
        return {"text": task["text"], "knowns": {}, "unknowns": [], "formula": None}

    def evaluate_solution(self, solution: Any) -> float:
        if solution is None:
            return -1.0
        try:
            true_answer = float(self.current_task["answer"])
            tolerance = self.current_task.get("tolerance", 0.05) # 5% tolerance
            sol_float = float(solution)
            
            error = abs(sol_float - true_answer)
            # Normalize error: 1.0 for perfect, 0.0 for 100% error
            reward = 1.0 - min(error / (abs(true_answer) + 1e-6), 1.0)
            
            if error <= abs(true_answer) * tolerance:
                reward += 1.0 # Success bonus
                
            return reward
        except:
            return -0.5

    def extract_features(self) -> torch.Tensor:
        # Simple features: task length, number of knowns, has solution?
        f = [len(self.context.get("text", "")) / 500.0]
        f.append(len(self.context.get("knowns", {})) / 10.0)
        f.append(len(self.context.get("unknowns", [])) / 5.0)
        f.append(1.0 if self.predicted_solution is not None else 0.0)
        while len(f) < 64:
            f.append(0.0)
        return torch.tensor(f[:64], dtype=torch.float32)

    def process_tool_result(self, tool_name: str, args: Dict, result: Dict) -> bool:
        if result.get("status") != "success":
            return False
            
        res_val = result.get("result")
        if tool_name == "python_call":
            # Try to extract a numeric answer from any library call
            if isinstance(res_val, (int, float)):
                self.predicted_solution = float(res_val)
                return True
            elif isinstance(res_val, str):
                # Try to find a number in the string (e.g. from SymPy)
                match = re.search(r'[\d\.]+', res_val)
                if match:
                    try:
                        self.predicted_solution = float(match.group())
                        return True
                    except:
                        pass
                        
            # Store structured results in context for future use
            if isinstance(res_val, dict):
                self.context.update(res_val)
            elif isinstance(res_val, list):
                self.context["data"] = res_val
                
        return False

    def _generate_call_args(self, module: str, function: str) -> Dict[str, Any]:
        args = super()._generate_call_args(module, function)
        
        # Adaptation: if calling spacy or tokenize, pass text
        if "tokenize" in function or "spacy" in module:
            args["text"] = self.context.get("text", "")
            
        return args
