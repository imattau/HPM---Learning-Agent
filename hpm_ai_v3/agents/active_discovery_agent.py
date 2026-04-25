"""
active_discovery_agent.py - Pure HPM agent for hidden formula discovery.
"""

import torch
import numpy as np
import random
from typing import Dict, Any, List, Tuple, Optional

from .base_discovery import PureAgnosticDiscoveryAgent, ActionPattern
from hpm_ai_v3.tools.registry import ToolRegistry


class MathDiscoveryAgent(PureAgnosticDiscoveryAgent):
    """
    Agnostic agent for hidden formula discovery.
    """
    def __init__(self, x_range: Tuple[float, float] = (-5, 5), max_points: int = 50):
        from hpm_ai_v3.tools.python_substrate import register_python_substrate
        register_python_substrate()
        
        super().__init__(context_feature_dim=32)
        
        # Add evaluate_at as a primitive ActionPattern
        self.population.patterns.append(ActionPattern("evaluate_at"))
        
        self.x_range = x_range
        self.max_points = max_points
        self.points: List[Tuple[float, float]] = []
        self.confidence = 0.0

    def initialize_task(self, task: Dict) -> Dict:
        self.points = []
        self.confidence = 0.0
        self.predicted_solution = None
        
        # Register environment tool
        true_fn = task.get("true_fn")
        if true_fn:
            def _env_eval(x: float):
                y = true_fn(x)
                self.points.append((x, y))
                return {"x": x, "y": y, "status": "success"}
            ToolRegistry.register("evaluate_at", _env_eval, ["x"], "result", 0.1)
            
            # Register explorer tool
            def _gen_x():
                return random.uniform(*self.x_range)
            ToolRegistry.register("generate_random_x", _gen_x, [], "result", 0.0)

        return {"points": self.points}

    def evaluate_solution(self, solution: Any) -> float:
        if solution is None or not self.points: return -1.0
        errors = []
        for px, py in self.points:
            try:
                pred = np.polyval(solution, px)
                errors.append((py - pred)**2)
            except:
                errors.append(100.0)
        mse = float(np.mean(errors))
        self.confidence = 1.0 - min(np.sqrt(mse) / 0.1, 1.0)
        return 1.0 - min(mse, 2.0)

    def extract_features(self) -> torch.Tensor:
        f = [len(self.points) / self.max_points]
        if len(self.points) >= 3:
            ys = np.array([p[1] for p in self.points])
            f.extend([float(np.mean(ys)), float(np.std(ys)), float(np.ptp(ys))])
        else:
            f.extend([0.0, 0.0, 0.0])
        f.append(self.confidence)
        f.append(1.0 if self.predicted_solution is not None else 0.0)
        while len(f) < 32: f.append(0.0)
        return torch.tensor(f[:32], dtype=torch.float32)

    def run_discovery(self, task: Dict, max_steps: int = 100) -> Dict[str, Any]:
        sol = self.run_episode(task, max_steps)
        hypothesis = None
        if sol:
            degree = len(sol) - 1
            terms = []
            for i, c in enumerate(sol):
                power = degree - i
                if abs(c) < 1e-4: continue
                if power == 0: terms.append(f"{c:.4f}")
                elif power == 1: terms.append(f"{c:.4f}*x")
                else: terms.append(f"{c:.4f}*x**{power}")
            hypothesis = " + ".join(terms).replace("+ -", "- ")
            
        return {
            "hypothesis": hypothesis,
            "confidence": self.confidence,
            "points": len(self.points)
        }
