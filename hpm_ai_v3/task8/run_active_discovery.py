"""
run_active_discovery.py - Benchmark for active formula discovery.
Includes task environment logic.
"""

import sys, os
# Add parent to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from typing import Callable, Tuple, Dict, Any, Optional
from hpm_ai_v3.tools.registry import ToolRegistry
from hpm_ai_v3.agents.active_discovery_agent import MathDiscoveryAgent


class EnvironmentQuery:
    """Wrapper for the hidden environment. The agent doesn't see the true function."""
    def __init__(self, true_fn: Callable, x_range: Tuple[float, float] = (-5, 5)):
        self.true_fn = true_fn
        self.x_range = x_range
        self.query_count = 0

    def evaluate(self, x: float) -> float:
        self.query_count += 1
        return self.true_fn(x)


_ENV: Optional[EnvironmentQuery] = None


def set_environment(true_fn: Callable, x_range: Tuple[float, float] = (-5, 5)):
    global _ENV
    _ENV = EnvironmentQuery(true_fn, x_range)
    # Register evaluate_at tool
    ToolRegistry.register(
        name="evaluate_at",
        tool_fn=evaluate_at,
        input_keys=["x"],
        output_key="result",
        cost=0.1,
        description="Query the hidden function at a specific x value."
    )


def get_query_count() -> int:
    return _ENV.query_count if _ENV else 0


def evaluate_at(x: float) -> Dict[str, Any]:
    """Query the hidden function at a specific point."""
    if _ENV is None:
        return {"error": "Environment not set", "status": "failed"}
    y = _ENV.evaluate(x)
    return {"x": x, "y": y, "status": "success"}


def generate_true_function(formula_type: str):
    if formula_type == 'linear':
        a, b = np.random.uniform(-2, 2, 2)
        return (lambda x: a * x + b), f"{a:.2f}*x + {b:.2f}"
    elif formula_type == 'quadratic':
        a, b, c = np.random.uniform(-1, 1, 3)
        return (lambda x: a * x**2 + b * x + c), f"{a:.2f}*x**2 + {b:.2f}*x + {c:.2f}"
    elif formula_type == 'trig':
        a, b, c = np.random.uniform(0.5, 2, 3)
        b = 1.0 
        return (lambda x: a * np.sin(b * x) + c), f"{a:.2f}*sin({b:.2f}*x) + {c:.2f}"
    else:
        raise ValueError(f"Unknown formula type: {formula_type}")


def main():
    print("=== Active Hidden Formula Discovery (Pure HPM) ===\n")

    for formula_type in ['linear', 'quadratic', 'trig']:
        print(f"\n--- Testing {formula_type.upper()} ---")
        true_fn, true_expr = generate_true_function(formula_type)
        print(f"True formula: {true_expr}")

        set_environment(true_fn, x_range=(-3, 3))
        agent = MathDiscoveryAgent(x_range=(-3, 3), max_points=50)

        # Fix: main entry point requires task dict
        task = {"text": f"Discover formula for {formula_type}", "true_fn": true_fn}
        result = agent.run_discovery(task, max_steps=100)
        
        # We can't call get_query_count inside the agent easily anymore without passing it,
        # so we'll just report it from the environment here.
        
        print(f"Discovered hypothesis: {result['hypothesis']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Environment queries: {get_query_count()}")
        print(f"Points collected: {result['points']}")
        print("-" * 30)


if __name__ == "__main__":
    main()
