"""
Operator-Level Abstraction Tools for SP57.

Defines the OperatorOracle for state embedding and the Operator class
for functional transformations (f(x) = Ax + B).
"""

import numpy as np
from typing import Any, List, Optional, Tuple, Callable

S_DIM = 30 

class Operator:
    """
    A functional transformation representing a relation or meta-relation.
    In this experiment, we use an Affine Transformation: f(x) = weight * x + bias.
    """
    def __init__(self, weight: float = 1.0, bias: float = 0.0, name: str = "identity"):
        self.weight = weight
        self.bias = bias
        self.name = name

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the operator to a state vector."""
        return x * self.weight + self.bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.apply(x)

    def compose(self, other: 'Operator') -> 'Operator':
        """
        Returns a new operator representing the composition f(g(x)).
        f(x) = w1*x + b1, g(x) = w2*x + b2
        f(g(x)) = w1*(w2*x + b2) + b1 = (w1*w2)*x + (w1*b2 + b1)
        """
        new_w = self.weight * other.weight
        new_b = self.weight * other.bias + self.bias
        new_name = f"{self.name}({other.name})"
        return Operator(weight=new_w, bias=new_b, name=new_name)

    def __repr__(self):
        return f"Op({self.name}: w={self.weight:.2f}, b={self.bias:.2f})"

class OperatorOracle:
    """
    Consistent embedding oracle for SP57.
    Maps data to continuous axes in S_DIM.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed

    def encode(self, data: Any) -> np.ndarray:
        vec = np.zeros(S_DIM)
        if data is None:
            return vec
            
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            # Axis 0 for numeric
            vec[0] = float(data)
        elif isinstance(data, bool):
            # Axis 1 for boolean
            vec[1] = 1.0 if data else -1.0
        elif isinstance(data, (tuple, list, np.ndarray)):
            # Spatial axes
            for i, val in enumerate(data):
                if i + 2 < S_DIM and isinstance(val, (int, float)):
                    vec[i+2] = float(val)
        
        # Scale for HFN stability
        return vec * 0.1

    def decode_numeric(self, vec: np.ndarray) -> float:
        """Inverse mapping for Axis 0."""
        return float(vec[0]) * 10.0

    def infer_operator(self, x_prev: np.ndarray, x_curr: np.ndarray, name: str = "inferred") -> Operator:
        """
        Simple affine inference from two points.
        Note: This is under-determined for a full matrix, but for our 
        1D numeric axis, we assume we're finding the best w, b.
        """
        v_prev = x_prev[0]
        v_curr = x_curr[0]
        
        # If we have only two points, we can't distinguish between Add and Mul.
        # For priming, we usually assume bias-only (Add) or weight-only (Mul)
        # depending on the context. In true HPM, the search resolves this.
        # Here we'll return a 'Delta' operator (Add) as the default primitive.
        return Operator(weight=1.0, bias=v_curr - v_prev, name=name)
