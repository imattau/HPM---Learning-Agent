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
    """
    def __init__(self, name: str = "identity"):
        self.name = name

    def apply(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.apply(x)

    def compose(self, other: 'Operator') -> 'Operator':
        """Returns a new operator representing the composition f(g(x))."""
        return ComposedOperator(self, other)

    def get_params(self) -> Tuple[float, ...]:
        """Returns a tuple of functional parameters for deduplication."""
        raise NotImplementedError()

    def __repr__(self):
        return f"Op({self.name})"

class AffineOperator(Operator):
    """f(x) = weight * x + bias"""
    def __init__(self, weight: float = 1.0, bias: float = 0.0, name: str = "affine"):
        super().__init__(name)
        self.weight = weight
        self.bias = bias

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x * self.weight + self.bias

    def get_params(self) -> Tuple[float, ...]:
        return (float(round(self.weight, 4)), float(round(self.bias, 4)), 0.0)

    def __repr__(self):
        return f"Affine({self.name}: w={self.weight:.2f}, b={self.bias:.2f})"

class ModOperator(Operator):
    """f(x) = x % modulus"""
    def __init__(self, modulus: float = 1.0, name: str = "mod"):
        super().__init__(name)
        self.modulus = modulus

    def apply(self, x: np.ndarray) -> np.ndarray:
        # Scale back to numeric for mod, then re-scale
        val = x[0] * 10.0
        result = (val % self.modulus) * 0.1
        res_vec = x.copy()
        res_vec[0] = result
        return res_vec

    def get_params(self) -> Tuple[float, ...]:
        return (0.0, 0.0, float(round(self.modulus, 4)))

    def __repr__(self):
        return f"Mod({self.name}: m={self.modulus:.1f})"

class ComposedOperator(Operator):
    """f(g(x))"""
    def __init__(self, f: Operator, g: Operator):
        super().__init__(f"{f.name} ∘ {g.name}")
        self.f = f
        self.g = g

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.f.apply(self.g.apply(x))

    def get_params(self) -> Tuple[float, ...]:
        # For simplicity, we can't easily flatten all compositions to a single 
        # (w, b, m) tuple if they are truly non-linear, but for affine ∘ affine 
        # it would work. However, since we have Mod, we'll just return the 
        # recursive params.
        return (self.f.get_params(), self.g.get_params())

    def __repr__(self):
        return f"Composed({self.name})"

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
