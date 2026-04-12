"""
Refactored Stateful Oracle for SP56 — Hierarchical 90D Manifold.

Calculates Level 1 (Base), Level 2 (Relational), and Level 3 (Meta-Relational)
vectors for sequence-based observations using continuous geometric embeddings.
"""

import numpy as np
from typing import Any, List, Optional

S_DIM = 30 # Slice Dimension
D = 90     # Total Dimension [30 L1 | 30 L2 | 30 L3]

class StatefulOracleSP56:
    """
    Tracks temporal state and populates the 90D hierarchical manifold.
    Uses continuous embeddings to ensure L2 and L3 deltas are geometrically meaningful.
    """
    def __init__(self, seed: int = 42):
        self.prev_l1: Optional[np.ndarray] = None
        self.prev_l2: Optional[np.ndarray] = None

    def reset(self):
        """Reset temporal state for a new sequence."""
        self.prev_l1 = None
        self.prev_l2 = None

    def encode_l1(self, data: Any) -> np.ndarray:
        """
        Map raw data to a dense 30D latent state, preserving basic geometry.
        This avoids hashing to ensure that transitions like '+1' produce consistent vectors.
        """
        vec = np.zeros(S_DIM)
        if data is None:
            return vec
            
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            # Numeric values sit on axis 0
            vec[0] = float(data)
        elif isinstance(data, bool):
            # Boolean values sit on axis 1
            vec[1] = 1.0 if data else -1.0
        elif isinstance(data, (tuple, list, np.ndarray)):
            # Spatial coordinates sit on axis 2, 3, 4...
            for i, val in enumerate(data):
                if i + 2 < S_DIM and isinstance(val, (int, float)):
                    vec[i+2] = float(val)
        
        # Scale to keep variance reasonable for HFN Gaussian kernels
        return vec * 0.1

    def compute(self, current_data: Any) -> np.ndarray:
        """
        Produce a 90D observation for the current data point.
        L1 = Current state
        L2 = Transition from previous state (L1_t - L1_{t-1})
        L3 = Transition from previous transition (L2_t - L2_{t-1})
        """
        l1 = self.encode_l1(current_data)
        l2 = np.zeros(S_DIM)
        l3 = np.zeros(S_DIM)
        
        if self.prev_l1 is not None:
            l2 = l1 - self.prev_l1
            if self.prev_l2 is not None:
                l3 = l2 - self.prev_l2
        
        full_vec = np.zeros(D)
        full_vec[0:30] = l1
        full_vec[30:60] = l2
        full_vec[60:90] = l3
        
        self.prev_l1 = l1
        self.prev_l2 = l2
        
        return full_vec

    def compute_sequence(self, sequence: List[Any]) -> List[np.ndarray]:
        """Convert a list of raw states into a list of 90D vectors."""
        self.reset()
        results = []
        for item in sequence:
            results.append(self.compute(item))
        return results
