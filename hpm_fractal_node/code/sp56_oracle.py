"""
Stateful Oracle for SP56 — Hierarchical 90D Manifold.

Calculates Level 1 (Base), Level 2 (Relational), and Level 3 (Meta-Relational)
vectors for sequence-based observations.
"""

import numpy as np
import zlib
from typing import Any, List, Optional, Tuple

S_DIM = 30 # Slice Dimension
D = 90     # Total Dimension [30 L1 | 30 L2 | 30 L3]

class StatefulOracleSP56:
    """
    Tracks temporal state and populates the 90D hierarchical manifold.
    
    L1: State at time t
    L2: Delta(S_t, S_{t-1})
    L3: Delta(R_t, R_{t-1})
    """
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        # Random projection matrix [32-bit hash -> 30D]
        self.projection = np.random.randn(32, S_DIM)
        self.projection /= np.linalg.norm(self.projection, axis=0)
        
        # Internal state tracking
        self.prev_l1: Optional[np.ndarray] = None
        self.prev_l2: Optional[np.ndarray] = None

    def reset(self):
        """Reset temporal state for a new sequence."""
        self.prev_l1 = None
        self.prev_l2 = None

    def _hash_to_vec(self, text: str) -> np.ndarray:
        """Map string to a 32D bit-vector using adler32 with different seeds."""
        bits = []
        for i in range(32):
            val = zlib.adler32(f"{i}:{text}".encode())
            bits.append(1.0 if val % 2 == 0 else -1.0)
        return np.array(bits)

    def encode_l1(self, data: Any) -> np.ndarray:
        """Map raw data to a dense 30D latent state."""
        if data is None:
            return np.zeros(S_DIM)
        text_rep = repr(data)
        high_dim_vec = self._hash_to_vec(text_rep)
        latent = high_dim_vec @ self.projection
        norm = np.linalg.norm(latent)
        if norm > 0:
            latent /= norm
        return latent

    def compute(self, current_data: Any) -> np.ndarray:
        """
        Produce a 90D observation for the current data point.
        Calculates L2 and L3 based on previous calls.
        """
        l1 = self.encode_l1(current_data)
        l2 = np.zeros(S_DIM)
        l3 = np.zeros(S_DIM)
        
        # 1. Calculate Level 2 (Relational Delta)
        if self.prev_l1 is not None:
            l2 = l1 - self.prev_l1
            
            # 2. Calculate Level 3 (Meta-Relational Delta)
            if self.prev_l2 is not None:
                l3 = l2 - self.prev_l2
        
        # 3. Build the 90D result
        full_vec = np.zeros(D)
        full_vec[0:30] = l1
        full_vec[30:60] = l2
        full_vec[60:90] = l3
        
        # 4. Update state
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
