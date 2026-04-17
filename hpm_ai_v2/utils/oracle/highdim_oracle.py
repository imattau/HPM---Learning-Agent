"""
HighDimOracle — computes a 20-D state vector for high-dimensional scaling (SP100).
"""
from __future__ import annotations
import numpy as np
from typing import Any, List, Optional, TYPE_CHECKING
from .base import BaseOracle

if TYPE_CHECKING:
    from hpm_ai_v2.domains.highdim_domain import HighDimDomainConfig

class HighDimOracle(BaseOracle):
    """
    Computes a fixed-D empirical state vector for high-D scaling.
    Heuristically identifies correlated dimensions to guide search.
    """
    def __init__(self, config: HighDimDomainConfig):
        self.config = config

    def compute_state(
        self,
        outputs: List[Any],
        errors: List[Optional[str]],
        code: str = "",
        inputs: Optional[List[Any]] = None,
    ) -> np.ndarray:
        s_dim = self.config.S_DIM
        s = np.zeros(s_dim)
        
        valid_indices = [i for i, e in enumerate(errors) if e is None]
        if not valid_indices or len(valid_indices) < 2:
            s[0] = 0.0 # invalid
            s[9] = 1.0 # error flag
            return s
        
        s[0] = 1.0 # valid
        valid_outputs = np.array([float(outputs[i]) for i in valid_indices])
        
        # Dim 3: Magnitude
        if np.std(valid_outputs) > 0:
            s[3] = float(np.mean(valid_outputs))
            
        # Correlation guidance (starting at index 20)
        if inputs is not None:
            v_inputs = np.array([inputs[i] for i in valid_indices]) # (N, D)
            y_std = np.std(valid_outputs)
            
            if y_std > 1e-6:
                for d in range(self.config.D):
                    if 20 + d >= s_dim:
                        break
                    feat = v_inputs[:, d]
                    if np.std(feat) > 1e-6:
                        # Absolute correlation as a guide
                        c = np.abs(np.corrcoef(feat, valid_outputs)[0, 1])
                        # We use a significant value (e.g. 10.0) to make it attractive in distance space
                        s[20 + d] = 10.0 * float(c)
        
        # Code structure flags (10-15)
        if code:
            s[10] = 1.0 if 'v**2' in code else 0.0
            s[11] = 1.0 if 'np.sin' in code else 0.0
            s[12] = 1.0 if 'inp[0]' in code else 0.0
            s[13] = 1.0 if 'inp[1]' in code else 0.0
            s[14] = 1.0 if 'inp[2]' in code else 0.0
            s[15] = 1.0 if 'inp[3]' in code else 0.0
            
        return s
