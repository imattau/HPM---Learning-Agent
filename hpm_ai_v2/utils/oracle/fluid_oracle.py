"""
FluidOracle — computes a 20-D state vector for fluid dynamics experiments (SP95).
"""
from __future__ import annotations
import numpy as np
from typing import Any, List, Optional, TYPE_CHECKING
from .base import BaseOracle

if TYPE_CHECKING:
    from hpm_ai_v2.domains.base import DomainConfig

class FluidOracle(BaseOracle):
    """
    Computes a fixed-D empirical state vector for fluid dynamics.
    Encodes the rotation direction (+1/-1) and code structure features.
    """
    def __init__(self, config: "DomainConfig"):
        self.config = config

    def compute_state(
        self,
        outputs: List[Any],
        errors: List[Optional[str]],
        code: str = "",
    ) -> np.ndarray:
        s_dim = self.config.S_DIM
        s = np.zeros(s_dim)
        
        valid_outputs = [o for o, e in zip(outputs, errors) if e is None]
        if not valid_outputs:
            # Handle failure state
            s[0] = 0.0
            s[9] = 1.0
            return s
        
        s[0] = 1.0
        
        # Dimension 3: Rotation direction (average of outputs)
        vals = [float(o) for o in valid_outputs if isinstance(o, (int, float, np.float64, np.int64))]
        if vals:
            s[3] = float(np.mean(vals))
            
        # Code structure flags (dimensions 10+)
        # We only keep generic complexity flags, no variable name leakage.
        s[10] = 1.0 if 'res**2' in code else 0.0
        s[11] = 1.0 if 'np.sin' in code else 0.0
        s[12] = 1.0 if 'np.sign' in code else 0.0
        
        return s
