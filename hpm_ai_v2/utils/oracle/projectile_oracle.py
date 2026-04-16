"""
ProjectileOracle — computes a 20-D state vector for projectile motion (SP96).
"""
from __future__ import annotations
import numpy as np
from typing import Any, List, Optional, TYPE_CHECKING
from .base import BaseOracle

if TYPE_CHECKING:
    from hpm_ai_v2.domains.base import DomainConfig

class ProjectileOracle(BaseOracle):
    """
    Computes a fixed-D empirical state vector for projectile motion.
    Encodes the current output value and code structure features.
    """
    def __init__(self, config: "DomainConfig"):
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
            s[0] = 0.0
            s[9] = 1.0
            return s
        
        s[0] = 1.0
        valid_outputs = np.array([float(outputs[i]) for i in valid_indices])
        
        # Dimension 3: Average height
        s[3] = float(np.mean(valid_outputs))
            
        # [NEW] Correlations with inputs (Dimensions 4, 5, 6, 7)
        if inputs is not None:
            # Inputs are typically lists of [theta, v0, t, g]
            v_inputs = [inputs[i] for i in valid_indices]
            theta_vals = np.array([v[0] for v in v_inputs])
            v0_vals = np.array([v[1] for v in v_inputs])
            t_vals = np.array([v[2] for v in v_inputs])
            
            # Simple correlations
            def get_corr(a, b):
                if np.std(a) < 1e-6 or np.std(b) < 1e-6: return 0.0
                return float(np.corrcoef(a, b)[0, 1])
            
            s[4] = 10.0 * get_corr(valid_outputs, np.sin(theta_vals))
            s[5] = 10.0 * get_corr(valid_outputs, v0_vals)
            s[6] = 10.0 * get_corr(valid_outputs, t_vals)
            s[7] = 10.0 * get_corr(valid_outputs, t_vals**2)
        
        # Code structure flags (dimensions 10+)
        # Only set if code is actually provided (empirical programs)
        # For the goal state (code=""), these remain 0.
        if code:
            s[10] = 1.0 if 'v**2' in code else 0.0
            s[11] = 1.0 if 'np.sin' in code else 0.0
            s[12] = 1.0 if 'push(g)' in code else 0.0
            s[13] = 1.0 if 'push(0.5)' in code else 0.0
            s[14] = 1.0 if 'push(t)' in code else 0.0
            s[15] = 1.0 if 'push(v0)' in code else 0.0
        
        return s
