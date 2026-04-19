"""MathOracle: Symbolic mathematics verification and state computation."""
from __future__ import annotations
import numpy as np
from typing import Any, List, Optional, TYPE_CHECKING
from hpm_ai_v2.utils.oracle.base import BaseOracle

if TYPE_CHECKING:
    from hpm_ai_v2.domains.math_domain import MathDomainConfig

class MathOracle(BaseOracle):
    """
    Oracle for symbolic math.
    Uses sympy for expression comparison and structural fingerprinting.
    """
    def __init__(self, config: MathDomainConfig):
        self.config = config

    def compute_state(
        self,
        outputs: List[Any],
        errors: List[Optional[str]],
        code: str = "",
        inputs: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Compute a state vector summarizing the mathematical results.
        If errors are present, state is zeroed out.
        """
        if not outputs or any(e is not None for e in errors):
            return np.zeros(self.config.S_DIM)
            
        import sympy
        
        # We take the first output (most math tasks are single-expression)
        result = outputs[0]
        
        if isinstance(result, (str, sympy.Basic)):
            # Use the config's encoding logic (structural features)
            mu = self.config.encode_expression(str(result)) if isinstance(result, str) else self.config.encode_sympy(result)
            return mu[:self.config.S_DIM]
            
        # Fallback for numeric results
        state = np.zeros(self.config.S_DIM)
        if isinstance(result, (int, float, np.number)):
            state[0] = float(result)
            
        return state

    def are_equal(self, expr1: Any, expr2: Any) -> bool:
        """Symbolic equality check using sympy."""
        if expr1 is None or expr2 is None: return False
        import sympy
        try:
            # Handle potential HFN nodes passed in
            e1 = getattr(expr1, "metadata", {}).get("sympy_expr", expr1)
            e2 = getattr(expr2, "metadata", {}).get("sympy_expr", expr2)
            
            # Convert strings to sympy if needed
            if isinstance(e1, str): e1 = sympy.simplify(e1)
            if isinstance(e2, str): e2 = sympy.simplify(e2)
            
            # Final check for basic equality
            if e1 == e2: return True
            
            diff = sympy.simplify(e1 - e2)
            if diff.is_zero: return True
            
            # Numerical check for float differences
            try:
                # Test at a few points if it's not obviously zero
                free = diff.free_symbols
                if not free:
                    return abs(float(diff.evalf())) < 1e-9
                
                # Sample a point
                test_point = {s: 1.2345 for s in free}
                val = diff.evalf(subs=test_point)
                return abs(float(val)) < 1e-9
            except Exception:
                return False
        except Exception:
            return str(expr1) == str(expr2)
