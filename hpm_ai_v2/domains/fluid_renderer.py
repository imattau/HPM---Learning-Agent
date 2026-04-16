"""
FluidRenderer — specialized renderer for fluid dynamics (SP95).
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, TYPE_CHECKING
from hpm_ai_v2.utils.base_renderer import Renderer

if TYPE_CHECKING:
    from hfn.hfn import HFN
    from hpm_ai_v2.domains.fluid_domain import FluidDomainConfig

class FluidRenderer(Renderer):
    """
    Translates HFN nodes into executable Python code for fluid dynamics.
    Supports physical primitives like MOMENTUM_FLUX and TORQUE.
    """
    def __init__(self, config: FluidDomainConfig):
        self.config = config

    def render(self, node: HFN) -> str:
        """Render a sequence of operations into Python code."""
        # For simplicity, we assume a sequential execution model
        # where 'res' is the running result.
        ops = self._extract_ops(node)
        lines = [
            "import numpy as np",
            "# Inputs: N=inp[0], L=inp[1], theta=inp[2], Q_sign=inp[3], Q_mag=inp[4], rho=inp[5]",
            "N, L, theta, Q_sign, Q_mag, rho = inp[0], inp[1], inp[2], inp[3], inp[4], inp[5]",
            "A = 1.0  # Assumed unit area",
            "res = 0.0"
        ]
        
        # Local state for intermediate variables
        # We'll use a stack-like or register-like approach if needed, 
        # but for SP95 a simple sequence is often enough.
        for op in ops:
            if op == "MOMENTUM_FLUX":
                # F = rho * Q^2 / A
                lines.append("F = rho * (Q_mag**2) / A")
                lines.append("res = F")
            elif op == "TORQUE":
                # T = F * L
                lines.append("res = res * L")
            elif op == "SINE":
                # Multiply by sin(theta)
                # theta is 0-1 (90 deg = 0.5)
                lines.append("res = res * np.sin(theta * np.pi)")
            elif op == "SIGN":
                lines.append("res = np.sign(res)")
            elif op == "MULTIPLY":
                # Generic multiplication by some other parameter? 
                # Let's say by N (number of arms)
                lines.append("res = res * N")
            elif op == "COSINE":
                lines.append("res = res * np.cos(theta * np.pi)")

        return "\n".join(lines)

    def render_function(self, node: HFN, func_name: str = "macro_func") -> str:
        code = self.render(node)
        indented = code.replace("\n", "\n    ")
        return f"def {func_name}(inp):\n    {indented}\n    return res"

    def _extract_ops(self, node: HFN) -> List[str]:
        ops = []
        if node.inputs:
            for child in node.inputs:
                ops.extend(self._extract_ops(child))
        else:
            concept = self._get_concept(node)
            if concept:
                ops.append(concept)
        return ops

    def _get_concept(self, node: HFN) -> Optional[str]:
        # Priority 1: Named prior
        for c in self.config.concepts:
            if node.id == f"prior_rule_{c}":
                return c
        
        # Priority 2: Vector representation
        start = self.config.S_DIM
        end = start + self.config.DIM
        vec = node.mu[start:end]
        if np.max(vec) > 0.5:
            idx = np.argmax(vec)
            return self.config.concepts[idx]
        return None
