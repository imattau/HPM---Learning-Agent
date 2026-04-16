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
        """Render a sequence of operations into a stack-based execution."""
        lines = [
            "import numpy as np",
            "# Inputs: N=inp[0], L=inp[1], theta=inp[2], Q=inp[3], rho=inp[4]",
            "N, L, theta, Q, rho = inp[0], inp[1], inp[2], inp[3], inp[4]",
            "stack = [0.0]",
            "def push(v): stack.append(float(v))",
            "def pop(): return stack.pop() if len(stack) > 1 else stack[0]",
            "def top(): return stack[-1]"
        ]
        lines.extend(self._render_node_hierarchical(node))
        lines.append("res = top()")
        return "\n".join(lines)

    def _render_node_hierarchical(self, node: HFN) -> List[str]:
        lines = []
        if node.relation_type == "macro":
            lines.append(f"# BEGIN MACRO: {node.id}")
            for child in node.inputs:
                lines.extend(self._render_node_hierarchical(child))
            lines.append(f"# END MACRO: {node.id}")
        else:
            concept = self._get_concept(node)
            if concept == "VAR_Q":
                lines.append("push(Q)")
            elif concept == "VAR_RHO":
                lines.append("push(rho)")
            elif concept == "VAR_L":
                lines.append("push(L)")
            elif concept == "VAR_THETA":
                lines.append("push(theta)")
            elif concept == "OP_SQUARE":
                lines.append("v = pop(); push(v**2)")
            elif concept == "OP_SIN":
                lines.append("v = pop(); push(np.sin(v * np.pi))")
            elif concept == "OP_MUL_Q":
                lines.append("v = pop(); push(v * Q)")
            elif concept == "OP_MUL_RHO":
                lines.append("v = pop(); push(v * rho)")
            elif concept == "OP_MUL":
                lines.append("b = pop(); a = pop(); push(a * b)")
            elif concept == "OP_SIGN":
                lines.append("v = pop(); push(np.sign(v))")
        return lines

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
