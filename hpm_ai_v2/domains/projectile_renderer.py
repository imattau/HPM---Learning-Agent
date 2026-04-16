"""
ProjectileRenderer — specialized renderer for projectile motion (SP96).
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, TYPE_CHECKING
from hpm_ai_v2.utils.base_renderer import Renderer

if TYPE_CHECKING:
    from hfn.hfn import HFN
    from hpm_ai_v2.domains.projectile_domain import ProjectileDomainConfig

class ProjectileRenderer(Renderer):
    """
    Translates HFN nodes into executable Python code for projectile motion.
    Supports physical primitives like SIN, SQUARE, and CONST_05.
    """
    def __init__(self, config: ProjectileDomainConfig):
        self.config = config

    def render(self, node: HFN) -> str:
        """Render a sequence of operations into a stack-based execution."""
        lines = [
            "import numpy as np",
            "# Inputs: theta=inp[0], v0=inp[1], t=inp[2], g=inp[3]",
            "theta, v0, t, g = inp[0], inp[1], inp[2], inp[3]",
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
            if concept == "VAR_THETA":
                lines.append("push(theta)")
            elif concept == "VAR_V0":
                lines.append("push(v0)")
            elif concept == "VAR_T":
                lines.append("push(t)")
            elif concept == "VAR_G":
                lines.append("push(inp[3])")
            elif concept == "VAR_Z1":
                lines.append("push(inp[4])")
            elif concept == "VAR_Z2":
                lines.append("push(inp[5])")
            elif concept == "VAR_Z3":
                lines.append("push(inp[6])")
            elif concept == "OP_MUL":

                lines.append("b = pop(); a = pop(); push(a * b)")
            elif concept == "OP_ADD":
                lines.append("b = pop(); a = pop(); push(a + b)")
            elif concept == "OP_SUB":
                lines.append("b = pop(); a = pop(); push(a - b)")
            elif concept == "OP_NEG":
                lines.append("a = pop(); push(-a)")
            elif concept == "OP_SIN":
                lines.append("v = pop(); push(np.sin(v))")
            elif concept == "OP_SQUARE":
                lines.append("v = pop(); push(v**2)")
            elif concept == "OP_CONST_05":
                lines.append("push(0.5)")
        return lines

    def render_function(self, node: HFN, func_name: str = "macro_func") -> str:
        code = self.render(node)
        indented = code.replace("\n", "\n    ")
        return f"def {func_name}(inp):\n    {indented}\n    return res"

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
