"""
HighDimRenderer — specialized renderer for high-dimensional scaling (SP100).
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional, TYPE_CHECKING
from hpm_ai_v2.utils.base_renderer import Renderer

if TYPE_CHECKING:
    from hfn.hfn import HFN
    from hpm_ai_v2.domains.highdim_domain import HighDimDomainConfig

class HighDimRenderer(Renderer):
    """
    Translates HFN nodes into executable Python code for high-D inputs.
    Supports SELECT_i primitives for feature extraction.
    """
    def __init__(self, config: HighDimDomainConfig):
        self.config = config

    def render(self, node: HFN) -> str:
        """Render a sequence of operations into a stack-based execution."""
        lines = [
            "import numpy as np",
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
            for child in node.inputs:
                lines.extend(self._render_node_hierarchical(child))
        else:
            concept = self._get_concept(node)
            if concept and concept.startswith("SELECT_"):
                try:
                    idx = int(concept.split("_")[1])
                    lines.append(f"push(inp[{idx}])")
                except (IndexError, ValueError):
                    lines.append("push(0.0)")
            elif concept == "OP_MUL":
                lines.append("b = pop(); a = pop(); push(a * b)")
            elif concept == "OP_ADD":
                lines.append("b = pop(); a = pop(); push(a + b)")
            elif concept == "OP_SUB":
                lines.append("b = pop(); a = pop(); push(a - b)")
            elif concept == "OP_DIV":
                lines.append("b = pop(); a = pop(); push(a / b if abs(b) > 1e-6 else 0.0)")
            elif concept == "OP_SIN":
                lines.append("v = pop(); push(np.sin(v))")
            elif concept == "OP_SQUARE":
                lines.append("v = pop(); push(v**2)")
            elif concept == "OP_CONST":
                lines.append("push(1.0)")
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
