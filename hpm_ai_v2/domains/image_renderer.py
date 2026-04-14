"""
ImageRenderer — Specialized renderer for the image domain.
"""
from __future__ import annotations
import ast
import numpy as np
from typing import Any, List, Optional, TYPE_CHECKING
from hpm_ai_v2.utils.base_renderer import Renderer
from hpm_ai_v2.domains.image_codegen import ImageCodeGenerator
from hfn.hfn import HFN

if TYPE_CHECKING:
    from hpm_ai_v2.domains.image_domain import ImageDomainConfig

class ImageRenderer(Renderer):
    """
    Renders image transformation macros into Python PIL code.
    """
    def __init__(self, config: ImageDomainConfig):
        self.config = config
        self.code_gen = ImageCodeGenerator()

    def render(self, node: HFN) -> str:
        # 1. Flatten the node's structure into a sequence of primitive ops
        ops = self._extract_ops(node)
        
        # 2. Build code body
        lines = [
            "from PIL import Image, ImageFilter, ImageEnhance",
            "img = inp",
            "# apply operations in order"
        ]
        for op in ops:
            ast_node = self.code_gen.generate(op, {"var": "img"})
            ast.fix_missing_locations(ast_node)
            code_line = ast.unparse(ast_node)
            lines.append(f"{code_line}")
        
        lines.append("res = img")
        return "\n".join(lines)

    def render_function(self, node: HFN, func_name: str = "macro_func") -> str:
        """Render a macro as a standalone Python function definition."""
        body = self.render(node)
        indented = body.replace('\n', '\n    ')
        return f"def {func_name}(inp, inputs):\n    {indented}\n    return res"

    def _extract_ops(self, node: HFN) -> List[str]:
        """Traverse the node's inputs (multi-polygraph) to collect primitive op names."""
        ops = []
        if node.inputs:
            for child in node.inputs:
                ops.extend(self._extract_ops(child))
        else:
            # Leaf node – extract concept from its mu vector
            concept = self._get_concept(node)
            if concept:
                ops.append(concept)
        return ops

    def _get_concept(self, node: HFN) -> Optional[str]:
        """Extract concept name from HFN mu vector or ID."""
        # Check ID prefix first (standard for priors)
        for c in self.config.concepts:
            if node.id == f"prior_rule_{c}":
                return c
                
        # Fallback to mu vector middle slice
        start = self.config.S_DIM
        end = start + self.config.DIM
        vec = node.mu[start:end]
        if np.max(vec) > 0.5:
            idx = np.argmax(vec)
            return self.config.concepts[idx]
        return None
