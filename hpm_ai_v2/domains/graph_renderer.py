"""
GraphRenderer — Specialized renderer for the graph domain.
"""
from __future__ import annotations
import numpy as np
from typing import Any, List, Optional, TYPE_CHECKING
from hpm_ai_v2.utils.base_renderer import Renderer
from hfn.hfn import HFN

if TYPE_CHECKING:
    from hpm_ai_v2.domains.graph_domain import GraphDomainConfig

class GraphRenderer(Renderer):
    """
    Renders graph transformation macros into Python NetworkX code.
    """
    def __init__(self, config: GraphDomainConfig):
        self.config = config

    def render(self, node: HFN) -> str:
        # 1. Flatten the node's structure into a sequence of primitive ops
        ops = self._extract_ops(node)
        
        # 2. Build code body
        lines = [
            "import networkx as nx",
            "G = inp.copy() if hasattr(inp, 'copy') else nx.Graph(inp)",
            "res = G"
        ]
        
        indent = 0
        for op in ops:
            if op == "BLOCK_END":
                if indent > 0: indent -= 1
                continue
                
            prefix = "    " * indent
            if op == "VAR_INP":
                lines.append(f"{prefix}G = inp.copy() if hasattr(inp, 'copy') else nx.Graph(inp)")
            elif op == "ADD_NODE":
                lines.append(f"{prefix}G.add_node(max(G.nodes)+1 if G.nodes else 0)")
            elif op == "REMOVE_NODE":
                lines.append(f"{prefix}if G.nodes: G.remove_node(max(G.nodes))")
            elif op == "ADD_EDGE":
                lines.append(f"{prefix}if len(G.nodes) >= 2:")
                lines.append(f"{prefix}    nodes = list(G.nodes)")
                lines.append(f"{prefix}    G.add_edge(nodes[0], nodes[1])")
            elif op == "REMOVE_EDGE":
                lines.append(f"{prefix}if G.edges: G.remove_edge(*list(G.edges)[0])")
            elif op == "CLEAR_GRAPH":
                lines.append(f"{prefix}G.clear()")
            elif op == "COPY_GRAPH":
                lines.append(f"{prefix}G = G.copy()")
            elif op == "FOR_EACH_NODE":
                lines.append(f"{prefix}for node in list(G.nodes()):")
                indent += 1
                prefix = "    " * indent # Update prefix for the 'res = G' and body
            elif op == "RELABEL_NODE":
                lines.append(f"{prefix}G = nx.relabel_nodes(G, {{node: node + 1}})")

            lines.append(f"{prefix}res = G")

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
            # Leaf node – extract concept
            concept = self._get_concept(node)
            if concept:
                ops.append(concept)
        return ops

    def _get_concept(self, node: HFN) -> Optional[str]:
        """Extract concept name from HFN mu vector or ID."""
        for c in self.config.concepts:
            if node.id == f"prior_rule_{c}" or node.id == f"graph_op_{c}":
                return c
        
        start = self.config.S_DIM
        end = start + self.config.DIM
        vec = node.mu[start:end]
        if np.max(vec) > 0.5:
            idx = np.argmax(vec)
            return self.config.concepts[idx]
        return None
