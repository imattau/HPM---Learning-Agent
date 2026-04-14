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
            "# apply operations in order"
        ]
        for op in ops:
            if op == "ADD_NODE":
                lines.append("G.add_node(max(G.nodes)+1 if G.nodes else 0)")
            elif op == "REMOVE_NODE":
                lines.append("if G.nodes: G.remove_node(max(G.nodes))")
            elif op == "ADD_EDGE":
                lines.append("if len(G.nodes) >= 2:")
                lines.append("    nodes = list(G.nodes)")
                lines.append("    G.add_edge(nodes[0], nodes[1])")
            elif op == "REMOVE_EDGE":
                lines.append("if G.edges: G.remove_edge(*list(G.edges)[0])")
            elif op == "CLEAR_GRAPH":
                lines.append("G.clear()")
            elif op == "COPY_GRAPH":
                lines.append("G = G.copy()")
            elif op == "ADD_STAR":
                # Special concept for the star extension
                lines.append("new_node = max(G.nodes)+1 if G.nodes else 0")
                lines.append("nodes_to_connect = list(G.nodes)")
                lines.append("G.add_node(new_node)")
                lines.append("for n in nodes_to_connect:")
                lines.append("    G.add_edge(new_node, n)")
        
        lines.append("res = G")
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
