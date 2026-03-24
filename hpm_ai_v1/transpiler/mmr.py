"""Middle-Manifold Representation (MMR) for HPM AI v2.1.

Decouples Relational Logic from Python Syntax by representing code as 
abstract graphs of functional primitives. Transitioned to Vectorized Topology.
"""
import ast
import numpy as np
from typing import Dict, List, Any, Optional

# Shared padded dimension for the MMR Manifold
MMR_DIM = 32

def _get_basis_vector(name: str) -> np.ndarray:
    """Generate a stable, deterministic unit vector for a given node type."""
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    v = rng.standard_normal(MMR_DIM)
    return v / (np.linalg.norm(v) + 1e-9)

class MMRNode:
    def __init__(self, node_type: str, value: Any = None):
        self.node_type = node_type
        # Vectorized Topology: node_type is mapped to an L3 Embedding
        self.embedding = _get_basis_vector(node_type)
        self.value = value
        self.children: List['MMRNode'] = []

    def to_dict(self) -> Dict:
        return {
            "type": self.node_type,
            "embedding": self.embedding.tolist(),
            "value": self.value,
            "children": [c.to_dict() for c in self.children]
        }

class MMRTranslator:
    """Bridges Python AST and the language-independent MMR Manifold."""

    def __init__(self):
        # Build inverse mapping for reconstruction
        self._type_cache: Dict[str, np.ndarray] = {}

    def to_relational_graph(self, node: ast.AST) -> MMRNode:
        """Convert Python AST to Abstract MMR Graph."""
        mmr = MMRNode(node_type=type(node).__name__)
        
        if isinstance(node, ast.Name):
            mmr.value = node.id
        elif isinstance(node, ast.Constant):
            mmr.value = node.value
        elif isinstance(node, ast.arg):
            mmr.value = node.arg
        elif isinstance(node, ast.FunctionDef):
            mmr.value = node.name

        for child in ast.iter_child_nodes(node):
            mmr.children.append(self.to_relational_graph(child))
            
        return mmr

    def _match_type(self, embedding: np.ndarray) -> str:
        """Find the closest AST type by cosine similarity."""
        # For prototype, we use the known types in ast module
        best_type = "Pass"
        best_sim = -1.0
        
        # In a real v3 system, this would be a learned dictionary.
        # Here we scan common AST types.
        common_types = [
            "FunctionDef", "Name", "Constant", "arg", "Return", "Assign", 
            "For", "While", "If", "BinOp", "Call", "Attribute", "Load", "Store"
        ]
        
        for t in common_types:
            v = _get_basis_vector(t)
            sim = float(np.dot(embedding, v))
            if sim > best_sim:
                best_sim = sim
                best_type = t
        return best_type

    def from_relational_graph(self, mmr: MMRNode) -> ast.AST:
        """Synthesize Python AST from Vectorized MMR Graph."""
        node_type = self._match_type(mmr.embedding)
        node_class = getattr(ast, node_type, None)
        if not node_class:
            return ast.Pass()

        # Handle leaf nodes / values
        if node_type == "Name":
            return ast.Name(id=mmr.value or "var", ctx=ast.Load())
        elif node_type == "Constant":
            return ast.Constant(value=mmr.value)
        elif node_type == "arg":
            return ast.arg(arg=mmr.value or "arg")
        elif node_type == "Pass":
            return ast.Pass()

        try:
            children_ast = [self.from_relational_graph(c) for c in mmr.children]
            if node_type == "FunctionDef":
                return ast.FunctionDef(
                    name=mmr.value or "inferred_func",
                    args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                    body=children_ast if children_ast else [ast.Pass()],
                    decorator_list=[],
                    returns=None
                )
            # Default empty constructor for structural blocks
            return node_class()
        except Exception:
            return ast.Pass()
