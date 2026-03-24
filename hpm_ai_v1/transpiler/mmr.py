"""Middle-Manifold Representation (MMR) for HPM AI v2.2.

Decouples Relational Logic from Python Syntax by representing code as 
abstract graphs of functional primitives.
Now supports Project-Level Manifolds (The 'Global Brain').
"""
import ast
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple

# Shared padded dimension for the MMR Manifold
MMR_DIM = 32

def _get_basis_vector(name: str) -> np.ndarray:
    """Generate a stable, deterministic unit vector for a given node type."""
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    v = rng.standard_normal(MMR_DIM)
    return v / (np.linalg.norm(v) + 1e-9)

class MMRNode:
    def __init__(self, node_type: str, value: Any = None, filepath: str = ""):
        self.node_type = node_type
        self.embedding = _get_basis_vector(node_type)
        self.value = value
        self.filepath = filepath
        self.children: List['MMRNode'] = []
        self.dependencies: List[str] = [] # Names this node calls
        self.lineno: int = 0

    def to_dict(self) -> Dict:
        return {
            "type": self.node_type,
            "embedding": self.embedding.tolist(),
            "value": self.value,
            "filepath": self.filepath,
            "children": [c.to_dict() for c in self.children],
            "dependencies": self.dependencies
        }

class ProjectTopology:
    """Maps the 'Relational Ecology' of the entire project."""
    def __init__(self):
        self.exports: Dict[str, str] = {}
        self.in_edges: Dict[str, List[Tuple[str, MMRNode]]] = {}
        self.modules: Dict[str, MMRNode] = {}

    def add_module(self, filepath: str, root_node: MMRNode):
        self.modules[filepath] = root_node
        self._index_node(filepath, root_node)

    def _index_node(self, filepath: str, node: MMRNode):
        if node.node_type in ("FunctionDef", "ClassDef") and node.value:
            self.exports[node.value] = filepath
        
        for dep in node.dependencies:
            if dep not in self.in_edges:
                self.in_edges[dep] = []
            self.in_edges[dep].append((filepath, node))
            
        for child in node.children:
            self._index_node(filepath, child)

    def get_impacted_files(self, changed_name: str) -> Set[str]:
        """Find all files that call the changed name."""
        return {filepath for filepath, _ in self.in_edges.get(changed_name, [])}

class MMRTranslator:
    """Bridges Python AST and the language-independent MMR Manifold."""

    def to_relational_graph(self, node: ast.AST, filepath: str = "") -> MMRNode:
        mmr = MMRNode(node_type=type(node).__name__, filepath=filepath)
        mmr.lineno = getattr(node, 'lineno', 0)
        
        if isinstance(node, ast.Name):
            mmr.value = node.id
        elif isinstance(node, ast.Constant):
            mmr.value = node.value
        elif isinstance(node, ast.arg):
            mmr.value = node.arg
        elif isinstance(node, ast.FunctionDef):
            mmr.value = node.name
        elif isinstance(node, ast.ClassDef):
            mmr.value = node.name
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                mmr.dependencies.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                mmr.dependencies.append(node.func.attr)

        for child in ast.iter_child_nodes(node):
            mmr.children.append(self.to_relational_graph(child, filepath))
            
        return mmr

    def _match_type(self, embedding: np.ndarray) -> str:
        best_type = "Pass"
        best_sim = -1.0
        common_types = [
            "FunctionDef", "ClassDef", "Name", "Constant", "arg", "Return", "Assign", 
            "For", "While", "If", "BinOp", "Call", "Attribute", "Load", "Store", "Module"
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
            return ast.Pass(lineno=1, col_offset=0)

        # Handle leaf nodes / values
        node = None
        if node_type == "Name":
            node = ast.Name(id=mmr.value or "var", ctx=ast.Load())
        elif node_type == "Constant":
            node = ast.Constant(value=mmr.value)
        elif node_type == "arg":
            node = ast.arg(arg=mmr.value or "arg")
        elif node_type == "Pass":
            node = ast.Pass()

        if node:
            node.lineno = 1
            node.col_offset = 0
            return node

        try:
            children_ast = [self.from_relational_graph(c) for c in mmr.children]
            if node_type == "FunctionDef":
                node = ast.FunctionDef(
                    name=mmr.value or "inferred_func",
                    args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                    body=children_ast if children_ast else [ast.Pass(lineno=1, col_offset=0)],
                    decorator_list=[],
                    returns=None
                )
            elif node_type == "Module":
                node = ast.Module(body=children_ast, type_ignores=[])
            else:
                node = node_class()
            
            node.lineno = 1
            node.col_offset = 0
            return node
        except Exception:
            return ast.Pass(lineno=1, col_offset=0)
