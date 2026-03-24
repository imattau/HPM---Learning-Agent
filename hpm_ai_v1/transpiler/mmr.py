"""Middle-Manifold Representation (MMR) for HPM AI v2.2.

Decouples Relational Logic from Python Syntax by representing code as 
abstract graphs of functional primitives.
Now supports Project-Level Manifolds (The 'Global Brain').
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
        self.dependencies: List[str] = [] # Project-level Call Edges / Imports

    def to_dict(self) -> Dict:
        return {
            "type": self.node_type,
            "embedding": self.embedding.tolist(),
            "value": self.value,
            "children": [c.to_dict() for c in self.children],
            "dependencies": self.dependencies
        }

class ProjectManifold:
    """The 'Global Brain' Topology.
    Maps inter-module dependencies across the entire codebase.
    """
    def __init__(self):
        self.modules: Dict[str, MMRNode] = {}
        
    def add_module(self, filepath: str, root_node: MMRNode):
        self.modules[filepath] = root_node
        
    def check_structural_immunity(self, changed_filepath: str, new_node: MMRNode) -> bool:
        """
        Detects if a local mutation breaks the global topology.
        e.g., changing a function signature used by other modules.
        """
        # For prototype: simply checks if the 'value' (e.g. func name) was deleted.
        if changed_filepath not in self.modules:
            return True
            
        old_root = self.modules[changed_filepath]
        old_exports = [c.value for c in old_root.children if c.node_type in ("FunctionDef", "ClassDef")]
        new_exports = [c.value for c in new_node.children if c.node_type in ("FunctionDef", "ClassDef")]
        
        # If an export is dropped, check if others depend on it
        dropped = set(old_exports) - set(new_exports)
        if dropped:
            # Check all other modules for calls to the dropped exports
            for path, mod in self.modules.items():
                if path == changed_filepath: continue
                # Simple check: does the dependency list contain the dropped name?
                # (In a full system, this would trace the exact CallEdge)
                all_deps = self._get_all_deps(mod)
                if any(d in all_deps for d in dropped):
                    print(f"Global Contradiction: Mutation breaks dependencies in {path}")
                    return False
        return True

    def _get_all_deps(self, node: MMRNode) -> List[str]:
        deps = list(node.dependencies)
        for c in node.children:
            deps.extend(self._get_all_deps(c))
        return deps

class MMRTranslator:
    """Bridges Python AST and the language-independent MMR Manifold."""

    def __init__(self):
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
        elif isinstance(node, ast.ClassDef):
            mmr.value = node.name
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                mmr.dependencies.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                mmr.dependencies.append(node.func.attr)

        for child in ast.iter_child_nodes(node):
            mmr.children.append(self.to_relational_graph(child))
            
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
            return ast.Pass()

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
            elif node_type == "Module":
                return ast.Module(body=children_ast, type_ignores=[])
            return node_class()
        except Exception:
            return ast.Pass()
