"""AST Encoders for HPM AI v1 Transpiler.

Maps Python AST structures into dense numerical vectors for the HPM stack.
"""
import ast
import numpy as np

# A mapping of common AST node types to vector indices for L1 Syntax Encoding
AST_NODE_TYPES = {
    ast.FunctionDef: 0,
    ast.ClassDef: 1,
    ast.Return: 2,
    ast.Assign: 3,
    ast.For: 4,
    ast.While: 5,
    ast.If: 6,
    ast.With: 7,
    ast.Raise: 8,
    ast.Try: 9,
    ast.Assert: 10,
    ast.Import: 11,
    ast.ImportFrom: 12,
    ast.Expr: 13,
    ast.Pass: 14,
    ast.Break: 15,
    ast.Continue: 16,
    ast.Call: 17,
    ast.Name: 18,
    ast.Constant: 19,
    ast.Attribute: 20,
    ast.BinOp: 21,
    ast.UnaryOp: 22,
    ast.Compare: 23,
    ast.Dict: 24,
    ast.List: 25,
    ast.Tuple: 26,
    ast.Set: 27,
    ast.ListComp: 28,
    ast.DictComp: 29,
    ast.GeneratorExp: 30,
    ast.Yield: 31
}

class ASTL1Encoder:
    """L1: Syntax (AST Node Distribution).
    Encodes the frequency of AST node types in a subtree.
    """
    feature_dim = 32

    def encode(self, node: ast.AST) -> np.ndarray:
        vec = np.zeros(self.feature_dim)
        for child in ast.walk(node):
            node_type = type(child)
            if node_type in AST_NODE_TYPES:
                vec[AST_NODE_TYPES[node_type]] += 1.0
        
        # Normalize to create a distribution profile
        total = vec.sum()
        if total > 0:
            vec /= total
        return vec

class ASTL2Encoder:
    """L2: Structural Anatomy.
    Encodes the shape of the function: number of args, depth, complexity.
    """
    feature_dim = 64

    def __init__(self):
        self.l1_enc = ASTL1Encoder()

    def _get_max_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        max_child_depth = current_depth
        for child in ast.iter_child_nodes(node):
            child_depth = self._get_max_depth(child, current_depth + 1)
            if child_depth > max_child_depth:
                max_child_depth = child_depth
        return max_child_depth

    def encode(self, node: ast.AST) -> np.ndarray:
        vec = np.zeros(self.feature_dim)
        
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            # 0: Number of arguments (or methods for class)
            if isinstance(node, ast.FunctionDef):
                vec[0] = len(node.args.args)
            else:
                vec[0] = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                
            # 1: Number of lines / body length
            vec[1] = len(node.body)
            # 2: Max AST depth (Cyclomatic complexity proxy)
            vec[2] = self._get_max_depth(node)
            # 3: Has return type annotation?
            vec[3] = 1.0 if hasattr(node, 'returns') and node.returns else 0.0
            # 4: Has docstring?
            vec[4] = 1.0 if ast.get_docstring(node) else 0.0
            
            # 5-36: Blend in L1 syntax signature (32-dim)
            l1_vec = self.l1_enc.encode(node)
            vec[5:37] = l1_vec
                
        return vec

class ASTL3Encoder:
    """L3: Relational Law (The Transformation Delta).
    Represents how an AST mutated from State A to State B.
    """
    feature_dim = 64

    def __init__(self):
        self.l2_enc = ASTL2Encoder()

    def encode(self, node_original: ast.AST, node_mutated: ast.AST) -> np.ndarray:
        """The 'Law' is the delta between the L2 anatomy vectors."""
        vec_a = self.l2_enc.encode(node_original)
        vec_b = self.l2_enc.encode(node_mutated)
        
        delta = vec_b - vec_a
        return delta # 64-dim
