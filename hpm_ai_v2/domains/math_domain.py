"""
MathDomainConfig — symbolic mathematics manifold configuration.
"""
from __future__ import annotations
import hashlib
import numpy as np
from typing import List, Dict, Optional, TYPE_CHECKING
from hpm_ai_v2.domains.base import DomainConfig

if TYPE_CHECKING:
    from hfn.hfn import HFN
    from hfn.forest import Forest

class MathDomainConfig(DomainConfig):
    """
    Manifold structure for symbolic mathematics.
    Represents expressions as HFN trees and provides structural fingerprinting.
    """
    def __init__(self, concepts: Optional[List[str]] = None, s_dim: int = 20):
        if concepts is None:
            concepts = [
                # Operators (Binary)
                "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_POW",
                # Functions (Unary)
                "FUNC_SIN", "FUNC_COS", "FUNC_TAN", "FUNC_LOG", "FUNC_EXP", "FUNC_SQRT",
                # Variables
                "VAR_X", "VAR_Y", "VAR_T",
                # Constants (Base)
                "CONST_E", "CONST_PI", "CONST_0", "CONST_1", "CONST_NEG1",
                # Transformations/Ops
                "TRANS_DERIVATIVE", "TRANS_INTEGRAL", "TRANS_SIMPLIFY", "TRANS_SOLVE",
                # Structural
                "MATH_EXPR", "MATH_EQUATION"
            ]
        super().__init__(concepts, s_dim=s_dim)

    @property
    def domain_type(self) -> str:
        return "math"

    def encode_expression(self, expr_str: str) -> np.ndarray:
        """
        Create a state vector (mu) for a math expression string.
        Uses structural features: hash, term count, depth.
        """
        import sympy
        try:
            expr = sympy.simplify(expr_str)
        except Exception:
            return np.zeros(self.m_dim)
            
        return self.encode_sympy(expr)

    def encode_sympy(self, expr) -> np.ndarray:
        """Encode a sympy expression into an HFN mu vector."""
        import sympy
        
        # 1. Structural Fingerprint
        # Simple hash of the structure (types of nodes in preorder)
        def get_struct_str(e):
            if e.is_Atom: return type(e).__name__
            return type(e).__name__ + "(" + ",".join(get_struct_str(a) for a in e.args) + ")"
            
        struct_hash = int(hashlib.md5(get_struct_str(expr).encode()).hexdigest(), 16) % 1000
        
        # 2. Numeric Features
        term_count = len(expr.free_symbols) + len(expr.atoms(sympy.Number))
        depth = 0
        def get_depth(e, d=0):
            if not e.args: return d
            return max(get_depth(a, d+1) for a in e.args)
        depth = get_depth(expr)
        
        # 3. Concept Vector (One-hot of the root operator)
        concept_vec = np.zeros(self.DIM)
        root_op = type(expr).__name__.upper()
        # Map sympy names to our concepts
        op_map = {
            "ADD": "OP_ADD", "MUL": "OP_MUL", "POW": "OP_POW",
            "SIN": "FUNC_SIN", "COS": "FUNC_COS", "LOG": "FUNC_LOG",
            "SYMBOL": "VAR_X", "INTEGER": "CONST_1", "FLOAT": "CONST_1",
            "DERIVATIVE": "TRANS_DERIVATIVE"
        }
        mapped_op = op_map.get(root_op, "MATH_EXPR")
        if mapped_op in self.concept_idx:
            concept_vec[self.concept_idx[mapped_op]] = 1.0

        # 4. Assemble Mu
        mu = np.zeros(self.m_dim)
        # S_DIM: [0: hash | 1: count | 2: depth | ...]
        mu[0] = struct_hash / 1000.0
        mu[1] = term_count / 10.0
        mu[2] = depth / 5.0
        
        mu[self.S_DIM: self.S_DIM + self.DIM] = concept_vec
        return mu

    def save_to_forest(self, forest: Forest) -> None:
        """Save math manifold to forest (Total Uniformity)."""
        super().save_to_forest(forest)
        print(f"      [CONFIG] Saved '{self.domain_type}' manifold to forest.")

    def reindex(self, forest: Forest, s_dim: int) -> None:
        """Ensure all math nodes are resized to the current forest dimension."""
        new_dim = forest._D
        # This is called during ReaderAgent.reindex_knowledge_base
        # The logic is already handled there by iterating over all active nodes.
        # We just need to update our internal dimension tracking if necessary.
        self.S_DIM = s_dim
        self.m_dim = new_dim
        self.DIM = new_dim - 2 * s_dim
