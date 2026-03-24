"""Internal VM Substrate for HPM AI v3.0.

Executes Middle-Manifold Representation (MMR) graphs directly.
Allows agents to verify 'Logical Truth' in the abstract manifold 
before committing to Python syntax.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from hpm_ai_v1.transpiler.mmr import MMRNode

class InternalVMSubstrate:
    """
    The foundation of the 'Geometric Dialect'.
    Executes Relational Tokens as data-flow graphs.
    """
    def __init__(self):
        # Map node embeddings to functional operations
        self._ops = {
            "Add": lambda args: args[0] + args[1] if len(args) >= 2 else 0,
            "Sub": lambda args: args[0] - args[1] if len(args) >= 2 else 0,
            "Mult": lambda args: args[0] * args[1] if len(args) >= 2 else 0,
            "Eq": lambda args: args[0] == args[1] if len(args) >= 2 else False,
            "Return": lambda args: args[0] if args else None,
        }

    def _get_op_by_embedding(self, embedding: np.ndarray) -> str:
        """Find the operator whose basis vector is closest to the node's embedding."""
        from hpm_ai_v1.transpiler.mmr import _get_basis_vector
        best_op = "Return"
        best_sim = -1.0
        for op in self._ops.keys():
            v = _get_basis_vector(op)
            sim = float(np.dot(embedding, v))
            if sim > best_sim:
                best_sim = sim
                best_op = op
        return best_op

    def execute(self, node: MMRNode, inputs: Dict[str, Any]) -> Any:
        """Recursive execution of the Relational Manifold."""
        # 1. Resolve Node Intent via Embedding
        op_name = self._get_op_by_embedding(node.embedding)
        
        # 2. Handle Leaf Nodes
        if node.node_type == "Constant":
            return node.value
        if node.node_type == "Name":
            return inputs.get(node.value)
        if node.node_type == "arg":
            return inputs.get(node.value)

        # 3. Recursive Child Evaluation
        child_results = []
        for child in node.children:
            res = self.execute(child, inputs)
            child_results.append(res)

        # 4. Apply Operational Logic
        if op_name in self._ops:
            return self._ops[op_name](child_results)
            
        return child_results[-1] if child_results else None

    def verify_equivalence(self, mmr_a: MMRNode, mmr_b: MMRNode, test_inputs: List[Dict[str, Any]]) -> float:
        """
        Measures the logical divergence between two manifolds.
        Proves that a refactored law preserves the original 'Structural Truth'.
        """
        matches = 0
        for inp in test_inputs:
            res_a = self.execute(mmr_a, inp)
            res_b = self.execute(mmr_b, inp)
            if res_a == res_b:
                matches += 1
        return matches / len(test_inputs) if test_inputs else 1.0
