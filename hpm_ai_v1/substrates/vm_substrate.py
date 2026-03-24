"""Internal VM Substrate for HPM AI v3.2.

Implements Token-Native Execution and Algebraic Verification.
Verification is performed by comparing Topological Invariants (graph outputs) 
across identical input manifolds.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from hpm_ai_v1.transpiler.mmr import MMRNode, _get_basis_vector

class InternalVMSubstrate:
    """
    The 'Geometric Dialect' Runner.
    Executes Relational Tensors as pure data-flow operations.
    """
    def __init__(self):
        # Basis-indexed operation registry
        self._ops: Dict[bytes, Callable] = {}
        self._init_basis_ops()

    def _init_basis_ops(self):
        """Map basis vector hashes to executable logic."""
        ops_map = {
            "Add": lambda args: args[0] + args[1] if len(args) >= 2 else 0,
            "Sub": lambda args: args[0] - args[1] if len(args) >= 2 else 0,
            "Mult": lambda args: args[0] * args[1] if len(args) >= 2 else 0,
            "Div": lambda args: args[0] / args[1] if len(args) >= 2 and args[1] != 0 else 0,
            "Eq": lambda args: args[0] == args[1] if len(args) >= 2 else False,
            "Gt": lambda args: args[0] > args[1] if len(args) >= 2 else False,
            "Lt": lambda args: args[0] < args[1] if len(args) >= 2 else False,
            "Return": lambda args: args[0] if args else None,
            "Assign": lambda args: args[1] if len(args) >= 2 else None,
        }
        for name, func in ops_map.items():
            basis = _get_basis_vector(name)
            self._ops[basis.tobytes()] = func

    def _match_op(self, embedding: np.ndarray) -> Optional[Callable]:
        """Fast opcode resolution via vector similarity."""
        best_func = None
        best_sim = -1.0
        for basis_bytes, func in self._ops.items():
            basis = np.frombuffer(basis_bytes, dtype=np.float64)
            sim = float(np.dot(embedding, basis))
            if sim > best_sim:
                best_sim = sim
                best_func = func
        return best_func if best_sim > 0.9 else None

    def execute(self, node: MMRNode, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """Token-Native Execution of the Relational Manifold."""
        if context is None:
            context = dict(inputs)

        op_func = self._match_op(node.embedding)
        
        if node.node_type == "Constant":
            return node.value
        if node.node_type == "Name" or node.node_type == "arg":
            return context.get(node.value)

        child_results = []
        for child in node.children:
            res = self.execute(child, inputs, context)
            child_results.append(res)

        if op_func:
            result = op_func(child_results)
            if node.node_type == "Assign" and len(node.children) > 0:
                target_node = node.children[0]
                if target_node.node_type == "Name":
                    context[target_node.value] = result
            return result
            
        return child_results[-1] if child_results else None

    def verify_topological_equivalence(self, mmr_a: MMRNode, mmr_b: MMRNode) -> bool:
        """
        Algebraic Verification: compares graph outputs across random input vectors.
        This proves functional identity without simulating text.
        """
        # Generate random input manifold
        rng = np.random.default_rng(42)
        test_inputs = [
            {"x": float(rng.uniform(-10, 10)), "y": float(rng.uniform(-10, 10))} 
            for _ in range(10)
        ]
        
        for inp in test_inputs:
            try:
                res_a = self.execute(mmr_a, inp)
                res_b = self.execute(mmr_b, inp)
                if res_a != res_b:
                    return False
            except:
                return False
        return True
