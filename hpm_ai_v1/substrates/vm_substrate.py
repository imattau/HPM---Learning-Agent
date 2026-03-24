"""Internal VM Substrate for HPM AI v3.1.

Implements Token-Native Execution: executes Middle-Manifold Representation (MMR)
graphs directly using Relational Vectors (embeddings).
Bypasses Python string checks for high-speed logic simulation.
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
        # Basis vectors are deterministic from the node name
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
            # Use the basis vector hash as the internal opcode
            basis = _get_basis_vector(name)
            self._ops[basis.tobytes()] = func

    def _match_op(self, embedding: np.ndarray) -> Optional[Callable]:
        """Fast opcode resolution via vector similarity (Geometric Dialect)."""
        # For performance, we pre-calculate basis vector keys.
        # In a real v3 system, this uses a KV-store or ANN search.
        best_func = None
        best_sim = -1.0
        
        # Check against all known internal op basis vectors
        for basis_bytes, func in self._ops.items():
            basis = np.frombuffer(basis_bytes, dtype=np.float64)
            sim = float(np.dot(embedding, basis))
            if sim > best_sim:
                best_sim = sim
                best_func = func
        
        return best_func if best_sim > 0.9 else None

    def execute(self, node: MMRNode, inputs: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Token-Native Execution Loop.
        Operates directly on the relational manifold coordinates.
        """
        if context is None:
            context = dict(inputs)

        # 1. Resolve Operator Intent (Geometric Matching)
        op_func = self._match_op(node.embedding)
        
        # 2. Handle Terminals (Leaves)
        if node.node_type == "Constant":
            return node.value
        if node.node_type == "Name" or node.node_type == "arg":
            return context.get(node.value)

        # 3. Recursive Evaluation of Child Manifolds
        child_results = []
        for child in node.children:
            res = self.execute(child, inputs, context)
            child_results.append(res)

        # 4. Apply Relational Logic
        if op_func:
            result = op_func(child_results)
            # Special case: Assignment updates the local context
            if node.node_type == "Assign" and len(node.children) > 0:
                target_node = node.children[0]
                if target_node.node_type == "Name":
                    context[target_node.value] = result
            return result
            
        return child_results[-1] if child_results else None

    def verify_logic_speed(self, mmr_graph: MMRNode, iterations: int = 1000) -> float:
        """Measures execution speed in the manifold (Dialect Sovereignty metric)."""
        import time
        start = time.time()
        for i in range(iterations):
            self.execute(mmr_graph, {"x": i, "y": i+1})
        end = time.time()
        return iterations / (end - start)
