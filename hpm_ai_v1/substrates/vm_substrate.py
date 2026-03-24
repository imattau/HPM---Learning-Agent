"""Internal VM Substrate for HPM AI v2.2.

Executes the Middle-Manifold Representation (MMR) graphs directly, 
bypassing Python. This is the foundation of the Dialect Genesis (SP22).
"""
import numpy as np
from typing import Dict, Any, List
from hpm_ai_v1.transpiler.mmr import MMRNode

class InternalVMSubstrate:
    """
    Executes Relational Tokens directly without translating back to Python.
    Treats the MMR graph as an abstract data flow machine.
    """
    def __init__(self):
        self.memory: Dict[str, Any] = {}
        # Map node embedding similarity to internal op codes
        self._op_registry = {
            "Add": lambda a, b: a + b,
            "Sub": lambda a, b: a - b,
            "Mult": lambda a, b: a * b,
            "Div": lambda a, b: a / b if b != 0 else 0,
            "Eq": lambda a, b: a == b,
            "Gt": lambda a, b: a > b,
            "Lt": lambda a, b: a < b
        }

    def _execute_node(self, node: MMRNode, context: Dict[str, Any]) -> Any:
        # Match node type using embedding (mocked using string type for prototype)
        # In a full v3 implementation, this would cluster embeddings into operations
        if node.node_type == "Constant":
            return node.value
        elif node.node_type == "Name":
            return context.get(node.value, self.memory.get(node.value, None))
        elif node.node_type == "BinOp":
            if len(node.children) >= 2:
                left = self._execute_node(node.children[0], context)
                right = self._execute_node(node.children[1], context)
                op_node = node.children[2] if len(node.children) > 2 else None
                op_name = op_node.node_type if op_node else "Add" # Default
                
                # Execute the relational op
                op_func = self._op_registry.get(op_name, self._op_registry["Add"])
                return op_func(left, right)
        elif node.node_type == "Return":
            if node.children:
                return self._execute_node(node.children[0], context)
        elif node.node_type == "Assign":
            if len(node.children) >= 2:
                target = node.children[0]
                value = self._execute_node(node.children[1], context)
                if target.node_type == "Name":
                    context[target.value] = value
                return value
        elif node.node_type == "FunctionDef":
            # Just execute the body for now
            result = None
            for child in node.children:
                result = self._execute_node(child, context)
            return result
        elif node.node_type == "Module":
            result = None
            for child in node.children:
                result = self._execute_node(child, context)
            return result
            
        # Fallback: traverse children
        result = None
        for child in node.children:
            result = self._execute_node(child, context)
        return result

    def execute(self, mmr_graph: MMRNode, inputs: Dict[str, Any]) -> Any:
        """
        Executes a Relational Manifold (MMR Graph) with the given inputs.
        This represents the agent 'thinking' directly in its own logic space.
        """
        context = dict(inputs)
        return self._execute_node(mmr_graph, context)
