"""MathAgent: Symbolic mathematics and rule discovery agent."""
from __future__ import annotations
import uuid
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from hfn.hfn import HFN
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.domains.math_domain import MathDomainConfig
from hpm_ai_v2.utils.oracle.math_oracle import MathOracle
from hpm_ai_v2.domains.math_renderer import MathRenderer

from hpm_ai_v2.utils.math_executor import MathExecutor

class MathAgent(BaseHFNAgent):
    """
    Agent specializing in symbolic math.
    Composes deterministic math primitives to solve equations and learn rules.
    """
    def __init__(self, config: MathDomainConfig, forest=None, **kwargs) -> None:
        if "renderer" not in kwargs:
            kwargs["renderer"] = MathRenderer(config)
        super().__init__(config, forest=forest, **kwargs)
        self.oracle = MathOracle(config)
        self.executor = MathExecutor(self)
        self._register_math_primitives()

    def _register_math_primitives(self) -> None:
        """Register L1 primitives for symbolic manipulation."""
        # Standard names for BFS/discovery
        self.add_strategy("differentiate", self.primitive_differentiate)
        self.add_strategy("integrate", self.primitive_integrate)
        self.add_strategy("simplify", self.primitive_simplify)
        self.add_strategy("solve", self.primitive_solve)
        
        # Manifold-aligned names (matched to DomainConfig concepts)
        self.add_strategy("trans_derivative", self.primitive_differentiate)
        self.add_strategy("trans_integral", self.primitive_integrate)
        self.add_strategy("trans_simplify", self.primitive_simplify)
        self.add_strategy("trans_solve", self.primitive_solve)
        
        # Tree Builders
        self.add_strategy("build_const", self.primitive_build_const)
        self.add_strategy("build_var", self.primitive_build_var)
        self.add_strategy("apply_op", self.primitive_apply_op)
        self.add_strategy("apply_func", self.primitive_apply_func)

    # --- Primitives (L1) ---

    def primitive_build_const(self, inputs: List[Any], outputs: List[Any]) -> Optional[List[HFN]]:
        """Primitive: Create a constant node."""
        if not inputs or not isinstance(inputs[0], (int, float)): return None
        val = inputs[0]
        node = self._ensure_const_node(val)
        return [node]

    def primitive_build_var(self, inputs: List[Any], outputs: List[Any]) -> Optional[List[HFN]]:
        """Primitive: Create a variable node."""
        if not inputs or not isinstance(inputs[0], str): return None
        name = inputs[0]
        node = self._ensure_var_node(name)
        return [node]

    def primitive_apply_op(self, inputs: List[Any], outputs: List[Any]) -> Optional[List[HFN]]:
        """Primitive: Apply binary operator (ADD, MUL, etc.) to two expression trees."""
        if len(inputs) < 3: return None
        op, left, right = inputs[0], inputs[1], inputs[2]
        if not isinstance(left, HFN) or not isinstance(right, HFN): return None
        
        res_node = self._build_expr_node(op, [left, right])
        return [res_node]

    def primitive_apply_func(self, inputs: List[Any], outputs: List[Any]) -> Optional[List[HFN]]:
        """Primitive: Apply unary function (SIN, COS, etc.) to an expression tree."""
        if len(inputs) < 2: return None
        func, arg = inputs[0], inputs[1]
        if not isinstance(arg, HFN): return None
        
        res_node = self._build_expr_node(func, [arg])
        return [res_node]

    def primitive_differentiate(self, inputs: List[Any], outputs: List[Any]) -> Optional[List[HFN]]:
        """Primitive: Compute derivative of expr w.r.t var."""
        if len(inputs) < 2: return None
        expr_node, var_node = inputs[0], inputs[1]
        if not isinstance(expr_node, HFN) or not isinstance(var_node, HFN): return None
        
        import sympy
        s_expr = self._to_sympy(expr_node)
        s_var = self._to_sympy(var_node)
        if not isinstance(s_var, sympy.Symbol): return None
        
        res_sympy = sympy.diff(s_expr, s_var)
        res_node = self._from_sympy(res_sympy)
        return [res_node]

    def primitive_integrate(self, inputs: List[Any], outputs: List[Any]) -> Optional[List[HFN]]:
        """Primitive: Indefinite integral of expr w.r.t var."""
        if len(inputs) < 2: return None
        expr_node, var_node = inputs[0], inputs[1]
        if not isinstance(expr_node, HFN) or not isinstance(var_node, HFN): return None
        
        import sympy
        s_expr = self._to_sympy(expr_node)
        s_var = self._to_sympy(var_node)
        
        res_sympy = sympy.integrate(s_expr, s_var)
        res_node = self._from_sympy(res_sympy)
        return [res_node]

    def primitive_simplify(self, inputs: List[Any], outputs: List[Any]) -> Optional[List[HFN]]:
        """Primitive: Algebraically simplify expression."""
        if not inputs or not isinstance(inputs[0], HFN): return None
        expr_node = inputs[0]
        
        import sympy
        s_expr = self._to_sympy(expr_node)
        res_sympy = sympy.simplify(s_expr)
        res_node = self._from_sympy(res_sympy)
        return [res_node]

    def primitive_solve(self, inputs: List[Any], outputs: List[Any]) -> Optional[List[HFN]]:
        """Primitive: Solve expr = 0 for var."""
        if len(inputs) < 2: return None
        expr_node, var_node = inputs[0], inputs[1]
        
        import sympy
        s_expr = self._to_sympy(expr_node)
        s_var = self._to_sympy(var_node)
        
        sols = sympy.solve(s_expr, s_var)
        if not sols: return None
        
        # Returns first solution as an HFN node
        res_node = self._from_sympy(sols[0])
        return [res_node]

    # --- HFN Tree Management ---

    def _ensure_const_node(self, val: float) -> HFN:
        node_id = f"math_const_{val}"
        node = self.forest.get(node_id)
        if not node:
            mu = np.zeros(self.m_dim)
            mu[0] = val / 100.0 # simple scaling
            node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.1, id=node_id, use_diag=True)
            node.relation_type = "math_constant"
            node.metadata = {"value": val, "sympy_expr": val}
            self.observer.register(node)
        return node

    def _ensure_var_node(self, name: str) -> HFN:
        node_id = f"math_var_{name}"
        node = self.forest.get(node_id)
        if not node:
            import sympy
            mu = np.zeros(self.m_dim)
            # Use hash of name for mu[0]
            mu[0] = (hash(name) % 1000) / 1000.0
            node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.1, id=node_id, use_diag=True)
            node.relation_type = "math_symbol"
            node.metadata = {"name": name, "sympy_expr": sympy.Symbol(name)}
            self.observer.register(node)
        return node

    def _build_expr_node(self, op: str, children: List[HFN]) -> HFN:
        """Create a composite expression node."""
        import sympy
        mu = np.mean([c.mu for c in children], axis=0)
        
        # Update root concept in mu
        mu[self.s_dim: self.s_dim + self.dim] = 0
        mapped_op = f"OP_{op.upper()}" if len(children) > 1 else f"FUNC_{op.upper()}"
        if mapped_op in self.config.concept_idx:
            mu[self.s_dim + self.config.concept_idx[mapped_op]] = 1.0
            
        node_id = f"math_expr_{uuid.uuid4().hex[:8]}"
        node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.1, id=node_id, use_diag=True)
        node.relation_type = "math_expr"
        for c in children: node.add_child(c)
        
        # Build sympy equivalent for the metadata
        s_args = [self._to_sympy(c) for c in children]
        if op.upper() == "ADD": s_expr = sum(s_args)
        elif op.upper() == "MUL": 
            s_expr = s_args[0]
            for a in s_args[1:]: s_expr *= a
        elif op.upper() == "POW": s_expr = s_args[0]**s_args[1]
        elif op.upper() == "SIN": s_expr = sympy.sin(s_args[0])
        elif op.upper() == "COS": s_expr = sympy.cos(s_args[0])
        else: s_expr = s_args[0] # fallback
        
        node.metadata = {"op": op, "sympy_expr": s_expr}
        self.observer.register(node)
        return node

    def _to_sympy(self, node: HFN):
        """Extract sympy expression from HFN metadata."""
        return node.metadata.get("sympy_expr")

    def _from_sympy(self, expr) -> HFN:
        """Convert a sympy expression into an HFN tree (Recursive)."""
        import sympy
        if expr.is_Symbol:
            return self._ensure_var_node(str(expr))
        if expr.is_Number:
            return self._ensure_const_node(float(expr))
            
        # Composite
        op_name = type(expr).__name__.upper()
        # Map some common sympy names to our ops
        if op_name == "ADD": op = "ADD"
        elif op_name == "MUL": op = "MUL"
        elif op_name == "POW": op = "POW"
        elif op_name == "SIN": op = "SIN"
        elif op_name == "COS": op = "COS"
        else: op = op_name
            
        children = [self._from_sympy(a) for a in expr.args]
        return self._build_expr_node(op, children)

    def parse(self, expr_str: str) -> HFN:
        """Parse string to HFN tree via sympy."""
        import sympy
        s_expr = sympy.simplify(expr_str)
        return self._from_sympy(s_expr)

    def detect_intent(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Detect if a query is a math request.
        Returns: (intent, expression_str) or None.
        """
        import re
        q = query.lower().strip()
        
        # Simple regex-based intent detection
        intents = {
            "differentiate": [r"derivative\s+of", r"differentiate", r"d/dx"],
            "integrate": [r"integral\s+of", r"integrate"],
            "simplify": [r"simplify"],
            "solve": [r"solve"]
        }
        
        for intent, patterns in intents.items():
            for p in patterns:
                # Use raw string and fix escape sequences
                pattern = rf"{p}\s*(.*)"
                match = re.search(pattern, q)
                if match:
                    expr_str = match.group(1).strip(' .?!')
                    # Basic validation: must contain mathy characters
                    if re.search(r"[\d\w\*\+\-\^/]", expr_str):
                        # print(f"      [MATH] Detected Intent: {intent} on '{expr_str}'")
                        return intent, expr_str
        return None

    def derive(self, intent: str, expr_str: str) -> Optional[HFN]:
        """Compute the answer for a math intent and return result node."""
        try:
            expr_node = self.parse(expr_str)
            # Use the registered primitives directly
            if intent == "differentiate":
                # Default to x for now
                var_node = self._ensure_var_node("x")
                res = self.primitive_differentiate([expr_node, var_node], [])
            elif intent == "integrate":
                var_node = self._ensure_var_node("x")
                res = self.primitive_integrate([expr_node, var_node], [])
            elif intent == "simplify":
                res = self.primitive_simplify([expr_node], [])
            elif intent == "solve":
                var_node = self._ensure_var_node("x")
                res = self.primitive_solve([expr_node, var_node], [])
            else:
                return None
                
            return res[0] if res else None
        except Exception:
            return None
