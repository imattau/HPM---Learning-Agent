"""MathExecutor: Executes symbolic math code using sympy and MathAgent primitives."""
from __future__ import annotations
from typing import Any, List, Optional, Tuple, TYPE_CHECKING
import sympy
from hpm_ai_v2.utils.executor import PythonExecutor

if TYPE_CHECKING:
    from hpm_ai_v2.agents.math_agent import MathAgent

class MathExecutor(PythonExecutor):
    """
    Extends PythonExecutor to support symbolic math operations.
    Wraps MathAgent's primitives as functions in the execution scope.
    """
    def __init__(self, agent: "MathAgent"):
        self.agent = agent

    def run_batch(
        self,
        code_str: str,
        inputs: List[Any],
        timeout: float = 0.5,
    ) -> Tuple[List[Any], List[Optional[str]]]:
        """Inject math primitives into the execution scope."""
        # Define local scope with math-friendly functions
        # Most primitives in MathAgent take (inputs: List, outputs: List)
        # We wrap them to take a single 'inp' (HFN node) and return a single 'HFN node'
        def differentiate(e):
            # Assumes 'inputs' in run_batch contains [expr, var] or similar
            # For simplicity in macros, we use 'x' as default variable
            var_node = self.agent._ensure_var_node("x")
            res = self.agent.primitive_differentiate([e, var_node], [])
            return res[0] if res else None

        def integrate(e):
            var_node = self.agent._ensure_var_node("x")
            res = self.agent.primitive_integrate([e, var_node], [])
            return res[0] if res else None

        def simplify(e):
            res = self.agent.primitive_simplify([e], [])
            return res[0] if res else None

        # Build local scope
        math_scope = {
            "differentiate": differentiate,
            "integrate": integrate,
            "simplify": simplify,
            "sin": lambda e: self.agent._build_expr_node("SIN", [e]),
            "cos": lambda e: self.agent._build_expr_node("COS", [e]),
            "sympy": sympy,
            "HFN": self.agent.forest.get # helper
        }
        
        # Now run via standard PythonExecutor logic, but with our scope
        # We'll override run_batch to use this scope
        indented = code_str.replace('\n', '\n    ')
        code = (
            "def test_func(inp, inputs):\n"
            "    res = None\n"
            "    " + indented + "\n"
            "    return res\n"
        )
        
        results: List[Any] = []
        errors: List[Optional[str]] = []
        try:
            local_ns = math_scope.copy()
            compile_obj = compile(code, "<math_executor>", "exec")
            exec(compile_obj, {}, local_ns)
            test_func = local_ns["test_func"]
        except Exception as e:
            return [None] * len(inputs), [type(e).__name__] * len(inputs)
            
        for inp in inputs:
            try:
                # 'inp' is an HFN node representing the expression
                results.append(test_func(inp, inputs))
                errors.append(None)
            except Exception as e:
                results.append(None)
                errors.append(type(e).__name__)
        return results, errors
