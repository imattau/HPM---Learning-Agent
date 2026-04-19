"""MathRenderer: Converts HFN math trees back to strings/LaTeX."""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hfn.hfn import HFN
from hpm_ai_v2.domains.math_domain import MathDomainConfig

class MathRenderer:
    """Recursively renders HFN math trees into mathematical expressions."""
    def __init__(self, config: MathDomainConfig):
        self.config = config

    def render(self, node: "HFN") -> str:
        metadata = getattr(node, "metadata", {})
        relation_type = getattr(node, "relation_type", None)
        
        # 0. Macro Rendering (Sequence of transformations)
        if relation_type == "macro":
            steps = node.inputs or []
            if not steps: return "res = inp"
            # Render as a chain: res = stepN(stepN-1(...step1(inp)...))
            code = "inp"
            for step in steps:
                step_code = self.render(step)
                # If step_code is a numeric string or doesn't look like a function/transformation,
                # we can't chain it as a function. 
                try:
                    float(step_code)
                    continue # Skip constants in transformation chains
                except ValueError:
                    pass

                # Special case: if it's a variable or something else non-callable
                if step.relation_type in ["math_symbol", "math_constant"]:
                    continue

                # If step is a transformation primitive name, wrap it
                if "(" not in step_code:
                    code = f"{step_code}({code})"
                else:
                    # Simple substitution if it's a template
                    code = step_code.replace("inp", code)
            return f"res = {code}"


        # 1. Base Case: Literal Symbols or Constants
        if relation_type == "math_symbol":
            return metadata.get("name", node.id.split("_")[-1])
        if relation_type == "math_constant":
            return str(metadata.get("value", node.id.split("_")[-1]))

        # 2. Recursive Rendering
        children = node.children()
        
        if relation_type == "math_expr":
            op = metadata.get("op", "").upper()
            
            if not children:
                return metadata.get("text", node.id)
                
            if op == "ADD":
                return "(" + " + ".join(self.render(c) for c in children) + ")"
            if op == "MUL":
                return "(" + " * ".join(self.render(c) for c in children) + ")"
            if op == "POW":
                if len(children) >= 2:
                    return f"{self.render(children[0])}**{self.render(children[1])}"
                return self.render(children[0])
            if op in ["SIN", "COS", "LOG", "EXP", "SQRT"]:
                return f"{op.lower()}({self.render(children[0])})"
            if op == "DERIVATIVE":
                if len(children) >= 2:
                    return f"d/d{self.render(children[1])} ({self.render(children[0])})"
                return f"derivative({self.render(children[0])})"
                
            # Default fallback for unknown ops
            return f"{op.lower()}(" + ",".join(self.render(c) for c in children) + ")"

        # 3. Handle Primitives (Strategies)
        # If it's a prior rule or a strategy node, return its name or code
        if relation_type == "primitive" or node.id.startswith("prior_rule_"):
            return metadata.get("code", node.id.replace("prior_rule_", "").lower())

        # 4. Fallback to ID or Top Concept
        return metadata.get("text", node.id)
