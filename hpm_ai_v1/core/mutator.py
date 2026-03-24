"""Code Mutation Actor for HPM AI v1.

Utilizes the L4 Generative Head to propose specific code changes (diffs) 
based on recognized L3 relational patterns.
"""
import difflib
import ast
import numpy as np
from hpm_ai_v1.transpiler.decoder import StructuralTranspiler
from hpm.agents.l4_generative import L4GenerativeHead

class CodeMutationActor:
    def __init__(self, l2_dim: int = 16, l3_dim: int = 32):
        self.transpiler = StructuralTranspiler()
        # The L4 Head intuits the L3 transformation from L2 structural anatomy
        self.l4_head = L4GenerativeHead(feature_dim_in=l2_dim, feature_dim_out=l3_dim)

    def generate_diff(self, original_source: str, new_source: str, filepath: str) -> str:
        """Generate a standard Unified Code Diff."""
        orig_lines = original_source.splitlines(keepends=True)
        new_lines = new_source.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            orig_lines, new_lines, 
            fromfile=filepath, 
            tofile=filepath,
            n=3
        )
        return "".join(diff)

    def propose_mutation(self, original_source: str, filepath: str, l2_input: np.ndarray) -> str:
        """
        Uses L4 intuition to predict the target law, transpires it, 
        and returns the unified diff patch.
        """
        # 1. Intuitive Leap: L2 (Anatomy) -> L3 (Law)
        target_l3_law = self.l4_head.predict(l2_input)
        if target_l3_law is None:
            # Fallback if L4 hasn't learned yet: use identity
            target_l3_law = np.zeros(32)

        # 2. Structural Transpilation: Law -> AST
        try:
            tree = ast.parse(original_source)
        except SyntaxError:
            return ""

        target_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                target_node = node
                break
                
        if not target_node:
            return ""
            
        new_source = self.transpiler.transpile(target_node, target_l3_law)
        return self.generate_diff(original_source, new_source, filepath)
