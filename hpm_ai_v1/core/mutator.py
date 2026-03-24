"""Code Mutation Actor for HPM AI v1.

Utilizes the L4 Generative Head to propose specific code changes 
based on recognized L3 relational patterns and crossovers.
"""
import ast
from typing import List, Optional, Tuple
import numpy as np
from hpm_ai_v1.transpiler.decoder import StructuralTranspiler
from hpm.agents.l4_generative import L4GenerativeHead

class CodeMutationActor:
    def __init__(self, l2_dim: int = 16, l3_dim: int = 32):
        self.transpiler = StructuralTranspiler()
        # The L4 Head intuits the L3 transformation from L2 structural anatomy
        self.l4_head = L4GenerativeHead(feature_dim_in=l2_dim, feature_dim_out=l3_dim)

    def propose_mutation(
        self, 
        original_source: str, 
        l2_input: np.ndarray, 
        l3_population: List[ast.AST] = []
    ) -> Optional[str]:
        """
        Uses L4 intuition to predict the target law and performs 
        Relational Recombination using the provided population.
        Returns the raw source of the new generation.
        """
        # 1. Intuitive Leap: L2 (Anatomy) -> L3 (Law)
        target_l3_law = self.l4_head.predict(l2_input)
        if target_l3_law is None:
            # Fallback to zero-delta law (identity) if no intuition yet
            target_l3_law = np.zeros(32)

        # 2. Structural Transpilation: Law + Recombination -> Source
        try:
            tree = ast.parse(original_source)
        except SyntaxError:
            return None

        # Focus on the first function for mutation
        target_node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None)
        if not target_node:
            return None
            
        new_source = self.transpiler.transpile(target_node, target_l3_law, l3_population)
        return new_source
