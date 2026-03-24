"""Code Mutation Actor for HPM AI v2.3.

Utilizes the L4 Generative Head to propose specific code changes 
based on recognized L3 relational patterns and crossovers.
Now supports Multi-File Change-Sets for Cascading Repair.
"""
import ast
import os
from typing import List, Optional, Tuple, Dict
import numpy as np
from hpm_ai_v1.transpiler.decoder import StructuralTranspiler
from hpm_ai_v1.transpiler.mmr import ProjectTopology
from hpm.agents.l4_generative import L4GenerativeHead

class ChangeSet:
    """A collection of mutations across multiple files."""
    def __init__(self):
        self.mutations: Dict[str, str] = {} # filepath -> new_source

    def add(self, filepath: str, source: str):
        self.mutations[filepath] = source

class CodeMutationActor:
    def __init__(self, l2_dim: int = 16, l3_dim: int = 32):
        self.transpiler = StructuralTranspiler()
        self.l4_head = L4GenerativeHead(feature_dim_in=l2_dim, feature_dim_out=l3_dim)

    def propose_cascading_mutation(
        self, 
        repo_path: str,
        target_file: str,
        topology: ProjectTopology,
        l2_input: np.ndarray, 
        l3_population: List[ast.AST] = []
    ) -> ChangeSet:
        """
        Proposes a mutation for the target file and recursively repairs 
        impacted dependencies in other files.
        """
        changeset = ChangeSet()
        
        # 1. Primary Mutation
        full_target_path = os.path.join(repo_path, target_file)
        with open(full_target_path, 'r') as f:
            original_source = f.read()
            
        # Prediction and Transpilation
        target_l3_law = self.l4_head.predict(l2_input) or np.zeros(32)
        tree = ast.parse(original_source)
        target_node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None)
        
        if not target_node:
            return changeset
            
        new_source = self.transpiler.transpile(target_node, target_l3_law, l3_population)
        if not new_source or new_source == original_source:
            return changeset
            
        changeset.add(target_file, new_source)
        
        # 2. Cascading Repair
        # Check if we renamed the function or changed signature
        new_tree = ast.parse(new_source)
        new_func = next((n for n in ast.walk(new_tree) if isinstance(n, ast.FunctionDef)), None)
        
        if new_func and new_func.name != target_node.name:
            # Function was renamed! Repair all call-sites.
            impacted_files = topology.get_impacted_files(target_node.name)
            for imp_path in impacted_files:
                if imp_path == full_target_path: continue
                
                # Load dependent file
                with open(imp_path, 'r') as f:
                    imp_source = f.read()
                
                # Perform simple name replacement in the dependent AST
                imp_tree = ast.parse(imp_source)
                class Renamer(ast.NodeTransformer):
                    def visit_Name(self, node):
                        if node.id == target_node.name:
                            return ast.Name(id=new_func.name, ctx=node.ctx)
                        return self.generic_visit(node)
                    def visit_Attribute(self, node):
                        if node.attr == target_node.name:
                            return ast.Attribute(value=node.value, attr=new_func.name, ctx=node.ctx)
                        return self.generic_visit(node)
                
                repaired_tree = Renamer().visit(imp_tree)
                # Note: path returned by topology is full, we need relative for the changeset if preferred
                rel_imp_path = os.path.relpath(imp_path, repo_path)
                changeset.add(rel_imp_path, ast.unparse(repaired_tree))
                
        return changeset
