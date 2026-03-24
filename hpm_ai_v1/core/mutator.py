"""L4 Architect Agent for HPM AI v3.0.

Standalone agent specializing in generating coherent ChangeSets via 
Relational Recombination and Cascading Repair.
"""
import ast
import os
from typing import List, Optional, Tuple, Dict
import numpy as np

from hpm.agents.agent import Agent
from hpm.config import AgentConfig
from hpm_ai_v1.transpiler.decoder import StructuralTranspiler
from hpm_ai_v1.transpiler.mmr import ProjectTopology
from hpm.agents.l4_generative import L4GenerativeHead

class ChangeSet:
    """A collection of mutations across multiple files."""
    def __init__(self):
        self.mutations: Dict[str, str] = {} # filepath -> new_source

    def add(self, filepath: str, source: str):
        self.mutations[filepath] = source

class L4ArchitectAgent(Agent):
    def __init__(self, config: AgentConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.transpiler = StructuralTranspiler()
        self.l4_head = L4GenerativeHead(feature_dim_in=config.feature_dim, feature_dim_out=32)

    def propose_cascading_mutation(
        self, 
        repo_path: str,
        target_file: str,
        topology: ProjectTopology,
        l2_input: np.ndarray, 
        l3_population: List[ast.AST] = []
    ) -> ChangeSet:
        """
        Synthesizes a ChangeSet that addresses the target law and repairs 
        all detected global contradictions in the project manifold.
        """
        changeset = ChangeSet()
        
        full_target_path = os.path.join(repo_path, target_file)
        if not os.path.exists(full_target_path):
            return changeset
            
        with open(full_target_path, 'r') as f:
            original_source = f.read()
            
        # 1. Primary Intuition: What is the target L3 law for this function?
        target_l3_law = self.l4_head.predict(l2_input) or np.zeros(32)
        
        # 2. Semantic Invention: MMR Crossover
        tree = ast.parse(original_source)
        target_node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None)
        
        if not target_node:
            return changeset
            
        new_source = self.transpiler.transpile(target_node, target_l3_law, l3_population)
        if not new_source or new_source == original_source:
            return changeset
            
        changeset.add(target_file, new_source)
        
        # 3. Manifold-Based Cascading Repair
        new_tree = ast.parse(new_source)
        new_func = next((n for n in ast.walk(new_tree) if isinstance(n, ast.FunctionDef)), None)
        
        if new_func and new_func.name != target_node.name:
            print(f"L4 Architect: Detecting Structural Shift in '{target_node.name}' -> '{new_func.name}'")
            impacted_files = topology.get_impacted_files(target_node.name)
            for imp_path in impacted_files:
                if os.path.abspath(imp_path) == os.path.abspath(full_target_path): continue
                
                print(f"L4 Architect: Repairing dependent manifold in {imp_path}")
                with open(imp_path, 'r') as f:
                    imp_source = f.read()
                
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
                rel_imp_path = os.path.relpath(imp_path, repo_path)
                changeset.add(rel_imp_path, ast.unparse(repaired_tree))
                
        return changeset
