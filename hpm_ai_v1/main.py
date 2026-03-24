"""HPM AI v2.1: Succession Controller.

Implements Geometric Relational Intelligence via Vectorized MMR, Metacognitive 
Exploration (Bloat Windows), and Autonomous Benchmark Expansion.
"""
import os
import ast
import argparse
import time
import numpy as np
from typing import List

# Core HPM Framework
from hpm.substrate.math import MathSubstrate
from hpm.substrate.pypi import PyPISubstrate
from hpm.agents.agent import Agent
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.config import AgentConfig
from hpm.field.field import PatternField
from hpm.patterns.factory import make_pattern

# HPM AI v1/v2 Modules
from hpm_ai_v1.substrates.code_substrate import LocalCodeSubstrate
from hpm_ai_v1.core.mutator import CodeMutationActor
from hpm_ai_v1.sandbox.executor import SandboxExecutor
from hpm_ai_v1.core.l5_compiler import L5Compiler
from hpm_ai_v1.store.concurrent_sqlite import ConcurrentSQLiteStore
from hpm_ai_v1.transpiler.encoders import ASTL2Encoder
from hpm_ai_v1.core.benchmark_generator import AutonomousBenchmarkGenerator
from hpm_ai_v1.transpiler.mmr import MMRTranslator, ProjectManifold

class SuccessionController:
    def __init__(self, repo_path: str, db_path: str, target_file: str):
        self.repo_path = repo_path
        self.target_file = target_file
        self.store = ConcurrentSQLiteStore(db_path)
        self.field = PatternField()
        self.code_sub = LocalCodeSubstrate(repo_path)
        self.math_sub = MathSubstrate(feature_dim=16)
        self.pypi_sub = PyPISubstrate(seed_packages=["functools", "numpy"])
        self.sandbox = SandboxExecutor(repo_path)
        self.mutator = CodeMutationActor(l2_dim=16, l3_dim=32)
        self.l2_enc = ASTL2Encoder()
        self.bench_gen = AutonomousBenchmarkGenerator(repo_path)
        self.mmr_trans = MMRTranslator()
        self.project_manifold = ProjectManifold()
        
        self.agents = [
            Agent(AgentConfig(agent_id=f"miner_{i}", feature_dim=16), store=self.store, field=self.field)
            for i in range(2)
        ]
        self.orchestrator = MultiAgentOrchestrator(self.agents, self.field)
        self.max_generations = 5
        self.test_command = "pytest tests/ -v"

    def build_project_manifold(self):
        """Ingest the entire codebase into the Global Brain (ProjectManifold)."""
        print("Ingesting Project Manifold (Global Brain)...")
        for filepath in self.code_sub.get_all_python_files():
            tree = self.code_sub.parse_ast(filepath)
            if tree:
                rel_graph = self.mmr_trans.to_relational_graph(tree)
                self.project_manifold.add_module(filepath, rel_graph)
        print(f"Ingested {len(self.project_manifold.modules)} modules.")

    def run_succession_loop(self):
        self.build_project_manifold()
        
        # 1. Initial Baseline
        print(f"Evaluating Baseline (Generation 0) for {self.target_file}...")
        target_path = os.path.join(self.repo_path, self.target_file)
        tree = self.code_sub.parse_ast(target_path)
        node_count = len(list(ast.walk(tree))) if tree else 1000
        
        baseline = self.sandbox.evaluate_code("", self.target_file, test_command=self.test_command)
        if not baseline["success"]:
            print("Baseline failing tests. Aborting.")
            return
            
        print(f"Baseline: Cost={baseline['cost_time']:.4f}s, Nodes={node_count}")
        self.compiler = L5Compiler(baseline_cost=baseline["cost_time"], baseline_node_count=node_count)

        # 2. Succession Loop
        for gen in range(1, self.max_generations + 1):
            print(f"\n--- Generation {gen} ---")
            
            # Step A: Relational Synthesis (Mutation via Vectorized MMR)
            current_source = self.code_sub.read_file(target_path)
            current_tree = self.code_sub.parse_ast(target_path)
            if not current_tree: break
            
            target_func = next((n for n in ast.walk(current_tree) if isinstance(n, ast.FunctionDef)), None)
            if not target_func: break
            
            l2_input = self.l2_enc.encode(target_func)
            # Mutation now uses relational recombination in manifold space
            new_source = self.mutator.propose_mutation(current_source, l2_input, l3_population=[])
            
            if not new_source or new_source == current_source:
                print("No novel logic forged. L5 stagnation tracking updated.")
                self.compiler.evaluate_mutation({"success": False}, current_source)
            else:
                # Project-Level Structural Immunity Check
                new_tree = ast.parse(new_source)
                new_mmr = self.mmr_trans.to_relational_graph(new_tree)
                if not self.project_manifold.check_structural_immunity(target_path, new_mmr):
                    self.compiler.evaluate_mutation({"success": False}, current_source)
                    continue

                # Step B: Sandbox Validation (Patch-based)
                print("Verifying New Generation in Sandbox (Unified Diff Head)...")
                result = self.sandbox.evaluate_code(new_source, self.target_file, test_command=self.test_command)
                
                # Step C: Metacognitive Gating (L5 + Structural Immunity)
                if self.compiler.evaluate_mutation(result, new_source):
                    self.compiler.commit_succession(new_source, self.repo_path, self.target_file)
                
            # Step D: Autonomous Benchmark Expansion (Stagnation Trigger)
            if self.compiler.stagnation_counter >= 3:
                print("!!! Stagnation Triggered (S < 0.05) !!!")
                # Forge new constraints to drive evolution
                new_test_path = self.bench_gen.generate_conflict_benchmark(reason="stagnation")
                # Add to subsequent test runs
                # (In real system, we'd append to test_command)
                
            time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="HPM AI v2.2 Succession Controller")
    parser.add_argument("--repo_path", type=str, default=".", help="Path to codebase")
    parser.add_argument("--target_file", type=str, default="hpm/evaluators/resource_cost.py", help="File to mutate")
    parser.add_argument("--db_path", type=str, default="hpm_ai_v2.db", help="Path to pattern store")
    args = parser.parse_args()

    controller = SuccessionController(args.repo_path, args.db_path, args.target_file)
    controller.run_succession_loop()

if __name__ == "__main__":
    main()
