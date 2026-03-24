"""HPM AI v1: Succession Controller.

Implements the recursive Generation Succession model. Moves from 
stochastic patching to Manifold Alignment and Relational Synthesis.
"""
import os
import ast
import argparse
import time
import numpy as np
from typing import List

# Core HPM Framework
from hpm.substrate.math import MathSubstrate
from hpm.agents.agent import Agent
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.config import AgentConfig
from hpm.field.field import PatternField
from hpm.patterns.factory import make_pattern

# HPM AI v1 Modules
from hpm_ai_v1.substrates.code_substrate import LocalCodeSubstrate
from hpm_ai_v1.core.mutator import CodeMutationActor
from hpm_ai_v1.sandbox.executor import SandboxExecutor
from hpm_ai_v1.core.l5_compiler import L5Compiler
from hpm_ai_v1.store.concurrent_sqlite import ConcurrentSQLiteStore
from hpm_ai_v1.transpiler.encoders import ASTL2Encoder

class SuccessionController:
    def __init__(self, repo_path: str, db_path: str, target_file: str):
        self.repo_path = repo_path
        self.target_file = target_file
        self.store = ConcurrentSQLiteStore(db_path)
        self.field = PatternField()
        self.code_sub = LocalCodeSubstrate(repo_path)
        self.math_sub = MathSubstrate()
        self.sandbox = SandboxExecutor(repo_path)
        self.mutator = CodeMutationActor(l2_dim=16, l3_dim=32)
        self.l2_enc = ASTL2Encoder()
        
        # Initialize ensemble
        self.agents = [
            Agent(AgentConfig(agent_id=f"miner_{i}", feature_dim=16), store=self.store, field=self.field)
            for i in range(2)
        ]
        self.orchestrator = MultiAgentOrchestrator(self.agents, self.field)
        
        self.stagnation_counter = 0
        self.max_generations = 5 # For safety in prototype

    def bridge_big_o_inhibitors(self):
        """Fetch 'Big O' patterns and inject them as negative inhibitors into the field."""
        print("Substrate Bridging: Injecting Big-O complexity inhibitors...")
        # Mocking Big-O discovery: high node count + high loop depth = negative
        complex_mu = np.zeros(16)
        complex_mu[1] = 50.0 # High line count
        complex_mu[2] = 10.0 # High AST depth
        
        inhibitor = make_pattern(complex_mu, np.eye(16), pattern_type="gaussian")
        # Direct field manipulation for prototype
        self.field.broadcast_negative(inhibitor, 0.5, "math_substrate")

    def train_mutator_intuition(self):
        """Pre-train the L4 head to recognize 'Elegance' (e.g., adding a docstring)."""
        print("Pre-training L4 Generative Head on 'Elegance' priors...")
        # Example: Input anatomy (no docstring) -> Target Law (add docstring)
        # L2 index 4 is 'has docstring'
        l2_no_doc = np.zeros(16)
        l2_no_doc[4] = 0.0
        
        l2_with_doc = np.zeros(16)
        l2_with_doc[4] = 1.0
        
        # L3 Law is the delta
        l3_add_doc = np.pad(l2_with_doc - l2_no_doc, (0, 16))
        
        for _ in range(5):
            self.mutator.l4_head.accumulate(l2_no_doc, l3_add_doc)
        self.mutator.l4_head.fit()

    def run_succession_loop(self):
        # 1. Initial Baseline
        print("Evaluating Baseline (Generation 0)...")
        target_path = os.path.join(self.repo_path, self.target_file)
        source = self.code_sub.read_file(target_path)
        if not source: return

        tree = self.code_sub.parse_ast(target_path)
        node_count = len(list(ast.walk(tree))) if tree else 1000
        
        baseline = self.sandbox.evaluate_code("", self.target_file)
        if not baseline["success"]:
            print("Baseline failing tests. Aborting.")
            return
            
        print(f"Baseline: Cost={baseline['cost_time']:.4f}s, Nodes={node_count}")
        self.compiler = L5Compiler(baseline_cost=baseline["cost_time"], baseline_node_count=node_count)

        # Pre-train intuition
        self.train_mutator_intuition()

        # 2. Succession Loop
        for gen in range(1, self.max_generations + 1):
            print(f"\n--- Generation {gen} ---")
            
            # Step A: Knowledge Mining & Inhibitor Injection
            self.bridge_big_o_inhibitors()
            
            # Step B: Relational Gathering (L3 Population)
            # Find high-weight patterns from previous successful refactors
            all_patterns = self.store.query_all()
            l3_population = []
            for p_dict, weight, _ in all_patterns:
                if weight > 0.5: # Only trust strong laws
                    # In a full system, p_dict would be converted back to AST subtrees
                    # Mocking a donor for prototype
                    l3_population.append(ast.parse("def donor(): pass").body[0])

            # Step C: Relational Synthesis (Mutation)
            current_source = self.code_sub.read_file(target_path)
            current_tree = self.code_sub.parse_ast(target_path)
            if not current_tree: break
            
            target_func = next((n for n in ast.walk(current_tree) if isinstance(n, ast.FunctionDef)), None)
            if not target_func: break
            
            l2_input = self.l2_enc.encode(target_func)
            new_source = self.mutator.propose_mutation(current_source, l2_input, l3_population)
            
            if not new_source or new_source == current_source:
                print("No novel logic forged. Stagnation increasing.")
                self.stagnation_counter += 1
            else:
                # Step D: Sandbox Validation
                print("Verifying New Generation in Sandbox...")
                result = self.sandbox.evaluate_code(new_source, self.target_file)
                
                # Step E: Metacognitive Gating (L5)
                if self.compiler.evaluate_mutation(result, new_source):
                    print(f"Succession Approved: Applying Generation {gen}")
                    self.compiler.commit_succession(new_source, self.repo_path, self.target_file)
                    self.stagnation_counter = 0
                else:
                    self.stagnation_counter += 1

            # Step F: Stagnation Trigger
            if self.stagnation_counter >= 3:
                print("!!! Stagnation Triggered (S < 0.01) !!!")
                print("Introducing new latent constraint: Memory Limit 256MB")
                # In real system, would modify config or environment
                self.stagnation_counter = 0
                
            # Slow down loop for visibility
            time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="HPM AI v1 Succession Controller")
    parser.add_argument("--repo_path", type=str, default=".", help="Path to codebase")
    parser.add_argument("--target_file", type=str, required=True, help="File to mutate")
    parser.add_argument("--db_path", type=str, default="hpm_ai_v1.db", help="Path to pattern store")
    args = parser.parse_args()

    controller = SuccessionController(args.repo_path, args.db_path, args.target_file)
    controller.run_succession_loop()

if __name__ == "__main__":
    main()
