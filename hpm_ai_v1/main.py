"""HPM AI v3.0: Sovereign Orchestrator.

Moves from a centralized loop to a Multi-Agent Structured Hierarchy.
Orchestrates L4 Architect and L5 Monitor agents to achieve project-wide 
Succession without Global Contradiction.
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

# HPM AI v3.0 Modules
from hpm_ai_v1.substrates.code_substrate import LocalCodeSubstrate
from hpm_ai_v1.core.mutator import L4ArchitectAgent, ChangeSet
from hpm_ai_v1.sandbox.executor import SandboxExecutor
from hpm_ai_v1.core.l5_compiler import L5MonitorAgent
from hpm_ai_v1.store.concurrent_sqlite import ConcurrentSQLiteStore
from hpm_ai_v1.transpiler.encoders import ASTL2Encoder
from hpm_ai_v1.transpiler.mmr import MMRTranslator, ProjectTopology
from hpm_ai_v1.core.librarian import CodeLibrarian

class SovereignOrchestrator:
    def __init__(self, repo_path: str, db_path: str, target_file: str):
        self.repo_path = os.path.abspath(repo_path)
        self.target_file = target_file
        self.store = ConcurrentSQLiteStore(db_path)
        self.field = PatternField()
        self.code_sub = LocalCodeSubstrate(self.repo_path)
        self.math_sub = MathSubstrate(feature_dim=16)
        self.sandbox = SandboxExecutor(self.repo_path)
        
        self.l2_enc = ASTL2Encoder()
        self.mmr_trans = MMRTranslator()
        self.topology = ProjectTopology()
        self.librarian = CodeLibrarian(self.topology)
        
        # 1. Initialize Specialist Agents
        self.l4_architect = L4ArchitectAgent(
            AgentConfig(agent_id="l4_architect", feature_dim=16),
            store=self.store, field=self.field
        )
        
        self.l5_monitor = None # Initialized in run_succession_loop
        
        self.miners = [
            Agent(AgentConfig(agent_id=f"miner_{i}", feature_dim=16), store=self.store, field=self.field)
            for i in range(2)
        ]
        
        self.ensemble = [self.l4_architect] + self.miners
        self.orchestrator = MultiAgentOrchestrator(self.ensemble, self.field)
        self.test_command = "pytest tests/benchmarks/ -v"

    def build_topology(self):
        """Map the Relational Ecology of the codebase."""
        print("Sovereign Ingestion: Building Project Manifold...")
        for filepath in self.code_sub.get_all_python_files():
            tree = self.code_sub.parse_ast(filepath)
            if tree:
                mmr_root = self.mmr_trans.to_relational_graph(tree, filepath)
                self.topology.add_module(filepath, mmr_root)
        print(f"Project Manifold active: {len(self.topology.modules)} nodes mapped.")

    def run_succession_loop(self, max_gens: int = 5):
        self.build_topology()
        target_path = os.path.join(self.repo_path, self.target_file)
        
        # Step 1: Establish Baseline
        current_tree = self.code_sub.parse_ast(target_path)
        node_count = len(list(ast.walk(current_tree))) if current_tree else 1000
        
        baseline = self.sandbox.evaluate_changeset(ChangeSet(), test_command=self.test_command)
        if not baseline["success"]:
            print("Baseline failure. Refactoring inhibited.")
            return
            
        print(f"Generation 0: Cost={baseline['cost_time']:.4f}s, Nodes={node_count}")
        
        self.l5_monitor = L5MonitorAgent(
            AgentConfig(agent_id="l5_monitor", feature_dim=16),
            baseline_cost=baseline["cost_time"],
            baseline_node_count=node_count,
            store=self.store, field=self.field
        )
        self.orchestrator.agents.append(self.l5_monitor)

        # Step 2: The Succession Loop
        for gen in range(1, max_gens + 1):
            print(f"\n--- Generation {gen} ---")
            
            target_func = next((n for n in ast.walk(current_tree) if isinstance(n, ast.FunctionDef)), None)
            if not target_func: break
            
            l2_in = self.l2_enc.encode(target_func)
            
            # DESIGN PHASE
            changeset = self.l4_architect.propose_cascading_mutation(
                self.repo_path, self.target_file, self.topology, l2_in, l3_population=[]
            )
            
            if not changeset.mutations:
                print("L4 Architect: No viable architectural shift discovered.")
                continue
                
            # REPAIR PHASE (Manifold-Based)
            new_source = changeset.mutations[self.target_file]
            new_tree = ast.parse(new_source)
            new_func = next((n for n in ast.walk(new_tree) if isinstance(n, ast.FunctionDef)), None)
            
            if new_func and new_func.name != target_func.name:
                # Trigger Structural Shift Broadcast
                impacted = self.librarian.broadcast_structural_shift(self.target_file, target_func.name, new_func.name)
                # The Architect already handles basic renaming in the ChangeSet
            
            # VALIDATION PHASE
            print(f"ChangeSet designed: {list(changeset.mutations.keys())}")
            result = self.sandbox.evaluate_changeset(changeset, test_command=self.test_command)
            
            if self.l5_monitor.evaluate_changeset(result, changeset):
                self.l5_monitor.commit_succession(changeset, self.repo_path)
                # Success: update global brain
                for path, src in changeset.mutations.items():
                    tree = ast.parse(src)
                    mmr = self.mmr_trans.to_relational_graph(tree, path)
                    self.librarian.update_manifold(path, mmr)
                
                current_tree = self.code_sub.parse_ast(target_path)
            else:
                if result.get("surprise", 0.0) >= 1.0:
                    print("L5 Monitor: GLOBAL CONTRADICTION detected. Triggering Repair Turn.")
                    # Trigger repair task (Phase 3 logic)
            
            time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="HPM AI v3.0 Sovereign Orchestrator")
    parser.add_argument("--repo_path", type=str, default=".", help="Path to project root")
    parser.add_argument("--target_file", type=str, default="hpm_ai_v1/core/l5_compiler.py", help="Target for refactor")
    parser.add_argument("--db_path", type=str, default="hpm_ai_v3.db", help="Persistent store path")
    args = parser.parse_args()

    orchestrator = SovereignOrchestrator(args.repo_path, args.db_path, args.target_file)
    orchestrator.run_succession_loop()

if __name__ == "__main__":
    main()
