"""HPM AI v3.2.3: Sovereign Orchestrator (Substrate-Anchored).

Implements Global Saliency: autonomously identifies refactor targets.
Implements Algebraic MMR: topological verification of relational invariants.
Implements Soft Pareto Gating: Lagrangian cost weighting.
RE-INTEGRATES External Substrates: Math, Wikipedia, and PyPI for prior-guided evolution.
"""
import os
import ast
import argparse
import time
import numpy as np
from typing import List, Optional

# Core HPM Framework
from hpm.substrate.math import MathSubstrate
from hpm.substrate.pypi import PyPISubstrate
from hpm.substrate.wikipedia import WikipediaSubstrate
from hpm.agents.agent import Agent
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.config import AgentConfig
from hpm.field.field import PatternField
from hpm.patterns.factory import make_pattern

# HPM AI Modules
from hpm_ai_v1.substrates.code_substrate import LocalCodeSubstrate
from hpm_ai_v1.core.mutator import L4ArchitectAgent, ChangeSet
from hpm_ai_v1.sandbox.executor import SandboxExecutor
from hpm_ai_v1.core.l5_compiler import L5MonitorAgent
from hpm_ai_v1.store.concurrent_sqlite import ConcurrentSQLiteStore
from hpm_ai_v1.transpiler.encoders import ASTL2Encoder
from hpm_ai_v1.transpiler.mmr import MMRTranslator, ProjectTopology
from hpm_ai_v1.core.librarian import CodeLibrarian

class SovereignOrchestrator:
    def __init__(self, repo_path: str, db_path: str):
        self.repo_path = os.path.abspath(repo_path)
        self.store = ConcurrentSQLiteStore(db_path)
        self.field = PatternField()
        self.code_sub = LocalCodeSubstrate(self.repo_path)
        
        # 1. External Substrates (The Knowledge Mine)
        self.math_sub = MathSubstrate(feature_dim=16)
        self.pypi_sub = PyPISubstrate(seed_packages=["functools", "numpy", "collections", "itertools"])
        self.wiki_sub = WikipediaSubstrate()
        
        self.sandbox = SandboxExecutor(self.repo_path)
        self.l2_enc = ASTL2Encoder()
        self.mmr_trans = MMRTranslator()
        self.topology = ProjectTopology()
        # Librarian persists state via SQLite
        self.librarian = CodeLibrarian(self.topology, store=self.store)
        
        # Specialists
        self.l4_architect = L4ArchitectAgent(
            AgentConfig(agent_id="l4_architect", feature_dim=16),
            store=self.store, field=self.field
        )
        
        # Miners utilize external substrates to find relational priors
        self.miners = [
            Agent(AgentConfig(agent_id="math_miner", feature_dim=16, alpha_int=0.3), 
                  store=self.store, substrate=self.math_sub, field=self.field),
            Agent(AgentConfig(agent_id="pypi_miner", feature_dim=16, alpha_int=0.3), 
                  store=self.store, substrate=self.pypi_sub, field=self.field)
        ]
        
        self.orchestrator = MultiAgentOrchestrator([self.l4_architect] + self.miners, self.field)
        self.test_command = "pytest tests/benchmarks/ -v"
        self.l5_monitor = None

    def build_topology(self):
        print("Sovereign Ingestion: Building Global Manifold...")
        for filepath in self.code_sub.get_all_python_files():
            tree = self.code_sub.parse_ast(filepath)
            if tree:
                mmr_root = self.mmr_trans.to_relational_graph(tree, filepath)
                self.topology.add_module(filepath, mmr_root)
        print(f"Global Brain active: {len(self.topology.modules)} nodes mapped.")

    def run_prior_harvesting(self):
        """Active Prior Acquisition Step: Mines external substrates for universal L3 laws."""
        print("Knowledge Mine: Harvesting relational priors from Math and PyPI...")
        
        # 1. Mine 'Pareto' and 'Big-O' laws from Math
        pareto_priors = self.math_sub.fetch("pareto")
        for p_vec in pareto_priors:
            p = make_pattern(p_vec[:16], np.eye(16)*0.1, pattern_type="gaussian")
            p.label = "pareto_efficiency"
            self.field.broadcast("math_substrate", p)

        # 2. Mine 'Complexity' inhibitors
        complexity_mu = np.zeros(16)
        complexity_mu[1] = 50.0 # High node count
        complexity_mu[2] = 10.0 # High depth
        inhibitor = make_pattern(complexity_mu, np.eye(16), pattern_type="gaussian")
        self.field.broadcast_negative(inhibitor, 0.5, "complexity_monitor")

        return []

    def run_sovereign_loop(self, max_gens: int = 10):
        self.build_topology()
        
        # Initial Baseline for project health
        baseline = self.sandbox.evaluate_changeset(ChangeSet(), test_command=self.test_command)
        if not baseline["success"]:
            print("Project root is structurally unstable. Aborting.")
            return
            
        print(f"Global Baseline: Cost={baseline['cost_time']:.4f}s")
        
        # Succession Loop
        for gen in range(1, max_gens + 1):
            print(f"\n--- Generation {gen} ---")
            
            # Step 0: Anchor Knowledge & Social Reinforcement
            self.run_prior_harvesting()
            
            # SOCIAL REINFORCEMENT: Miners 'step' to align with manifold
            for _ in range(5):
                # Simulated mining tasks
                obs_math = self.math_sub.fetch("statistics")[0][:16]
                obs_code = np.random.standard_normal(16) 
                
                self.orchestrator.step({
                    "math_miner": obs_math,
                    "pypi_miner": obs_code
                })

            # Step 1: Autonomous Saliency Scan
            target_path = self.librarian.get_most_salient_target()
            if not target_path:
                print("No salient targets found. Evolution complete?")
                break
            
            rel_target = os.path.relpath(target_path, self.repo_path)
            print(f"Logic Forge: Targeting {rel_target} for structural refinement.")
            
            # Initialize L5 Monitor for this specific target
            tree = self.code_sub.parse_ast(target_path)
            node_count = len(list(ast.walk(tree)))
            self.l5_monitor = L5MonitorAgent(
                AgentConfig(agent_id=f"l5_gen_{gen}", feature_dim=16),
                baseline_cost=baseline["cost_time"],
                baseline_node_count=node_count,
                store=self.store, field=self.field
            )

            # Step 2: Relational Synthesis (Prior-Guided)
            target_func = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None)
            if not target_func: continue
            
            l2_in = self.l2_enc.encode(target_func)
            l3_population = [] 
            
            changeset = self.l4_architect.propose_cascading_mutation(
                self.repo_path, rel_target, self.topology, l2_in, l3_population=l3_population
            )
            
            if not changeset.mutations:
                print("L4 Architect: No viable shift discovered.")
                continue
                
            # Step 3: Sandbox Verification
            print(f"ChangeSet Proposed for {list(changeset.mutations.keys())}")
            result = self.sandbox.evaluate_changeset(changeset, test_command=self.test_command)
            
            # Step 4: Soft Pareto Gating
            if self.l5_monitor.evaluate_changeset(result, changeset):
                self.l5_monitor.commit_succession(changeset, self.repo_path)
                
                # Update global brain with NEW structural truth
                for path, src in changeset.mutations.items():
                    try:
                        new_tree = ast.parse(src)
                        mmr = self.mmr_trans.to_relational_graph(new_tree, path)
                        self.librarian.update_manifold(path, mmr)
                    except: pass
                
                # Re-sync baseline
                baseline = result
                self.build_topology()
            else:
                # Feedback to Librarian to prevent Saliency Traps
                self.librarian.report_failure(target_path)
                
                if result.get("surprise", 0.0) >= 1.0:
                    print("L5 Monitor: GLOBAL CONTRADICTION detected. Triggering Repair Turn.")
            
            time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="HPM AI v3.2.3 Logic Forge")
    parser.add_argument("--repo_path", type=str, default=".", help="Path to project root")
    parser.add_argument("--db_path", type=str, default="hpm_ai_v3.db", help="Persistent store path")
    args = parser.parse_args()

    orchestrator = SovereignOrchestrator(args.repo_path, args.db_path)
    orchestrator.run_sovereign_loop()

if __name__ == "__main__":
    main()
