"""L5 Metacognitive Compiler for HPM AI v2.1.

Enforces the HPM Elegance Principle with a stagnation-triggered 'Bloat Window'
and formal Structural Immunity Review of unified diff patches.
"""
import ast
import os
from typing import Dict, Any
from hpm.agents.l5_monitor import L5MetaMonitor
from hpm.evaluators.resource_cost import ResourceCostEvaluator

class L5Compiler:
    def __init__(self, baseline_cost: float, baseline_node_count: int, baseline_surprise: float = 0.0):
        self.best_cost = baseline_cost
        self.best_node_count = baseline_node_count
        self.monitor = L5MetaMonitor()
        self.resource_eval = ResourceCostEvaluator(lambda_cost=1.0)
        self.generation = 1
        self.stagnation_counter = 0
        self.allow_bloat = False

    def _get_node_count(self, source: str) -> int:
        """Calculate the Description Length (node count) of the source code."""
        try:
            tree = ast.parse(source)
            return len(list(ast.walk(tree)))
        except SyntaxError:
            return 1000000

    def _is_structurally_immune(self, patch: str) -> bool:
        """Review the unified diff for logic-breaking patterns (Taboos)."""
        if not patch: return True
        
        # Example Taboos: 
        # - Deleting a function entirely without replacement
        if "---" in patch and "+++" in patch:
            lines = patch.splitlines()
            deleted_count = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
            added_count = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
            
            # If we delete 10+ lines and add only 1 (likely 'pass'), it's a structural regression
            if deleted_count > 10 and added_count <= 2:
                return False
        return True

    def evaluate_mutation(self, sandbox_result: Dict[str, Any], new_source: str) -> bool:
        """
        Gating logic for accepting a self-authored generation.
        1. Must pass tests (success = True)
        2. Must pass Structural Immunity Review
        3. Must be Pareto-efficient (Elegance Principle)
        """
        if not sandbox_result["success"]:
            print(f"L5 Reject: Sandbox tests failed.")
            self.stagnation_counter += 1
            return False
            
        # 1. Structural Immunity Review
        patch = sandbox_result.get("patch", "")
        if not self._is_structurally_immune(patch):
            print("L5 Reject: Mutation failed Structural Immunity Review (Logic Contradiction).")
            self.stagnation_counter += 1
            return False

        new_node_count = self._get_node_count(new_source)
        new_cost = sandbox_result["cost_time"]
        surprise = sandbox_result.get("surprise", 0.0)

        # Update allow_bloat flag based on stagnation
        if self.stagnation_counter > 3:
            self.allow_bloat = True
            print("L5 Metacognition: Stagnation detected. Enabling Bloat Window.")
        else:
            self.allow_bloat = False

        # BLOAT WINDOW Logic
        if self.allow_bloat and new_node_count <= self.best_node_count * 1.2:
            if surprise > 0.5: 
                print(f"L5 Accept (Bloat Window): Complexity increased for novel logic (S={surprise:.2f})")
                self._update_best(new_cost, new_node_count)
                return True

        # Standard Pareto / Elegance Gating
        cost_improvement = (self.best_cost - new_cost) / self.best_cost if self.best_cost > 0 else 0
        
        if new_node_count > self.best_node_count:
            if cost_improvement < 0.15:
                print(f"L5 Reject: Complexity increased ({new_node_count} > {self.best_node_count}) "
                      f"without sufficient performance gain ({cost_improvement:.1%}).")
                self.stagnation_counter += 1
                return False
        
        if new_cost > self.best_cost and new_node_count >= self.best_node_count:
            print(f"L5 Reject: Not Pareto efficient.")
            self.stagnation_counter += 1
            return False

        # If we made it here, the mutation is verified
        self._update_best(new_cost, new_node_count)
        return True

    def _update_best(self, cost: float, node_count: int):
        self.best_cost = cost
        self.best_node_count = node_count
        self.generation += 1
        self.stagnation_counter = 0 # Reset on success
        print(f"L5 Accept: Generation {self.generation} verified.")
        print(f"  Cost: {self.best_cost:.4f}s, Nodes: {self.best_node_count}")

    def commit_succession(self, new_source: str, repo_path: str, target_file: str) -> None:
        """Pass the torch: Permanently overwrite the live source code."""
        full_path = os.path.join(repo_path, target_file)
        with open(full_path, "w", encoding='utf-8') as f:
            f.write(new_source)
        print(f"*** Succession Complete: Generation {self.generation} is now live ***")
