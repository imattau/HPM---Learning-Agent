"""L5 Monitor Agent for HPM AI v3.2.

Standalone agent specializing in Structural Immunity and Soft Pareto Gating.
Uses a Lagrangian approach to balance Accuracy vs Complexity.
"""
import ast
import os
from typing import Dict, Any, List, Optional
import numpy as np

from hpm.agents.agent import Agent
from hpm.config import AgentConfig
from hpm.agents.l5_monitor import L5MetaMonitor
from hpm.evaluators.resource_cost import ResourceCostEvaluator
from hpm_ai_v1.core.mutator import ChangeSet

class L5MonitorAgent(Agent):
    def __init__(self, config: AgentConfig, baseline_cost: float, baseline_node_count: int, **kwargs):
        super().__init__(config, **kwargs)
        self.best_cost = baseline_cost
        self.best_node_count = baseline_node_count
        self.meta_monitor = L5MetaMonitor()
        self.generation = 1
        self.stagnation_counter = 0
        self.allow_bloat = False
        # Lagrangian Multiplier for node count penalty
        self.lam = 0.5 

    def _get_node_count(self, source: str) -> int:
        """Calculate the Description Length (node count) of the source code."""
        try:
            tree = ast.parse(source)
            return len(list(ast.walk(tree)))
        except SyntaxError:
            return 1000000

    def _is_structurally_immune(self, patch: str) -> bool:
        """Review the unified diff for logic-breaking patterns."""
        if not patch: return True
        if "---" in patch and "+++" in patch:
            lines = patch.splitlines()
            deleted_count = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
            added_count = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
            if deleted_count > 5 and added_count <= 2:
                return False
        return True

    def evaluate_changeset(self, sandbox_result: Dict[str, Any], changeset: ChangeSet) -> bool:
        """
        Soft Pareto Gating logic.
        Score = delta_accuracy - lambda * delta_nodes
        """
        if not sandbox_result["success"]:
            if sandbox_result.get("error_type") in ("TypeError", "AttributeError", "NameError"):
                print(f"L5 Monitor: Global Contradiction detected. S=1.0")
                self.stagnation_counter += 1
                return False 
            print(f"L5 Monitor: Sandbox execution failed.")
            self.stagnation_counter += 1
            return False
            
        patch = sandbox_result.get("patch", "")
        if not self._is_structurally_immune(patch):
            print("L5 Monitor: Mutation failed Structural Immunity Review.")
            self.stagnation_counter += 1
            return False

        primary_file = list(changeset.mutations.keys())[0]
        new_source = changeset.mutations[primary_file]
        new_node_count = self._get_node_count(new_source)
        new_cost = sandbox_result["cost_time"]
        surprise = sandbox_result.get("surprise", 0.0)

        # Dynamic Lambda Adjustment
        current_lam = self.lam
        if self.stagnation_counter > 3:
            current_lam = 0.05 # Lower penalty during Bloat Window
            print(f"L5 Metacognition: Stagnation detected. Lowering Lagrangian Multiplier (lambda={current_lam})")
        
        # Performance Delta (Normalised speedup)
        d_perf = (self.best_cost - new_cost) / self.best_cost if self.best_cost > 0 else 0
        # Node Delta (Normalised complexity increase)
        d_nodes = (new_node_count - self.best_node_count) / self.best_node_count if self.best_node_count > 0 else 0
        
        # Soft Pareto Score
        # Novelty Vector is represented by sandbox surprise (divergence from baseline)
        novelty = surprise 
        
        # 1. If nodes decreased, AUTO-ACCEPT unless tests failed (Elegance Gain)
        if new_node_count < self.best_node_count:
            print(f"L5 Monitor: Accept Elegance Gain ({new_node_count} < {self.best_node_count})")
            self._update_best(new_cost, new_node_count)
            return True
            
        # 2. If nodes increased, only accept if stagnation is active and novelty is high
        if self.stagnation_counter > 3:
            if novelty > 0.7 and new_node_count <= self.best_node_count * 1.2:
                print(f"L5 Monitor: Accept High-Novelty Bloat (Novelty={novelty:.2f})")
                self._update_best(new_cost, new_node_count)
                return True
            else:
                print(f"L5 Reject: Bloat rejected (Novelty {novelty:.2f} <= 0.7)")
        else:
            print(f"L5 Reject: Complexity increase forbidden outside Bloat Window.")

        self.stagnation_counter += 1
        return False

    def _update_best(self, cost: float, node_count: int):
        self.best_cost = cost
        self.best_node_count = node_count
        self.generation += 1
        self.stagnation_counter = 0
        print(f"L5 Monitor: Generation {self.generation} promoted.")

    def commit_succession(self, changeset: ChangeSet, repo_path: str) -> None:
        """Permanently apply the ChangeSet."""
        for rel_path, source in changeset.mutations.items():
            full_path = os.path.join(repo_path, rel_path)
            with open(full_path, "w", encoding='utf-8') as f:
                f.write(source)
        print(f"*** Generation {self.generation} Succession Complete ***")
