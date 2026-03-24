"""L5 Monitor Agent for HPM AI v3.1.

Standalone agent specializing in Structural Immunity and Elegance Gating.
Treats dependency breaks as high-surprise sensory signals.
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
        
        if "---" in patch and "+++" in patch:
            lines = patch.splitlines()
            deleted_count = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
            added_count = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
            
            # If we delete many lines and add almost none, it's a structural regression
            if deleted_count > 5 and added_count <= 2:
                return False
        return True

    def evaluate_changeset(self, sandbox_result: Dict[str, Any], changeset: ChangeSet) -> bool:
        """
        Gating logic for accepting a self-authored generation.
        Integrated into the agent's task evaluation.
        """
        # 1. Functional Integrity
        if not sandbox_result["success"]:
            # Dependency breaks are high-surprise signals (S=1.0)
            if sandbox_result.get("error_type") in ("TypeError", "AttributeError", "NameError"):
                print(f"L5 Monitor: Global Contradiction detected ({sandbox_result['error_type']}). S=1.0")
                self.stagnation_counter += 1
                return False 
            
            print(f"L5 Monitor: Logic failure in sandbox.")
            self.stagnation_counter += 1
            return False
            
        # 2. Structural Immunity
        patch = sandbox_result.get("patch", "")
        if not self._is_structurally_immune(patch):
            print("L5 Monitor: Structural Taboo violated (deletion without synthesis).")
            self.stagnation_counter += 1
            return False

        # 3. Elegance Principle
        primary_file = list(changeset.mutations.keys())[0]
        new_source = changeset.mutations[primary_file]
        new_node_count = self._get_node_count(new_source)
        new_cost = sandbox_result["cost_time"]
        surprise = sandbox_result.get("surprise", 0.0)

        # Stagnation and Bloat Window logic
        if self.stagnation_counter > 3:
            self.allow_bloat = True
            print("L5 Monitor: Stagnation detected. Enabling Bloat Window.")
        else:
            self.allow_bloat = False

        if self.allow_bloat and new_node_count <= self.best_node_count * 1.2:
            if surprise > 0.5: 
                print(f"L5 Monitor: Accept (Bloat Window) for novel logic discovery.")
                self._update_best(new_cost, new_node_count)
                return True

        cost_improvement = (self.best_cost - new_cost) / self.best_cost if self.best_cost > 0 else 0
        
        if new_node_count > self.best_node_count and cost_improvement < 0.15:
            print(f"L5 Monitor: Reject complexity increase (gain {cost_improvement:.1%}).")
            self.stagnation_counter += 1
            return False
        
        if new_cost >= self.best_cost * 0.95 and new_node_count >= self.best_node_count:
            print(f"L5 Monitor: Reject non-Pareto mutation.")
            self.stagnation_counter += 1
            return False

        self._update_best(new_cost, new_node_count)
        return True

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
