"""L5 Metacognitive Compiler for HPM AI v1.

Acts as the final judge of code quality. Enforces the HPM Elegance Principle:
A mutation is rejected unless it reduces AST node count (MDL) or 
significantly improves performance (>15%).
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

    def _get_node_count(self, source: str) -> int:
        """Calculate the Description Length (node count) of the source code."""
        try:
            tree = ast.parse(source)
            return len(list(ast.walk(tree)))
        except SyntaxError:
            return float('inf')

    def evaluate_mutation(self, sandbox_result: Dict[str, Any], new_source: str) -> bool:
        """
        Gating logic for accepting a self-authored generation.
        1. Must pass tests (success = True)
        2. Must not increase epistemic surprise (Logical Contradiction)
        3. Must satisfy Elegance Principle (Pareto efficiency on Length vs. Speed)
        """
        if not sandbox_result["success"]:
            print(f"L5 Reject: Sandbox tests failed.")
            return False
            
        if sandbox_result["surprise"] > 0.5:
            print(f"L5 Reject: High epistemic surprise detected.")
            return False

        new_node_count = self._get_node_count(new_source)
        new_cost = sandbox_result["cost_time"]

        # Elegance Principle: 
        # Reject if node count increased UNLESS cost decreased by > 15%
        cost_improvement = (self.best_cost - new_cost) / self.best_cost if self.best_cost > 0 else 0
        
        if new_node_count > self.best_node_count:
            if cost_improvement < 0.15:
                print(f"L5 Reject: Complexity increased ({new_node_count} > {self.best_node_count}) "
                      f"without sufficient performance gain ({cost_improvement:.1%}).")
                return False
        
        # Pareto Check: Must not be worse in either dimension than best known
        if new_cost > self.best_cost and new_node_count >= self.best_node_count:
            print(f"L5 Reject: Not Pareto efficient. Cost and complexity are both worse.")
            return False

        # If we made it here, the mutation is verified
        self.best_cost = new_cost
        self.best_node_count = new_node_count
        self.generation += 1
        
        print(f"L5 Accept: Generation {self.generation} verified.")
        print(f"  Cost: {new_cost:.4f}s ({cost_improvement:+.1%})")
        print(f"  Nodes: {new_node_count} (Best: {self.best_node_count})")
        return True

    def commit_succession(self, new_source: str, repo_path: str, target_file: str) -> None:
        """Pass the torch: Permanently overwrite the live source code."""
        full_path = os.path.join(repo_path, target_file)
        with open(full_path, "w", encoding='utf-8') as f:
            f.write(new_source)
        print(f"*** Succession Complete: Generation {self.generation} is now live ***")
