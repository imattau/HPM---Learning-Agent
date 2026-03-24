"""L5 Metacognitive Compiler for HPM AI v1.

Acts as the final judge of code quality. Rejects diffs that introduce
syntax errors, logical contradictions, or violate Pareto Efficiency.
from typing import Dict, Any
from hpm.agents.l5_monitor import L5MetaMonitor
from hpm.evaluators.resource_cost import ResourceCostEvaluator

class L5Compiler:
    def __init__(self, baseline_cost: float, baseline_surprise: float = 0.0):
        self.best_cost = baseline_cost
        self.monitor = L5MetaMonitor()
        self.resource_eval = ResourceCostEvaluator(lambda_cost=1.0)
        self.generation = 1

    def evaluate_mutation(self, sandbox_result: Dict[str, Any], patch_content: str) -> bool:
        """
        Gating logic for accepting a self-authored patch.
        1. Must compile and pass tests (success = True)
        2. Must not increase epistemic surprise (Logical Contradiction)
        3. Must be Pareto-efficient (Better or equal accuracy, better or equal cost)
        4. Factor in Resource Pressure (description length / energy constraint)
        """
        if not sandbox_result["success"]:
            print(f"L5 Reject: Sandbox tests failed.")
            return False

        # Check current system pressure
        pressure = self.resource_eval.pressure()
        print(f"Current System Resource Pressure: {pressure:.2f}")
...

        # We simulate a 'prediction' vs 'actual' using the surprise field from sandbox
        # (In a real run, monitor.update() would be called inside the sandbox benchmarks)
        if sandbox_result["surprise"] > 0.5:
            print(f"L5 Reject: High epistemic surprise detected in results.")
            return False
            
        # Pareto Efficiency Check
        if sandbox_result["cost_time"] >= self.best_cost:
            print(f"L5 Reject: Not Pareto efficient. Cost increased ({sandbox_result['cost_time']:.4f}s >= {self.best_cost:.4f}s)")
            return False
            
        # If we made it here, the mutation is structurally sound and strictly better
        self.best_cost = sandbox_result["cost_time"]
        self.generation += 1
        
        print(f"L5 Accept: Mutation verified. Advancing to Generation {self.generation}")
        print(f"  New Cost: {self.best_cost:.4f}s")
        return True

    def commit_succession(self, patch_content: str, repo_path: str) -> None:
        """Pass the torch: Apply the diff to the live codebase permanently."""
        import subprocess
        import os
        patch_file = os.path.join(repo_path, "succession.patch")
        with open(patch_file, "w") as f:
            f.write(patch_content)
            
        apply_cmd = ["patch", "-p0", "-i", patch_file]
        subprocess.run(apply_cmd, cwd=repo_path, check=True)
        os.remove(patch_file)
        print(f"*** Generation {self.generation} Succession Complete ***")
