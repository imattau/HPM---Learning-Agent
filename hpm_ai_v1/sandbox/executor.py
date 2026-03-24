"""Sandbox Executor for HPM AI v1.

Safely applies patches to a temporary copy of the codebase and runs benchmarks
to verify if the mutation improves Epistemic Residual or Resource Cost.
"""
import os
import shutil
import tempfile
import subprocess
from typing import Dict, Any

class SandboxExecutor:
    def __init__(self, original_repo_path: str):
        self.original_repo_path = original_repo_path

    def evaluate_patch(self, patch_content: str, test_command: str = "pytest tests/") -> Dict[str, Any]:
        """Applies a patch in a temp directory and runs the test command."""
        result = {
            "success": False,
            "cost_time": float('inf'),
            "surprise": 1.0,
            "output": ""
        }
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy codebase (excluding heavy things like .git, .venv if needed)
            # For simplicity, we assume original_repo_path is small enough, 
            # or we just copy specific folders.
            dest_path = os.path.join(tmpdir, "repo")
            shutil.copytree(
                self.original_repo_path, 
                dest_path, 
                ignore=shutil.ignore_patterns('.git', '.venv', '__pycache__', 'data')
            )
            
            patch_file = os.path.join(tmpdir, "mutation.patch")
            with open(patch_file, "w") as f:
                f.write(patch_content)
                
            # Apply patch
            apply_cmd = ["patch", "-p0", "-i", patch_file]
            patch_proc = subprocess.run(apply_cmd, cwd=dest_path, capture_output=True, text=True)
            
            if patch_proc.returncode != 0:
                result["output"] = f"Patch failed: {patch_proc.stderr}"
                return result
                
            # Run test
            import time
            start_time = time.time()
            test_proc = subprocess.run(
                test_command.split(), 
                cwd=dest_path, 
                capture_output=True, 
                text=True
            )
            end_time = time.time()
            
            result["cost_time"] = end_time - start_time
            result["output"] = test_proc.stdout + "\n" + test_proc.stderr
            
            if test_proc.returncode == 0:
                result["success"] = True
                result["surprise"] = 0.0 # No epistemic residual if tests pass perfectly
            else:
                result["surprise"] = 1.0 # High surprise if tests fail
                
        return result
