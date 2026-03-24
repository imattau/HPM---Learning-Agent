"""Sandbox Executor for HPM AI v2.3.

Safely executes the 'Next Generation' codebase in a temporary directory.
Now supports Multi-File ChangeSets and Traceback capture.
"""
import os
import shutil
import tempfile
import subprocess
import time
from typing import Dict, Any, List
from hpm_ai_v1.core.mutator import ChangeSet

class SandboxExecutor:
    def __init__(self, original_repo_path: str):
        self.original_repo_path = original_repo_path

    def evaluate_changeset(self, changeset: ChangeSet, test_command: str = "pytest tests/") -> Dict[str, Any]:
        """Runs the improved code in a temp directory and captures metrics."""
        result = {
            "success": False,
            "cost_time": float('inf'),
            "surprise": 1.0,
            "output": "",
            "error_type": None
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dest_path = os.path.join(tmpdir, "repo")
            shutil.copytree(
                self.original_repo_path, 
                dest_path, 
                ignore=shutil.ignore_patterns('.git', '.venv', '__pycache__', 'data', '*.db')
            )
            
            # Apply all mutations in the changeset
            for rel_path, new_source in changeset.mutations.items():
                full_path = os.path.join(dest_path, rel_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding='utf-8') as f:
                    f.write(new_source)
                
            # Run test suite
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
                result["surprise"] = 0.0 
            else:
                result["surprise"] = 1.0 
                # Analyze error type
                if "TypeError" in result["output"]: result["error_type"] = "TypeError"
                elif "AttributeError" in result["output"]: result["error_type"] = "AttributeError"
                elif "NameError" in result["output"]: result["error_type"] = "NameError"
                
        return result
