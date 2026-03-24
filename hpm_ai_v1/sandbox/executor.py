"""Sandbox Executor for HPM AI v1.

Safely executes the 'Next Generation' codebase in a temporary directory.
Uses AST-Native Refactoring (direct code injection) to verify if the 
mutation improves accuracy/cost without relying on fragile CLI patches.
"""
import os
import shutil
import tempfile
import subprocess
import time
from typing import Dict, Any

class SandboxExecutor:
    def __init__(self, original_repo_path: str):
        self.original_repo_path = original_repo_path

    def evaluate_code(self, new_source: str, target_file: str, test_command: str = "pytest tests/") -> Dict[str, Any]:
        """Runs the improved code in a temp directory and captures metrics."""
        result = {
            "success": False,
            "cost_time": float('inf'),
            "surprise": 1.0,
            "output": ""
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dest_path = os.path.join(tmpdir, "repo")
            shutil.copytree(
                self.original_repo_path, 
                dest_path, 
                ignore=shutil.ignore_patterns('.git', '.venv', '__pycache__', 'data', '*.db')
            )
            
            # AST-Native Refactoring: Direct overwrite of the target file
            # This is more robust than unified diff patching.
            full_target_path = os.path.join(dest_path, target_file)
            
            # Ensure parent directories exist
            os.makedirs(os.path.dirname(full_target_path), exist_ok=True)
            
            # Only overwrite if new_source is provided (otherwise it's a baseline run)
            if new_source:
                with open(full_target_path, "w", encoding='utf-8') as f:
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
                
        return result
