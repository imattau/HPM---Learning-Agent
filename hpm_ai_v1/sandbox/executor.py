"""Sandbox Executor for HPM AI v2.1.

Implements the 'Decoder Head' for Unified Diffs. Generates and applies 
standard .patch files to ensure repository integrity during self-modification.
"""
import os
import shutil
import tempfile
import subprocess
import time
import difflib
from typing import Dict, Any, Tuple

class SandboxExecutor:
    def __init__(self, original_repo_path: str):
        self.original_repo_path = original_repo_path

    def generate_patch(self, original_source: str, new_source: str, filepath: str) -> str:
        """Create a standard Unified Diff patch."""
        orig_lines = original_source.splitlines(keepends=True)
        new_lines = new_source.splitlines(keepends=True)
        diff = difflib.unified_diff(
            orig_lines, new_lines,
            fromfile=filepath,
            tofile=filepath,
            n=3
        )
        return "".join(diff)

    def evaluate_code(self, new_source: str, target_file: str, test_command: str = "pytest tests/") -> Dict[str, Any]:
        """Runs the improved code in a temp directory using a .patch file."""
        result = {
            "success": False,
            "cost_time": float('inf'),
            "surprise": 1.0,
            "output": "",
            "patch": ""
        }
        
        # Read original source for patching
        full_path = os.path.join(self.original_repo_path, target_file)
        if not os.path.exists(full_path):
            return result
            
        with open(full_path, 'r') as f:
            original_source = f.read()

        # Only generate patch if there's a change
        if new_source and new_source != original_source:
            result["patch"] = self.generate_patch(original_source, new_source, target_file)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dest_path = os.path.join(tmpdir, "repo")
            shutil.copytree(
                self.original_repo_path, 
                dest_path, 
                ignore=shutil.ignore_patterns('.git', '.venv', '__pycache__', 'data', '*.db')
            )
            
            # Apply patch if exists
            if result["patch"]:
                patch_path = os.path.join(tmpdir, "mutation.patch")
                with open(patch_path, "w") as f:
                    f.write(result["patch"])
                
                # Apply via CLI patch tool
                apply_cmd = ["patch", "-p0", "-i", patch_path]
                patch_proc = subprocess.run(apply_cmd, cwd=dest_path, capture_output=True, text=True)
                
                if patch_proc.returncode != 0:
                    result["output"] = f"Patch application failed: {patch_proc.stderr}"
                    return result
            
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
