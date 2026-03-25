"""Smoke test for the ARC completion comparison adapter."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).parent.parent.parent)


def test_arc_completion_compare_smoke():
    result = subprocess.run(
        [sys.executable, "benchmarks/arc_completion_compare.py", "--max-tasks", "2"],
        capture_output=True,
        text=True,
        timeout=240,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"Benchmark failed:\n{result.stderr}"
    assert "ARC completion comparison" in result.stdout
    assert "Baseline:" in result.stdout
    assert "Completion:" in result.stdout
    assert "Delta:" in result.stdout
