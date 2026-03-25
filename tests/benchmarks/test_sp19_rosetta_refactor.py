"""Smoke test for SP19 Rosetta Refactor Benchmark."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).parent.parent.parent)

def test_rosetta_refactor_smoke():
    """Benchmark runs end-to-end and produces correct output."""
    result = subprocess.run(
        [sys.executable, "benchmarks/sp19_rosetta_refactor.py"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"Benchmark failed:\n{result.stderr}"
    assert "SP19: The Rosetta Refactor" in result.stdout
    assert "RESULT: SUCCESS ✅" in result.stdout
