"""Smoke test for SP16 Geometric Rosetta Benchmark."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).parent.parent.parent)

def test_geometric_rosetta_smoke():
    """Benchmark runs end-to-end and produces correct output."""
    result = subprocess.run(
        [sys.executable, "benchmarks/rosetta_geometric_benchmark.py"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"Benchmark failed:\n{result.stderr}"
    assert "SP16 Geometric Rosetta — Concept-Driven Discovery" in result.stdout
    assert "L5 Surprise" in result.stdout
    assert "SUCCESS ✅" in result.stdout
