"""Smoke test for SP14 L4/L5 Linguistic Register Shift Benchmark."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).parent.parent.parent)

def test_benchmark_linguistic_v2_smoke():
    """Benchmark runs end-to-end with --smoke flag."""
    result = subprocess.run(
        [sys.executable, "benchmarks/structured_linguistic_l4l5.py", "--smoke"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"Benchmark failed:\n{result.stderr}"
    assert "l2l3" in result.stdout
    assert "l4_only" in result.stdout
    assert "l4l5_full" in result.stdout
    assert "SP14 Linguistic Register Shift" in result.stdout
