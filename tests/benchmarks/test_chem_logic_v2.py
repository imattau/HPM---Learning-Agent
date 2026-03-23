"""Smoke test for SP13 L4/L5 Chem-Logic II Benchmark."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).parent.parent.parent)

def test_benchmark_v2_smoke():
    """Benchmark runs end-to-end with --smoke flag."""
    result = subprocess.run(
        [sys.executable, "benchmarks/structured_chem_logic_v2.py", "--smoke"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"Benchmark failed:\n{result.stderr}"
    assert "l2l3" in result.stdout
    assert "l4_only" in result.stdout
    assert "l4l5_full" in result.stdout
    assert "SP13 Chem-Logic II" in result.stdout
