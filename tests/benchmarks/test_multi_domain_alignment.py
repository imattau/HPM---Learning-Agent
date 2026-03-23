"""Smoke test for SP15 Generalized Cross-Domain Alignment."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).parent.parent.parent)

def test_multi_domain_alignment_smoke():
    """Benchmark runs end-to-end with --smoke flag."""
    result = subprocess.run(
        [sys.executable, "benchmarks/multi_domain_alignment.py", "--smoke"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"Benchmark failed:\n{result.stderr}"
    assert "SP15 Generalized Cross-Domain Transfer" in result.stdout
    assert "SP15 Surprise Transfer" in result.stdout
    assert "Linguistic" in result.stdout
    assert "Chemistry" in result.stdout
