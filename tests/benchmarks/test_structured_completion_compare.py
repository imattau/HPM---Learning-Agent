"""Smoke test for structured completion comparison benchmark."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).parent.parent.parent)


def _run_and_assert(*args: str):
    result = subprocess.run(
        [sys.executable, "benchmarks/structured_completion_compare.py", *args],
        capture_output=True,
        text=True,
        timeout=240,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"Benchmark failed:\n{result.stderr}"
    assert "baseline" in result.stdout
    assert "completion" in result.stdout
    assert "SeedStable" in result.stdout
    assert "trace_completeness" in result.stdout
    return result.stdout


def test_benchmark_smoke_normal():
    _run_and_assert("--smoke")


def test_benchmark_smoke_hard():
    out = _run_and_assert("--smoke", "--difficulty", "hard")
    assert "Hard-mode settings" in out
