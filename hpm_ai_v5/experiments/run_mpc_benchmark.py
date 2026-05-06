"""Run the Multi-Pattern Composition (MPC) Benchmark."""

from __future__ import annotations

from hpm_ai_v5.planning.mpc import MultiPatternCompositionBenchmark
from hpm_ai_v5.core import Pattern, PatternManager


def setup_mock_archive(manager: PatternManager):
    """Seed the archive with the invariants learned in the CLT benchmark."""
    # Universal nodes already mapped in CLT
    invariants = [
        ("U_RESOURCE", (1.0,)), # Simplified IDs
        ("U_NULL", (2.0,)),
        ("U_THROW", (3.0,)),
        ("MAP_TRANSFORM", (4.0,)),
        ("FILTER_TRANSFORM", (5.0,))
    ]
    for name, template in invariants:
        p = Pattern(name=name, template=template)
        p.utility = 1.0
        p.support = 10
        manager.archive[name] = p


def run_mpc_benchmark():
    # We use a shared manager to simulate 'long-term memory' from CLT
    manager = PatternManager()
    setup_mock_archive(manager)
    
    benchmark = MultiPatternCompositionBenchmark(pattern_manager=manager)
    
    results = []
    
    # Task 1: Safe File Read (Resource + Null Check + Exception)
    results.append(benchmark.run_composition_task(
        "Safe File Read", ["safe_file_read"], "java"
    ))
    
    # Task 2: Map with Filter
    results.append(benchmark.run_composition_task(
        "Map with Filter", ["map_with_filter"], "java"
    ))
    
    # Task 3: Retry Wrapper (Loop + Exception + Delay)
    results.append(benchmark.run_composition_task(
        "Retry Wrapper", ["retry_wrapper"], "go"
    ))
    
    print("\nMPC Benchmark Results:")
    for res in results:
        print(f"{res.task}: {res.status} (Score: {res.score:.2f})")


if __name__ == "__main__":
    run_mpc_benchmark()
