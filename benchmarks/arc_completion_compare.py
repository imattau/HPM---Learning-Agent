
"""ARC comparison benchmark using the existing ARC benchmark as a thin adapter."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import benchmarks.arc_benchmark as arc_bench
from benchmarks.completion_adapter import make_agent_factory, patch_attr

ARC_AGENT_OVERRIDES = dict(
    evaluator_arbitration_mode="adaptive",
    meta_evaluator_learning_rate=0.2,
    lifecycle_decay_rate=0.08,
    lifecycle_consolidation_window=2,
    lifecycle_absence_window=2,
    lifecycle_stable_weight_threshold=0.2,
    lifecycle_retire_weight_threshold=0.04,
)


def _eligible_tasks():
    all_tasks = arc_bench.load_tasks()
    eligible = [(i, t) for i, t in enumerate(all_tasks) if arc_bench.task_fits(t)]
    return all_tasks, eligible


def _run_condition(tasks, all_tasks, condition: str, seed: int):
    metrics = {"correct": 0, "rank_sum": 0}
    base_factory = arc_bench.make_agent
    if condition == "completion":
        base_factory = make_agent_factory(base_factory, seed=seed, overrides=ARC_AGENT_OVERRIDES)

    with patch_attr(arc_bench, "make_agent", base_factory):
        for task_idx, task in tasks:
            correct, rank = arc_bench.evaluate_task(task, all_tasks, task_idx)
            metrics["correct"] += int(correct)
            metrics["rank_sum"] += rank

    n = len(tasks)
    accuracy = (metrics["correct"] / n * 100.0) if n else 0.0
    mean_rank = (metrics["rank_sum"] / n) if n else 3.0
    return {
        "correct": metrics["correct"],
        "accuracy": accuracy,
        "mean_rank": mean_rank,
        "vs_chance": accuracy - 20.0,
    }


def run(max_tasks: int | None = None, seed: int = 42) -> dict[str, dict[str, float]]:
    all_tasks, eligible = _eligible_tasks()
    if max_tasks is not None:
        eligible = eligible[:max_tasks]
    baseline = _run_condition(eligible, all_tasks, "baseline", seed)
    completion = _run_condition(eligible, all_tasks, "completion", seed)
    return {
        "baseline": baseline,
        "completion": completion,
        "delta": {
            "accuracy": completion["accuracy"] - baseline["accuracy"],
            "mean_rank": completion["mean_rank"] - baseline["mean_rank"],
            "vs_chance": completion["vs_chance"] - baseline["vs_chance"],
        },
        "tasks_run": len(eligible),
        "excluded": len(all_tasks) - len(eligible),
    }


def main():
    parser = argparse.ArgumentParser(description="ARC comparison benchmark for baseline vs completion-aware agents")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Running ARC completion comparison...", flush=True)
    result = run(max_tasks=args.max_tasks, seed=args.seed)
    base = result["baseline"]
    comp = result["completion"]

    print(f"Tasks: {result['tasks_run']} evaluated, {result['excluded']} excluded")
    print(f"Baseline:   accuracy={base['accuracy']:.1f}% mean_rank={base['mean_rank']:.2f} vs_chance={base['vs_chance']:+.1f}%")
    print(f"Completion: accuracy={comp['accuracy']:.1f}% mean_rank={comp['mean_rank']:.2f} vs_chance={comp['vs_chance']:+.1f}%")
    print(f"Delta:      accuracy={result['delta']['accuracy']:+.1f} mean_rank={result['delta']['mean_rank']:+.2f} vs_chance={result['delta']['vs_chance']:+.1f}")


if __name__ == "__main__":
    main()
