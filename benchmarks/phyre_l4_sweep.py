"""Sweep n_train_pairs for L4GenerativeHead on PhyRE benchmark.

Regenerates tasks with n_train_pairs in [3, 5, 10, 20] and runs
l4_only and l2l3 conditions for each. Prints a comparison table.
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.phyre_sim import generate_family_tasks
from benchmarks.structured_phyre import run_benchmark

FAMILIES = ["Projectile", "Bounce", "Slide", "Collision"]
N_TASKS_PER_FAMILY = 60
SEED = 42
N_TRAIN_VALUES = [3, 5, 10, 20]
CONDITIONS = ["flat", "l2l3", "l4_only", "l4l5_full"]


def generate_all_tasks(n_train_pairs: int) -> list:
    tasks = []
    for family in FAMILIES:
        family_tasks = generate_family_tasks(
            family,
            n_tasks=N_TASKS_PER_FAMILY,
            seed=SEED,
            n_train_pairs=n_train_pairs,
        )
        tasks.extend(family_tasks)
    return tasks


def main():
    results = {}

    for n_train in N_TRAIN_VALUES:
        print(f"\nGenerating tasks with n_train_pairs={n_train}...")
        tasks = generate_all_tasks(n_train)
        actual_train_counts = [len(t["train"]) for t in tasks]
        print(f"  Generated {len(tasks)} tasks, avg train pairs: "
              f"{sum(actual_train_counts)/len(actual_train_counts):.1f}, "
              f"min: {min(actual_train_counts)}")

        row = {}
        for cond in CONDITIONS:
            print(f"  Running condition: {cond}...", end="", flush=True)
            acc = run_benchmark(tasks, cond)
            row[cond] = acc
            print(f" {acc:.3f}")
        results[n_train] = row

    # Print summary table
    print("\n" + "=" * 65)
    print("Results Table: Accuracy by n_train_pairs and condition")
    print("=" * 65)
    header = f"{'n_train':>8} | {'flat':>7} | {'l2l3':>7} | {'l4_only':>8} | {'l4l5_full':>10}"
    print(header)
    print("-" * 65)
    for n_train in N_TRAIN_VALUES:
        row = results[n_train]
        print(f"{n_train:>8} | {row['flat']:>7.3f} | {row['l2l3']:>7.3f} | "
              f"{row['l4_only']:>8.3f} | {row['l4l5_full']:>10.3f}")
    print("=" * 65)

    # Print delta vs n_train=3
    if 3 in results:
        print("\nDelta vs n_train=3 (l4_only):")
        base = results[3]["l4_only"]
        for n_train in N_TRAIN_VALUES:
            delta = results[n_train]["l4_only"] - base
            print(f"  n_train={n_train:>2}: l4_only={results[n_train]['l4_only']:.3f}  "
                  f"delta={delta:+.3f}")


if __name__ == "__main__":
    main()
