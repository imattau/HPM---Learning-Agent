"""SWE-bench Verified dataset preflight for structured-agent benchmarking.

This module validates access to the verified dataset and extracts a small set of
metadata summaries that the future execution harness can use.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset

SWE_BENCH_VERIFIED_DATASET_ENV = "SWE_BENCH_VERIFIED_DATASET"
SWE_BENCH_VERIFIED_DEFAULT = "R2E-Gym/SWE-Bench-Verified"


def load_instances(split: str = "test", limit: int | None = None, dataset_name: str | None = None) -> list[dict]:
    name = dataset_name or os.environ.get(SWE_BENCH_VERIFIED_DATASET_ENV, SWE_BENCH_VERIFIED_DEFAULT)
    ds = load_dataset(name, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(row) for row in ds]


def summarize_instances(instances: list[dict]) -> dict[str, object]:
    repos = Counter()
    lengths = []
    file_counts = []
    for item in instances:
        repo = item.get("repo") or item.get("repository") or "unknown"
        repos[repo] += 1
        text = item.get("problem_statement") or item.get("text") or ""
        lengths.append(len(text))
        files = item.get("files_changed") or item.get("patch") or []
        if isinstance(files, list):
            file_counts.append(len(files))
    return {
        "instances": len(instances),
        "repositories": len(repos),
        "top_repos": repos.most_common(5),
        "mean_problem_statement_chars": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "mean_files_changed": (sum(file_counts) / len(file_counts)) if file_counts else 0.0,
    }


def run(split: str = "test", limit: int | None = 10, dataset_name: str | None = None) -> dict[str, object]:
    instances = load_instances(split=split, limit=limit, dataset_name=dataset_name)
    summary = summarize_instances(instances)
    summary["split"] = split
    summary["dataset_name"] = dataset_name or os.environ.get(SWE_BENCH_VERIFIED_DATASET_ENV, SWE_BENCH_VERIFIED_DEFAULT)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SWE-bench Verified preflight summary")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dataset-name", default=None)
    args = parser.parse_args()

    result = run(split=args.split, limit=args.limit, dataset_name=args.dataset_name)
    print(f"Dataset: {result['dataset_name']}")
    print(f"Split: {result['split']}")
    print(f"Instances: {result['instances']}")
    print(f"Repositories: {result['repositories']}")
    print(f"Mean problem statement chars: {result['mean_problem_statement_chars']:.1f}")
    print(f"Mean files changed: {result['mean_files_changed']:.1f}")
    print(f"Top repos: {result['top_repos']}")


if __name__ == "__main__":
    main()
