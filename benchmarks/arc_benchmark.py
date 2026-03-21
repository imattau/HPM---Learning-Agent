"""
Benchmark 4: ARC (Abstraction and Reasoning Corpus)
=====================================================
Evaluates HPM on all 400 ARC training tasks using 5-way discrimination.

Protocol:
  - Train agent on 3-5 (input, output) grid pairs (10 steps each)
  - Test: identify correct output from 4 distractors drawn from other tasks
  - Metric: accuracy vs 20% chance baseline

Run:
    python benchmarks/arc_benchmark.py

Note: Downloads ARC dataset from HuggingFace on first run (~5MB, cached).
Sign convention: GaussianPattern.log_prob returns NLL (lower = more probable).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from benchmarks.common import make_agent, print_results_table

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_GRID_DIM = 20                         # max rows/cols; tasks exceeding this are excluded
GRID_SIZE = MAX_GRID_DIM * MAX_GRID_DIM   # 400
FEATURE_DIM = GRID_SIZE * 2               # 200: encode(input) + encode(output)
TRAIN_REPS = 10                           # agent.step() calls per training pair
N_DISTRACTORS = 4                         # distractor outputs per test


def encode_grid(grid: list[list[int]]) -> np.ndarray:
    """
    Encode a 2D integer grid (values 0-9) to a float vector of dim GRID_SIZE.
    Flattens row-major, pads with 0 to MAX_GRID_DIM x MAX_GRID_DIM, divides by 9.
    """
    flat = []
    for row in grid:
        for val in row:
            flat.append(float(val) / 9.0)
    # Pad to GRID_SIZE
    flat.extend([0.0] * (GRID_SIZE - len(flat)))
    return np.array(flat[:GRID_SIZE], dtype=float)


def encode_pair(input_grid: list[list[int]], output_grid: list[list[int]]) -> np.ndarray:
    """Concatenate encoded input and output into a FEATURE_DIM observation vector."""
    return np.concatenate([encode_grid(input_grid), encode_grid(output_grid)])


def grid_fits(grid: list[list[int]]) -> bool:
    """Return True if grid is within MAX_GRID_DIM x MAX_GRID_DIM."""
    if len(grid) > MAX_GRID_DIM:
        return False
    return all(len(row) <= MAX_GRID_DIM for row in grid)


def task_fits(task: dict) -> bool:
    """Return True if all grids in train and test pairs fit within MAX_GRID_DIM."""
    for pair in task["train"]:
        if not grid_fits(pair["input"]) or not grid_fits(pair["output"]):
            return False
    for pair in task["test"]:
        if not grid_fits(pair["input"]) or not grid_fits(pair["output"]):
            return False
    return True


def load_tasks() -> list[dict]:
    """
    Load all ARC training tasks from HuggingFace.
    Returns list of task dicts with 'train' and 'test' keys.
    Downloads on first call (~5MB), cached in ~/.cache/huggingface/.

    Dataset: lordspline/arc-agi (mirror of fchollet/ARC-AGI, split='training').
    Each item has 'train': list of {input, output} dicts and 'test': same.
    """
    from datasets import load_dataset
    ds = load_dataset("lordspline/arc-agi", split="training")
    tasks = []
    for item in ds:
        tasks.append({
            "train": item["train"],
            "test": item["test"],
        })
    return tasks


def score_candidate(agent, vec: np.ndarray) -> float:
    """
    Compute ensemble NLL score for a candidate vector.
    score = sum(w_i * pattern_i.log_prob(vec))
    Lower score = more probable under the ensemble.
    Returns 0.0 if store is empty (tie — correct loses by conservative rule).
    """
    records = agent.store.query(agent.agent_id)
    if not records:
        return 0.0
    return sum(w * p.log_prob(vec) for p, w in records)


def make_arc_agent():
    """Fresh HPM agent configured for ARC (feature_dim=200, beta_comp=0.0)."""
    return make_agent(
        feature_dim=FEATURE_DIM,
        agent_id="arc_bench",
        beta_comp=0.0,
        T_recomb=5,
        recomb_cooldown=3,
        init_sigma=2.0,
    )


def evaluate_task(task: dict, all_tasks: list[dict], task_idx: int) -> tuple[bool, int]:
    """
    Run one ARC task.

    Returns:
        (correct, rank): correct=True if agent picks right output; rank in [1,5]
        (rank 1 = lowest NLL = most probable; ties broken against correct).

    Protocol:
    1. Train: for each training pair, call agent.step(encode_pair(...)) TRAIN_REPS times
    2. Evaluate: score correct output vs N_DISTRACTORS from other tasks
    3. Correct if correct candidate has strictly lowest NLL score
    """
    agent = make_arc_agent()

    # Training phase
    for pair in task["train"]:
        obs = encode_pair(pair["input"], pair["output"])
        for _ in range(TRAIN_REPS):
            agent.step(obs)

    # Evaluation phase: build candidate set
    test_pair = task["test"][0]
    test_input = test_pair["input"]
    correct_output = test_pair["output"]

    correct_vec = encode_pair(test_input, correct_output)

    # Sample N_DISTRACTORS other tasks (exclude self; skip tasks with no train pairs)
    rng = np.random.default_rng(task_idx)
    other_indices = [j for j in range(len(all_tasks)) if j != task_idx and all_tasks[j]["train"]]
    distractor_indices = rng.choice(other_indices, size=N_DISTRACTORS, replace=False)
    distractor_vecs = [
        encode_pair(test_input, all_tasks[j]["train"][0]["output"])
        for j in distractor_indices
    ]

    # Score all candidates (lower NLL = more probable)
    correct_score = score_candidate(agent, correct_vec)
    distractor_scores = [score_candidate(agent, v) for v in distractor_vecs]

    # Rank: count how many distractors score <= correct (ties go against correct)
    rank = 1 + sum(1 for s in distractor_scores if s <= correct_score)
    correct = correct_score < min(distractor_scores)

    return correct, rank


def main():
    print("Loading ARC dataset...", flush=True)
    all_tasks = load_tasks()

    eligible = [(i, t) for i, t in enumerate(all_tasks) if task_fits(t)]
    excluded = len(all_tasks) - len(eligible)

    print(
        f"Tasks: {len(all_tasks)} total, {excluded} excluded "
        f"(grid > {MAX_GRID_DIM}x{MAX_GRID_DIM}), {len(eligible)} evaluated"
    )
    print("Running...", flush=True)

    correct_count = 0
    rank_sum = 0

    for run_idx, (task_idx, task) in enumerate(eligible):
        if (run_idx + 1) % 50 == 0:
            pct = correct_count / (run_idx + 1) * 100
            print(f"  {run_idx + 1}/{len(eligible)} tasks — accuracy so far: {pct:.1f}%", flush=True)

        correct, rank = evaluate_task(task, all_tasks, task_idx)
        if correct:
            correct_count += 1
        rank_sum += rank

    n = len(eligible)
    accuracy = correct_count / n * 100 if n > 0 else 0.0
    mean_rank = rank_sum / n if n > 0 else 3.0
    vs_chance = accuracy - 20.0

    assert 0.0 <= accuracy <= 100.0
    assert 1.0 <= mean_rank <= 5.0

    print_results_table(
        title="ARC Benchmark (5-way discrimination, fchollet/arc-agi train split)",
        cols=["Tasks Run", "Excluded", "Correct", "Accuracy", "Mean Rank", "vs Chance"],
        rows=[{
            "Tasks Run": str(n),
            "Excluded": str(excluded),
            "Correct": str(correct_count),
            "Accuracy": f"{accuracy:.1f}%",
            "Mean Rank": f"{mean_rank:.2f}",
            "vs Chance": f"{vs_chance:+.1f}%",
        }],
    )


if __name__ == "__main__":
    main()
