"""SP8: Cross-Task L4 Generalisation benchmark.

Trains a single global L4GenerativeHead on 80% of tasks (seen tasks), then
evaluates it on the remaining 20% (held-out tasks) without any per-task fitting.

Conditions reported:
    flat          — random baseline
    l2l3          — material properties + physics law scoring (no L4)
    per_task_l4   — L4 trained per-task on train pairs (SP7 baseline)
    cross_task_l4 — L4 trained globally on 192 seen tasks, evaluated on 48 held-out
"""
from __future__ import annotations

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hpm.agents.l4_generative import L4GenerativeHead
from hpm.encoders.phyre_encoders import PhyreL1Encoder, PhyreL2Encoder, PhyreL3Encoder
from benchmarks.phyre_sim import load_tasks as _load_tasks


def load_tasks(path: str) -> list:
    return _load_tasks(path)


def fit_global_l4(
    seen_tasks: list,
    l2_enc: PhyreL2Encoder,
    l3_enc: PhyreL3Encoder,
) -> L4GenerativeHead:
    """Fit a single L4GenerativeHead on all train pairs from seen_tasks.

    Accumulates every (L2_vec, L3_vec) pair from every training pair of every
    seen task, then calls .fit() once. Returns the fitted head.
    """
    head = L4GenerativeHead(feature_dim_in=14, feature_dim_out=12)
    for task in seen_tasks:
        for pair in task["train"]:
            l2_vec = l2_enc.encode((pair["init"], pair["final"]), epistemic=None)[0]
            l3_vec = l3_enc.encode((pair["init"], pair["final"], 0), epistemic=None)[0]
            head.accumulate(l2_vec, l3_vec)
    head.fit()
    return head


def score_cross_task_l4(
    task: dict,
    global_l4: L4GenerativeHead,
    l2_enc: PhyreL2Encoder,
    l3_enc: PhyreL3Encoder,
) -> int:
    """Score a held-out task using the pre-fitted global L4 head.

    Returns the index of the candidate with the highest score
    (i.e. lowest L3 prediction error).
    """
    test = task["test"]
    test_init = test["init"]
    scores = []
    for candidate in test["candidates"]:
        final = candidate["final"]
        cand_l2 = l2_enc.encode((test_init, final), epistemic=None)[0]
        cand_l3 = l3_enc.encode((test_init, final, 0), epistemic=None)[0]
        pred_l3 = global_l4.predict(cand_l2)
        if pred_l3 is None:
            # Fallback to goal flag if head not fitted
            scores.append(float(cand_l3[4]))
        else:
            error = float(np.linalg.norm(pred_l3 - cand_l3))
            scores.append(-error)
    return int(np.argmax(scores))


def _score_flat(task: dict) -> int:
    """Random baseline."""
    return int(np.random.randint(0, len(task["test"]["candidates"])))


def _score_l2l3(task: dict, l1_enc: PhyreL1Encoder, l3_enc: PhyreL3Encoder) -> int:
    """L2L3 condition: goal flag + distance improvement."""
    test = task["test"]
    test_init = test["init"]
    scores = []
    for candidate in test["candidates"]:
        final = candidate["final"]
        l1_vec = l1_enc.encode((test_init, final), epistemic=None)[0]
        l3_vec = l3_enc.encode((test_init, final, 0), epistemic=None)[0]
        goal_flag = float(l3_vec[4])
        dist_improvement = float(l1_vec[12]) - float(l1_vec[13])
        scores.append(goal_flag + 0.5 * dist_improvement)
    return int(np.argmax(scores))


def _score_per_task_l4(
    task: dict,
    l2_enc: PhyreL2Encoder,
    l3_enc: PhyreL3Encoder,
) -> int:
    """Per-task L4: train on this task's train pairs only."""
    head = L4GenerativeHead(feature_dim_in=14, feature_dim_out=12)
    for pair in task["train"]:
        l2_vec = l2_enc.encode((pair["init"], pair["final"]), epistemic=None)[0]
        l3_vec = l3_enc.encode((pair["init"], pair["final"], 0), epistemic=None)[0]
        head.accumulate(l2_vec, l3_vec)
    head.fit()

    test = task["test"]
    test_init = test["init"]
    scores = []
    for candidate in test["candidates"]:
        final = candidate["final"]
        cand_l2 = l2_enc.encode((test_init, final), epistemic=None)[0]
        cand_l3 = l3_enc.encode((test_init, final, 0), epistemic=None)[0]
        pred_l3 = head.predict(cand_l2)
        if pred_l3 is None:
            scores.append(float(cand_l3[4]))
        else:
            scores.append(-float(np.linalg.norm(pred_l3 - cand_l3)))
    return int(np.argmax(scores))


def run_benchmark(
    tasks: list,
    seed: int = 42,
    train_frac: float = 0.8,
) -> dict[str, float]:
    """Run all four conditions and return accuracy dict.

    Args:
        tasks: Full list of task dicts (240 expected).
        seed: RNG seed for task shuffle and flat condition.
        train_frac: Fraction of tasks used for global L4 fitting.

    Returns:
        Dict mapping condition name -> accuracy in [0, 1].
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(tasks))
    rng.shuffle(indices)
    n_seen = int(len(tasks) * train_frac)
    seen_indices = indices[:n_seen]
    held_out_indices = indices[n_seen:]

    seen_tasks = [tasks[i] for i in seen_indices]
    held_out_tasks = [tasks[i] for i in held_out_indices]

    l1_enc = PhyreL1Encoder()
    l2_enc = PhyreL2Encoder()
    l3_enc = PhyreL3Encoder()

    # Fit global L4 on seen tasks
    global_l4 = fit_global_l4(seen_tasks, l2_enc, l3_enc)

    np.random.seed(seed)

    results: dict[str, float] = {}

    # flat — evaluated on held-out tasks only (random, so all tasks equivalent)
    correct_flat = sum(
        1 for t in held_out_tasks
        if _score_flat(t) == t["test"]["correct_idx"]
    )
    results["flat"] = correct_flat / len(held_out_tasks) if held_out_tasks else 0.0

    # l2l3 — on held-out tasks
    correct_l2l3 = sum(
        1 for t in held_out_tasks
        if _score_l2l3(t, l1_enc, l3_enc) == t["test"]["correct_idx"]
    )
    results["l2l3"] = correct_l2l3 / len(held_out_tasks) if held_out_tasks else 0.0

    # per_task_l4 — on held-out tasks (uses only each task's own train pairs)
    correct_per = sum(
        1 for t in held_out_tasks
        if _score_per_task_l4(t, l2_enc, l3_enc) == t["test"]["correct_idx"]
    )
    results["per_task_l4"] = correct_per / len(held_out_tasks) if held_out_tasks else 0.0

    # cross_task_l4 — global L4 fitted on seen tasks, evaluated on held-out
    correct_cross = sum(
        1 for t in held_out_tasks
        if score_cross_task_l4(t, global_l4, l2_enc, l3_enc) == t["test"]["correct_idx"]
    )
    results["cross_task_l4"] = correct_cross / len(held_out_tasks) if held_out_tasks else 0.0

    return results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="SP8: Cross-Task L4 benchmark")
    parser.add_argument("--tasks", default="data/phyre_tasks.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    print(f"Loading tasks from {args.tasks}...")
    tasks = load_tasks(args.tasks)
    n_seen = int(len(tasks) * args.train_frac)
    n_held = len(tasks) - n_seen
    print(f"Loaded {len(tasks)} tasks ({n_seen} seen, {n_held} held-out)\n")

    results = run_benchmark(tasks, seed=args.seed, train_frac=args.train_frac)

    print(f"SP8 Cross-Task L4 Benchmark ({n_held} held-out tasks, {n_seen} seen)")
    print(f"{'Condition':<18} {'Accuracy':>8}")
    print("-" * 28)
    for condition, acc in results.items():
        print(f"{condition:<18} {acc:>8.3f}")

    # Interpretation
    print()
    if results["cross_task_l4"] > results["per_task_l4"]:
        print("RESULT: cross_task_l4 > per_task_l4 — hypothesis VALIDATED")
    else:
        print("RESULT: cross_task_l4 <= per_task_l4 — hypothesis NOT validated")

    if results["cross_task_l4"] >= results["l2l3"]:
        print("BONUS: cross_task_l4 >= l2l3 — strong result (genuine cross-task abstraction)")


if __name__ == "__main__":
    main()
