"""PhyRE structured benchmark — pure numpy scoring; pymunk only at generation time.

Four conditions:
    flat        — uniform random baseline (expected accuracy ~0.20 for 5 candidates)
    l2l3        — PhyreL2 + PhyreL3 only; score = goal_flag + 0.5 * dist_improvement
    l4_only     — L1+L2+L3+L4GenerativeHead; L4 trains on train pairs, scores by prediction loss
    l4l5_full   — full stack adding L5MetaMonitor adjustment
"""
from __future__ import annotations
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.phyre_sim import load_tasks as _load_tasks
from hpm.encoders.phyre_encoders import PhyreL1Encoder, PhyreL2Encoder, PhyreL3Encoder


def load_tasks(path: str) -> list:
    return _load_tasks(path)


def _score_candidate_l4(train_pairs, test_init, candidate, l1_enc, l2_enc, l3_enc):
    """Score using L4GenerativeHead: train on train pairs, predict L3 for candidate."""
    from hpm.agents.l4_generative import L4GenerativeHead
    final = candidate["final"]

    l4 = L4GenerativeHead(feature_dim_in=14, feature_dim_out=12)

    for pair in train_pairs:
        tr_l2 = l2_enc.encode((pair["init"], pair["final"]), epistemic=None)[0]
        tr_l3 = l3_enc.encode((pair["init"], pair["final"], 0), epistemic=None)[0]
        l4.accumulate(tr_l2, tr_l3)

    l4.fit()

    cand_l2 = l2_enc.encode((test_init, final), epistemic=None)[0]
    cand_l3 = l3_enc.encode((test_init, final, 0), epistemic=None)[0]

    pred_l3 = l4.predict(cand_l2)
    if pred_l3 is None:
        # fallback: use goal flag
        return float(cand_l3[4])

    # Score = negative prediction error (higher is better)
    error = float(np.linalg.norm(pred_l3 - cand_l3))
    return -error


def _score_candidate(train_pairs, test_init, candidate, condition,
                     l1_enc, l2_enc, l3_enc):
    """Return a scalar score for one candidate under the given condition."""
    final = candidate["final"]

    if condition == "flat":
        return float(np.random.uniform())

    l1_vec = l1_enc.encode((test_init, final), epistemic=None)[0]
    l2_vec = l2_enc.encode((test_init, final), epistemic=None)[0]
    l3_vec = l3_enc.encode((test_init, final, 0), epistemic=None)[0]

    if condition == "l2l3":
        goal_flag = float(l3_vec[4])
        dist_improvement = float(l1_vec[12]) - float(l1_vec[13])
        return goal_flag + 0.5 * dist_improvement

    # l4_only or l4l5_full
    l4_score = _score_candidate_l4(train_pairs, test_init, candidate, l1_enc, l2_enc, l3_enc)

    if condition == "l4_only":
        return l4_score

    # l4l5_full: apply L5MetaMonitor strategic confidence as a scaling factor
    from hpm.agents.l5_monitor import L5MetaMonitor
    l5 = L5MetaMonitor()
    # Use epistemic info from L2 (last two dims are epi_weight, epi_loss)
    epi_weight = float(l2_vec[-2])
    epi_loss = float(l2_vec[-1])
    # Update monitor with a proxy surprise value (epi_loss if available, else 0)
    if epi_loss > 0:
        # simulate an l4 prediction to drive L5
        l5.update(np.zeros(12, dtype=np.float32), l3_vec)
    gamma = l5.strategic_confidence()
    return gamma * l4_score + (1.0 - gamma) * float(l3_vec[4])


def run_benchmark(tasks: list, condition: str) -> float:
    """Evaluate all tasks under condition; return accuracy in [0.0, 1.0]."""
    assert condition in ("flat", "l2l3", "l4_only", "l4l5_full")
    l1_enc = PhyreL1Encoder()
    l2_enc = PhyreL2Encoder()
    l3_enc = PhyreL3Encoder()
    np.random.seed(0)
    correct = 0
    for task in tasks:
        test = task["test"]
        test_init = test["init"]
        scores = [
            _score_candidate(task["train"], test_init, c, condition,
                             l1_enc, l2_enc, l3_enc)
            for c in test["candidates"]
        ]
        if int(np.argmax(scores)) == test["correct_idx"]:
            correct += 1
    return correct / len(tasks) if tasks else 0.0


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run PhyRE structured benchmark")
    parser.add_argument("--tasks", default="data/phyre_tasks.pkl")
    args = parser.parse_args()
    print(f"Loading tasks from {args.tasks}...")
    tasks = load_tasks(args.tasks)
    print(f"Loaded {len(tasks)} tasks\n")
    conditions = ["flat", "l2l3", "l4_only", "l4l5_full"]
    print(f"{'Condition':<14} {'Accuracy':>8}")
    print("-" * 24)
    for cond in conditions:
        acc = run_benchmark(tasks, cond)
        print(f"{cond:<14} {acc:>8.3f}")


if __name__ == "__main__":
    main()
