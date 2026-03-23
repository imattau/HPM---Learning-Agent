"""SP11 benchmark: DS-1000 Boss Fight (5-Level Stack).

Three conditions run on the simulated DS-1000 data science tasks:
  - l2l3:      Baseline — L3 NLL scoring only, no L4/L5
  - l4_only:   L4 generative intuition always trusted (gamma = 1.0 fixed)
  - l4l5_full: Adaptive gamma from L5MetaMonitor based on structural surprise

Usage:
    python3 benchmarks/structured_ds1000_l4l5.py              # full run
    python3 benchmarks/structured_ds1000_l4l5.py --smoke      # fast smoke test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.ds1000_sim import generate_ds1000_tasks
from benchmarks.ds1000_encoders import DS1000L1Encoder, DS1000L2Encoder, DS1000L3Encoder
from hpm.agents.l4_generative import L4GenerativeHead
from hpm.agents.l5_monitor import L5MetaMonitor


def _encode_vec(encoder, pair: tuple, epistemic=None) -> np.ndarray:
    """Encode one (in, out) pair and return mean vector."""
    vecs = encoder.encode(pair, epistemic=epistemic)
    if not vecs:
        return np.zeros(encoder.feature_dim)
    return np.mean(vecs, axis=0)


def _l3_nll(l3_vec: np.ndarray, prototype: np.ndarray) -> float:
    """L2-distance NLL: lower = better match to prototype."""
    return float(np.sum((l3_vec - prototype) ** 2))


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2]. Returns 1.0 if either vector is near-zero."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def run_condition(
    tasks: list[dict],
    condition: str,
) -> dict[str, float]:
    """Run one condition and return accuracy metrics."""
    l1_enc = DS1000L1Encoder()
    l2_enc = DS1000L2Encoder()
    l3_enc = DS1000L3Encoder()

    correct = 0
    total = 0

    for task in tasks:
        train = task['train']
        test_input = task['test_input']
        test_output = task['test_output']
        candidates = task['candidates']

        # --- Build Prototypes ---
        l2_train_vecs = []
        l3_train_vecs = []
        mean_l1_nll = 0.5  # Simulated L1 loss
        epistemic_l1 = (1.0, mean_l1_nll)

        for pair in train:
            obs = (pair["input"], pair["output"])
            l2_vec = _encode_vec(l2_enc, obs, epistemic=epistemic_l1)
            l2_train_vecs.append(l2_vec)

        l2_prototype = np.mean(l2_train_vecs, axis=0) if l2_train_vecs else np.zeros(l2_enc.feature_dim)
        
        # Simulated epistemic state from L2
        l2_nlls = [_l3_nll(v, l2_prototype) for v in l2_train_vecs] if l2_train_vecs else [1.0]
        epistemic_l2 = (1.0, float(np.mean(l2_nlls)))

        for pair in train:
            obs = (pair["input"], pair["output"])
            l3_vec = _encode_vec(l3_enc, obs, epistemic=epistemic_l2)
            l3_train_vecs.append(l3_vec)

        l3_prototype = np.mean(l3_train_vecs, axis=0) if l3_train_vecs else np.zeros(l3_enc.feature_dim)

        # --- L4/L5 Setup ---
        if condition in ('l4_only', 'l4l5_full'):
            head = L4GenerativeHead(feature_dim_in=l2_enc.feature_dim, feature_dim_out=l3_enc.feature_dim)
            monitor = L5MetaMonitor() if condition == 'l4l5_full' else None

            # Accumulate training pairs
            for i, pair in enumerate(train):
                obs = (pair["input"], pair["output"])
                l2_vec = l2_train_vecs[i] if i < len(l2_train_vecs) else _encode_vec(l2_enc, obs, epistemic=epistemic_l1)
                l3_vec = l3_train_vecs[i] if i < len(l3_train_vecs) else _encode_vec(l3_enc, obs, epistemic=epistemic_l2)
                head.accumulate(l2_vec, l3_vec)
                
                if monitor is not None:
                    l4_pred = head.predict(l2_vec)
                    monitor.update(l4_pred, l3_vec)

            head.fit()

            if condition == 'l4_only':
                gamma = 1.0  # Full trust in intuition
            else:
                gamma = monitor.strategic_confidence()
        else:
            head = None
            gamma = 0.0

        # --- Score Candidates ---
        scores = []
        for cand in candidates:
            obs = (test_input, cand)
            l2_cand = _encode_vec(l2_enc, obs, epistemic=epistemic_l1)
            l3_cand = _encode_vec(l3_enc, obs, epistemic=epistemic_l2)

            l3_nll = _l3_nll(l3_cand, l3_prototype)

            if condition == 'l2l3':
                l2_nll = _l3_nll(l2_cand, l2_prototype)
                score = l2_nll + l3_nll
            else:
                l4_pred = head.predict(l2_cand)
                if l4_pred is None:
                    score = l3_nll
                else:
                    cos_d = _cosine_dist(l3_cand, l4_pred)
                    score = gamma * cos_d + (1.0 - gamma) * l3_nll

            scores.append(score)

        predicted_idx = int(np.argmin(scores))
        
        # Test exact array match for simulation correctness
        if np.allclose(candidates[predicted_idx], test_output, atol=1e-5):
            correct += 1
            
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(description="SP11 DS-1000 Benchmark")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    parser.add_argument("--n_per_library", type=int, default=20, help="Tasks per library")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_per_lib = 2 if args.smoke else args.n_per_library
    all_tasks = generate_ds1000_tasks(n_per_library=n_per_lib, seed=args.seed)

    conditions = ["l2l3", "l4_only", "l4l5_full"]
    print(f"\nSP11 DS-1000 Benchmark — {len(all_tasks)} tasks, seed={args.seed}")
    print(f"{'Condition':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    print("-" * 52)

    for cond in conditions:
        metrics = run_condition(all_tasks, condition=cond)
        print(
            f"{cond:<20} {metrics['accuracy']:>10.3f}"
            f" {metrics['correct']:>10} {metrics['total']:>8}"
        )

    print()


if __name__ == "__main__":
    main()
