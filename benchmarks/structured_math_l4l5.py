"""SP6 benchmark: L4/L5 Math.

Three conditions run on the same structured math tasks (seed=42):
  - l2l3:      Baseline — L3 NLL scoring only, no L4/L5 (same as 'l2_l3' in structured_math.py)
  - l4_only:   L4 intuition always trusted (gamma = 1.0 fixed, no adaptive gating)
  - l4l5_full: Adaptive gamma from L5MetaMonitor

Usage:
    python3 benchmarks/structured_math_l4l5.py              # full run (n_per_family=60)
    python3 benchmarks/structured_math_l4l5.py --smoke      # 2 tasks per family
    python3 benchmarks/structured_math_l4l5.py --n_tasks 3  # synonym for smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sympy

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.structured_math import generate_tasks  # reuse existing task generation
from hpm.agents.l4_generative import L4GenerativeHead
from hpm.agents.l5_monitor import L5MetaMonitor
from hpm.encoders.math_encoders import MathL2Encoder, MathL3Encoder

# Encoder feature dims: L2=10, L3=12
_L2_DIM = 10
_L3_DIM = 12


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
    seed: int = 42,
) -> dict[str, float]:
    """Run one condition and return accuracy metrics.

    All conditions use L2+L3 for the base prototype. L4/L5 conditions additionally
    learn a ridge regressor (L4) that predicts L3 from L2, gated by L5 surprise.

    condition:
        l2l3      - baseline: pure L3-NLL scoring (no L4/L5)
        l4_only   - L4 ridge prediction, gamma=1.0 (full L4 trust, no adaptive gating)
        l4l5_full - L4 + L5 adaptive gamma
    """
    l2_enc = MathL2Encoder()
    l3_enc = MathL3Encoder()

    correct = 0
    total = 0

    for task in tasks:
        train = task['train']
        test_input = task['test_input']
        test_output = task['test_output']
        candidates = task['candidates']

        # --- Build L2/L3 prototypes from training pairs ---
        l2_train_vecs = []
        l3_train_vecs = []
        mean_l2_nll = 0.0

        for pair in train:
            l2_vec = _encode_vec(l2_enc, pair, epistemic=None)
            l2_train_vecs.append(l2_vec)

        if l2_train_vecs:
            l2_prototype = np.mean(l2_train_vecs, axis=0)
            # Compute mean L2 NLL for epistemic threading
            l2_nlls = [_l3_nll(v, l2_prototype) for v in l2_train_vecs]
            mean_l2_nll = float(np.mean(l2_nlls))
        else:
            l2_prototype = np.zeros(_L2_DIM)

        epistemic_l2 = (1.0, mean_l2_nll)

        for pair in train:
            l3_vec = _encode_vec(l3_enc, pair, epistemic=epistemic_l2)
            l3_train_vecs.append(l3_vec)

        l3_prototype = np.mean(l3_train_vecs, axis=0) if l3_train_vecs else np.zeros(_L3_DIM)

        # --- L4/L5 setup (per task) ---
        if condition in ('l4_only', 'l4l5_full'):
            head = L4GenerativeHead(feature_dim_in=_L2_DIM, feature_dim_out=_L3_DIM)
            monitor = L5MetaMonitor() if condition == 'l4l5_full' else None

            # Accumulate training pairs into L4 (and update L5 surprise)
            for i, pair in enumerate(train):
                l2_vec = l2_train_vecs[i] if i < len(l2_train_vecs) else _encode_vec(l2_enc, pair, epistemic=None)
                l3_vec = l3_train_vecs[i] if i < len(l3_train_vecs) else _encode_vec(l3_enc, pair, epistemic=epistemic_l2)
                head.accumulate(l2_vec, l3_vec)
                if monitor is not None:
                    l4_pred = head.predict(l2_vec)  # None until >= 2 pairs
                    monitor.update(l4_pred, l3_vec)

            # Fit L4 on training data
            head.fit()

            # Determine gamma
            if condition == 'l4_only':
                gamma = 1.0  # fixed: always trust L4
            else:
                gamma = monitor.strategic_confidence()  # adaptive from L5
        else:
            head = None
            gamma = 0.0

        # --- Score candidates ---
        scores = []
        for cand in candidates:
            obs = (test_input, cand)
            l2_cand = _encode_vec(l2_enc, obs, epistemic=None)
            l3_cand = _encode_vec(l3_enc, obs, epistemic=epistemic_l2)

            l3_nll = _l3_nll(l3_cand, l3_prototype)

            if condition == 'l2l3':
                # Baseline: pure L2 + L3 NLL
                l2_nll = _l3_nll(l2_cand, l2_prototype)
                score = l2_nll + l3_nll
            else:
                # L4/L5: interpolate between L4 prediction and L3 prototype NLL
                l4_pred = head.predict(l2_cand)
                if l4_pred is None:
                    # L4 not fitted (< 2 training pairs): fall back to L3 NLL
                    score = l3_nll
                else:
                    cos_d = _cosine_dist(l3_cand, l4_pred)
                    score = gamma * cos_d + (1.0 - gamma) * l3_nll

            scores.append(score)

        # Pick candidate with lowest score
        predicted_idx = int(np.argmin(scores))
        predicted = candidates[predicted_idx]

        try:
            if sympy.simplify(predicted - test_output) == 0:
                correct += 1
        except Exception:
            if predicted == test_output:
                correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(description="SP6 L4/L5 Math Benchmark")
    parser.add_argument("--smoke", action="store_true", help="Run 2 tasks per family (fast)")
    parser.add_argument("--n_tasks", type=int, default=0, help="Tasks per family (0=full)")
    parser.add_argument("--n_per_family", type=int, default=60, help="Tasks per family (full run)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.smoke or args.n_tasks > 0:
        n_per_family = args.n_tasks if args.n_tasks > 0 else 2
    else:
        n_per_family = args.n_per_family

    all_tasks = generate_tasks(n_per_family=n_per_family, seed=args.seed)

    conditions = ["l2l3", "l4_only", "l4l5_full"]
    print(f"\nSP6 L4/L5 Math Benchmark — {len(all_tasks)} tasks, seed={args.seed}")
    print(f"{'Condition':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    print("-" * 52)

    for cond in conditions:
        metrics = run_condition(all_tasks, condition=cond, seed=args.seed)
        print(
            f"{cond:<20} {metrics['accuracy']:>10.3f}"
            f" {metrics['correct']:>10} {metrics['total']:>8}"
        )

    print()


if __name__ == "__main__":
    main()
