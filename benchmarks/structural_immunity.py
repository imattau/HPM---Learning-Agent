"""
Benchmark 2: Structural Immunity
==================================
Tests whether pattern density (kappa_D) protects consolidated patterns
during a noise storm and enables clean recovery.

3-phase data stream:
  Phase 1 (500 steps): Gaussian signal, fixed mean, std=0.1
  Phase 2 (20 steps):  Uniform noise in [-1, 1]^d
  Phase 3 (500 steps): Same Gaussian signal as Phase 1

Run:
    python benchmarks/structural_immunity.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from benchmarks.common import make_agent, print_results_table

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURE_DIM = 16
PHASE1_STEPS = 500
PHASE2_STEPS = 20
PHASE3_STEPS = 500
SIGNAL_STD = 0.1
BASELINE_WINDOW = 50        # steps at end of Phase 1 for baseline measurement
ACC_RECOVERY_FRAC = 0.9     # must reach >= 0.9 * B_acc
TOP_WEIGHT_THRESHOLD = 0.5
IMMUNE_THRESHOLD = 100      # T_rec <= 100 => IMMUNE
RNG_SEED = 42


def unit_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def run() -> dict:
    rng = np.random.default_rng(RNG_SEED)

    # Fixed signal mean (random unit vector)
    mu0 = unit_vector(rng, FEATURE_DIM)

    # Use kappa_D > 0 to enable density stickiness
    agent = make_agent(
        feature_dim=FEATURE_DIM,
        agent_id="immunity_bench",
        kappa_D=1.0,
        kappa_d_levels=[0.2, 0.4, 0.6, 0.8, 1.0],
    )

    # -----------------------------------------------------------------------
    # Phase 1: Stable Gaussian signal
    # -----------------------------------------------------------------------
    phase1_accs = []
    baseline_window_accs = []

    for step in range(PHASE1_STEPS):
        obs = rng.normal(loc=mu0, scale=SIGNAL_STD)
        result = agent.step(obs)
        acc = result["mean_accuracy"]
        phase1_accs.append(acc)
        if step >= PHASE1_STEPS - BASELINE_WINDOW:
            baseline_window_accs.append(acc)

    # Baseline measurements at end of Phase 1
    b_acc = float(np.mean(baseline_window_accs))
    records = agent.store.query(agent.agent_id)
    b_weight = max((w for _, w in records), default=0.0)

    # -----------------------------------------------------------------------
    # Phase 2: Uniform noise storm
    # -----------------------------------------------------------------------
    phase2_accs = []
    for _ in range(PHASE2_STEPS):
        obs = rng.uniform(-1.0, 1.0, size=FEATURE_DIM)
        result = agent.step(obs)
        phase2_accs.append(result["mean_accuracy"])

    # -----------------------------------------------------------------------
    # Phase 3: Same Gaussian signal returns — measure recovery
    # -----------------------------------------------------------------------
    phase3_accs = []
    t_rec = None  # steps into Phase 3 when recovery criterion first met

    for step in range(PHASE3_STEPS):
        obs = rng.normal(loc=mu0, scale=SIGNAL_STD)
        result = agent.step(obs)
        acc = result["mean_accuracy"]
        phase3_accs.append(acc)

        if t_rec is None:
            top_w = result["max_weight"]
            if acc >= ACC_RECOVERY_FRAC * b_acc and top_w >= TOP_WEIGHT_THRESHOLD:
                t_rec = step + 1  # 1-indexed steps into Phase 3

    # Sanity check
    t_rec_display = t_rec if t_rec is not None else ">500"
    assert t_rec is None or t_rec > 0, f"T_rec must be positive, got {t_rec}"

    return {
        "phase1_acc": float(np.mean(phase1_accs)),
        "phase2_acc": float(np.mean(phase2_accs)),
        "phase3_acc": float(np.mean(phase3_accs)),
        "b_acc": b_acc,
        "b_weight": b_weight,
        "t_rec": t_rec,
        "t_rec_display": t_rec_display,
    }


def main():
    metrics = run()

    t_rec = metrics["t_rec"]
    if t_rec is None:
        result_label = "✗ FAILED"
    elif t_rec <= IMMUNE_THRESHOLD:
        result_label = "✓ IMMUNE"
    else:
        result_label = "~ SLOW"

    print_results_table(
        title="Structural Immunity",
        cols=["Phase 1 Acc", "Phase 2 Acc", "Phase 3 Acc", "T_rec", "Result"],
        rows=[{
            "Phase 1 Acc": f"{metrics['phase1_acc']:.2f}",
            "Phase 2 Acc": f"{metrics['phase2_acc']:.2f}",
            "Phase 3 Acc": f"{metrics['phase3_acc']:.2f}",
            "T_rec": str(metrics["t_rec_display"]),
            "Result": result_label,
        }],
    )


if __name__ == "__main__":
    main()
