"""
Multi-Agent Benchmark 2: Structural Immunity
=============================================
Same 3-phase protocol as structural_immunity.py, but run with two agents
sharing a PatternField and StructuralLawMonitor.

New output vs single-agent: field redundancy snapshots at end of each phase,
showing how the shared field evolves under noise then recovers.

3-phase data stream:
  Phase 1 (500 steps): Gaussian signal, fixed mean, std=0.1
  Phase 2 (20 steps):  Uniform noise in [-1, 1]^d
  Phase 3 (500 steps): Same Gaussian signal as Phase 1

Run:
    python benchmarks/multi_agent_structural_immunity.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from benchmarks.multi_agent_common import (
    make_orchestrator, step_all, avg_metric, compute_redundancy, print_results_table,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURE_DIM = 16
PHASE1_STEPS = 500
PHASE2_STEPS = 20
PHASE3_STEPS = 500
SIGNAL_STD = 0.1
BASELINE_WINDOW = 50
ACC_RECOVERY_FRAC = 0.9
TOP_WEIGHT_THRESHOLD = 0.5
IMMUNE_THRESHOLD = 100
RNG_SEED = 42


def unit_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def run() -> dict:
    rng = np.random.default_rng(RNG_SEED)
    mu0 = unit_vector(rng, FEATURE_DIM)

    orch, agents, store = make_orchestrator(
        n_agents=2,
        feature_dim=FEATURE_DIM,
        agent_ids=["immunity_a", "immunity_b"],
        with_monitor=True,
        T_monitor=50,
        agent_seeds=[42, 99],   # different initial patterns → agents have complementary starts
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
        result = step_all(orch, agents, obs)
        acc = avg_metric(result, agents, "mean_accuracy")
        phase1_accs.append(acc)
        if step >= PHASE1_STEPS - BASELINE_WINDOW:
            baseline_window_accs.append(acc)

    b_acc = float(np.mean(baseline_window_accs))
    b_redundancy = compute_redundancy(orch)

    all_records = store.query_all()
    b_weight = max((w for _, w, _ in all_records), default=0.0)

    # -----------------------------------------------------------------------
    # Phase 2: Uniform noise storm
    # -----------------------------------------------------------------------
    phase2_accs = []

    for _ in range(PHASE2_STEPS):
        obs = rng.uniform(-1.0, 1.0, size=FEATURE_DIM)
        result = step_all(orch, agents, obs)
        phase2_accs.append(avg_metric(result, agents, "mean_accuracy"))

    phase2_redundancy = compute_redundancy(orch)

    # -----------------------------------------------------------------------
    # Phase 3: Same Gaussian signal — measure recovery
    # -----------------------------------------------------------------------
    phase3_accs = []
    t_rec = None

    for step in range(PHASE3_STEPS):
        obs = rng.normal(loc=mu0, scale=SIGNAL_STD)
        result = step_all(orch, agents, obs)
        acc = avg_metric(result, agents, "mean_accuracy")
        phase3_accs.append(acc)

        if t_rec is None:
            max_w = max((w for _, w, _ in store.query_all()), default=0.0)
            if abs(acc - b_acc) <= (1.0 - ACC_RECOVERY_FRAC) * abs(b_acc) and max_w >= TOP_WEIGHT_THRESHOLD:
                t_rec = step + 1

    phase3_redundancy = compute_redundancy(orch)

    t_rec_display = t_rec if t_rec is not None else ">500"
    assert t_rec is None or t_rec > 0

    def _fmt_r(r):
        return f"{r:.3f}" if r is not None else "N/A"

    return {
        "phase1_acc": float(np.mean(phase1_accs)),
        "phase2_acc": float(np.mean(phase2_accs)),
        "phase3_acc": float(np.mean(phase3_accs)),
        "b_acc": b_acc,
        "b_weight": b_weight,
        "b_redundancy": b_redundancy,
        "phase2_redundancy": phase2_redundancy,
        "phase3_redundancy": phase3_redundancy,
        "t_rec": t_rec,
        "t_rec_display": t_rec_display,
        "_fmt_r": _fmt_r,
    }


def main():
    metrics = run()
    _fmt_r = metrics["_fmt_r"]

    t_rec = metrics["t_rec"]
    if t_rec is None:
        result_label = "✗ FAILED"
    elif t_rec <= IMMUNE_THRESHOLD:
        result_label = "✓ IMMUNE"
    else:
        result_label = "~ SLOW"

    print_results_table(
        title="Multi-Agent Structural Immunity (2 agents, shared field)",
        cols=["Phase", "Avg Accuracy", "Field Redundancy"],
        rows=[
            {"Phase": "Phase 1 (signal)",  "Avg Accuracy": f"{metrics['phase1_acc']:.2f}", "Field Redundancy": _fmt_r(metrics["b_redundancy"])},
            {"Phase": "Phase 2 (noise)",   "Avg Accuracy": f"{metrics['phase2_acc']:.2f}", "Field Redundancy": _fmt_r(metrics["phase2_redundancy"])},
            {"Phase": "Phase 3 (recover)", "Avg Accuracy": f"{metrics['phase3_acc']:.2f}", "Field Redundancy": _fmt_r(metrics["phase3_redundancy"])},
        ],
    )

    print_results_table(
        title="Recovery",
        cols=["T_rec", "Result"],
        rows=[{"T_rec": str(metrics["t_rec_display"]), "Result": result_label}],
    )


if __name__ == "__main__":
    main()
