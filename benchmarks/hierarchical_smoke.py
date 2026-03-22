"""
Hierarchical Abstraction Stack — Smoke Benchmark
=================================================
Validates the 2-level HierarchicalOrchestrator end-to-end on a synthetic
Gaussian signal. Not ARC — just verifies that:
  1. Level 2 receives bundles at the correct K-cadence
  2. Level 2 accuracy is finite and non-NaN after training
  3. The _t counter increments correctly

Run:
    python benchmarks/hierarchical_smoke.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from hpm.agents.hierarchical import make_hierarchical_orchestrator
from benchmarks.common import print_results_table

L1_FEATURE_DIM = 16
N_L1_AGENTS = 2
N_L2_AGENTS = 1
N_STEPS = 100
K = 5
RNG_SEED = 42


def run() -> dict:
    rng = np.random.default_rng(RNG_SEED)
    mu = rng.standard_normal(L1_FEATURE_DIM)
    mu /= np.linalg.norm(mu)

    h_orch, l1_agents, l2_agents = make_hierarchical_orchestrator(
        n_l1_agents=N_L1_AGENTS,
        n_l2_agents=N_L2_AGENTS,
        l1_feature_dim=L1_FEATURE_DIM,
        K=K,
    )

    l2_cadence_ticks = 0   # how many times the cadence fired
    l2_accs = []

    for step in range(N_STEPS):
        obs = rng.normal(loc=mu, scale=0.1)
        result = h_orch.step(obs)

        if result["level2"]:
            l2_cadence_ticks += 1
            # Collect mean accuracy from Level 2 agents
            for aid, metrics in result["level2"].items():
                if "mean_accuracy" in metrics:
                    l2_accs.append(metrics["mean_accuracy"])

    expected_l2_ticks = N_STEPS // K
    expected_t = N_STEPS

    return {
        "n_steps": N_STEPS,
        "K": K,
        "l2_cadence_ticks": l2_cadence_ticks,
        "expected_l2_ticks": expected_l2_ticks,
        "final_t": h_orch._t,
        "expected_t": expected_t,
        "l2_mean_acc": float(np.mean(l2_accs)) if l2_accs else float("nan"),
        "l2_acc_finite": bool(np.isfinite(l2_accs).all()) if l2_accs else False,
        "cadence_correct": l2_cadence_ticks == expected_l2_ticks,
        "t_correct": h_orch._t == expected_t,
    }


def main():
    print(f"Running Hierarchical Smoke Benchmark "
          f"({N_L1_AGENTS} L1 agents → {N_L2_AGENTS} L2 agent, K={K}, {N_STEPS} steps)...")
    m = run()

    cadence_ok = m["cadence_correct"]
    t_ok = m["t_correct"]
    acc_ok = m["l2_acc_finite"]
    passed = cadence_ok and t_ok and acc_ok

    print_results_table(
        title="Hierarchical Smoke Benchmark",
        cols=["Check", "Expected", "Actual", "Status"],
        rows=[
            {
                "Check": "L2 cadence ticks",
                "Expected": str(m["expected_l2_ticks"]),
                "Actual": str(m["l2_cadence_ticks"]),
                "Status": "✓" if cadence_ok else "✗",
            },
            {
                "Check": "Final _t",
                "Expected": str(m["expected_t"]),
                "Actual": str(m["final_t"]),
                "Status": "✓" if t_ok else "✗",
            },
            {
                "Check": "L2 accuracy finite",
                "Expected": "yes",
                "Actual": f"{m['l2_mean_acc']:.4f}" if acc_ok else "NaN",
                "Status": "✓" if acc_ok else "✗",
            },
            {
                "Check": "Overall",
                "Expected": "",
                "Actual": "",
                "Status": "✓ PASS" if passed else "✗ FAIL",
            },
        ],
    )


if __name__ == "__main__":
    main()
