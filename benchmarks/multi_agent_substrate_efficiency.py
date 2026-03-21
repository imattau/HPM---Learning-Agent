"""
Multi-Agent Benchmark 3: Substrate Efficiency
===============================================
Same 3-cluster data stream as substrate_efficiency.py, but run with two agents
sharing a PatternField and StructuralLawMonitor.

New output vs single-agent:
  - Per-agent pattern counts (do agents specialise to different clusters?)
  - Field redundancy at end of run (shared patterns across agents)
  - Combined Pareto comparison includes multi-agent HPM as a single model

Run:
    python benchmarks/multi_agent_substrate_efficiency.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from benchmarks.multi_agent_common import (
    make_orchestrator, avg_metric, compute_redundancy, print_results_table,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATURE_DIM = 16
N_STEPS = 2000
EVAL_EVERY = 20
N_CLUSTERS = 3
CLUSTER_STD = 0.3
GMM_SAMPLE_SIZE = 500
GMM_K_VALUES = [1, 2, 3, 4, 5]
RNG_SEED = 42


def make_cluster_means(rng, n, dim):
    means = []
    for _ in range(n):
        v = rng.standard_normal(dim)
        means.append(v / np.linalg.norm(v))
    return means


def sample_clusters(rng, means, std, n):
    dim = means[0].shape[0]
    samples = []
    for _ in range(n):
        cluster_idx = rng.integers(0, len(means))
        obs = rng.normal(loc=means[cluster_idx], scale=std, size=dim)
        samples.append(obs)
    return np.array(samples)


def is_pareto_dominated(complexity, accuracy, others_c, others_a):
    for c, a in zip(others_c, others_a):
        if c <= complexity and a >= accuracy and (c < complexity or a > accuracy):
            return True
    return False


def run() -> dict:
    rng = np.random.default_rng(RNG_SEED)
    cluster_means = make_cluster_means(rng, N_CLUSTERS, FEATURE_DIM)

    orch, agents, store = make_orchestrator(
        n_agents=2,
        feature_dim=FEATURE_DIM,
        agent_ids=["efficiency_a", "efficiency_b"],
        with_monitor=True,
        T_monitor=99999,   # suppress console output; redundancy computed at end
    )

    eval_records = []
    hpm_nll_readings = []
    final_field_quality = {}

    for step in range(1, N_STEPS + 1):
        cluster_idx = rng.integers(0, N_CLUSTERS)
        obs = rng.normal(loc=cluster_means[cluster_idx], scale=CLUSTER_STD, size=FEATURE_DIM)

        observations = {a.agent_id: obs for a in agents}
        result = orch.step(observations)

        if step % EVAL_EVERY == 0:
            nll_val = avg_metric(result, agents, "mean_accuracy")
            compress_val = avg_metric(result, agents, "compress_mean")
            # Sum pattern counts across agents
            total_patterns = sum(
                result[a.agent_id].get("n_patterns", 0) for a in agents
            )
            eval_records.append({
                "step": step,
                "mean_accuracy": nll_val,
                "compress_mean": compress_val,
                "n_patterns_total": total_patterns,
                "n_patterns_a": result[agents[0].agent_id].get("n_patterns", 0),
                "n_patterns_b": result[agents[1].agent_id].get("n_patterns", 0),
            })
            hpm_nll_readings.append(nll_val)

    final_redundancy = compute_redundancy(orch)
    final = eval_records[-1]
    max_gmm_k = max(GMM_K_VALUES)
    # Complexity: total patterns across both agents, normalised to [0,1] vs max GMM k * 2
    hpm_complexity = min(final["n_patterns_total"], max_gmm_k * 2) / (max_gmm_k * 2)

    nll_min = min(hpm_nll_readings)
    nll_max = max(hpm_nll_readings)
    raw_nll = final["mean_accuracy"]
    if nll_max > nll_min:
        hpm_accuracy = (nll_max - raw_nll) / (nll_max - nll_min)
    else:
        hpm_accuracy = 0.5

    # GMM comparison
    gmm_samples = sample_clusters(rng, cluster_means, CLUSTER_STD, GMM_SAMPLE_SIZE)
    try:
        from sklearn.mixture import GaussianMixture
        sklearn_available = True
    except ImportError:
        sklearn_available = False

    gmm_results = []
    if sklearn_available:
        for k in GMM_K_VALUES:
            gmm = GaussianMixture(n_components=k, random_state=RNG_SEED, max_iter=200)
            gmm.fit(gmm_samples)
            bic = float(gmm.bic(gmm_samples))
            mean_ll = float(gmm.score(gmm_samples))
            gmm_results.append({"k": k, "bic": bic, "mean_ll": mean_ll})

    if gmm_results:
        bics = [g["bic"] for g in gmm_results]
        lls = [g["mean_ll"] for g in gmm_results]
        bic_min, bic_max = min(bics), max(bics)
        ll_min, ll_max = min(lls), max(lls)
        for g in gmm_results:
            g["norm_complexity"] = (g["bic"] - bic_min) / (bic_max - bic_min) if bic_max > bic_min else 0.5
            g["norm_accuracy"] = (g["mean_ll"] - ll_min) / (ll_max - ll_min) if ll_max > ll_min else 0.5

    all_complexities = [hpm_complexity] + [g["norm_complexity"] for g in gmm_results]
    all_accuracies = [hpm_accuracy] + [g["norm_accuracy"] for g in gmm_results]

    all_acc_min = min(all_accuracies)
    all_acc_max = max(all_accuracies)
    all_comp_min = min(all_complexities)
    all_comp_max = max(all_complexities)

    def norm_val(v, lo, hi):
        return (v - lo) / (hi - lo) if hi > lo else 0.5

    models = []
    hpm_nc = norm_val(hpm_complexity, all_comp_min, all_comp_max)
    hpm_na = norm_val(hpm_accuracy, all_acc_min, all_acc_max)
    models.append({
        "name": "HPM 2-agent (final)",
        "complexity": hpm_complexity,
        "accuracy": hpm_accuracy,
        "norm_complexity": hpm_nc,
        "norm_accuracy": hpm_na,
    })
    for g in gmm_results:
        nc = norm_val(g["norm_complexity"], all_comp_min, all_comp_max)
        na = norm_val(g["norm_accuracy"], all_acc_min, all_acc_max)
        models.append({
            "name": f"GMM k={g['k']}",
            "complexity": g["norm_complexity"],
            "accuracy": g["norm_accuracy"],
            "norm_complexity": nc,
            "norm_accuracy": na,
        })

    all_nc = [m["norm_complexity"] for m in models]
    all_na = [m["norm_accuracy"] for m in models]
    for i, m in enumerate(models):
        others_c = all_nc[:i] + all_nc[i+1:]
        others_a = all_na[:i] + all_na[i+1:]
        m["on_pareto"] = not is_pareto_dominated(m["norm_complexity"], m["norm_accuracy"], others_c, others_a)

    assert any(m["on_pareto"] for m in models)

    return {
        "models": models,
        "hpm_final": final,
        "final_redundancy": final_redundancy,
        "sklearn_available": sklearn_available,
    }


def main():
    result = run()
    models = result["models"]
    final = result["hpm_final"]
    redundancy = result["final_redundancy"]

    if not result["sklearn_available"]:
        print("WARNING: scikit-learn not installed. GMM comparison skipped.")

    rows = []
    for m in models:
        rows.append({
            "Model": m["name"],
            "Complexity": f"{m['complexity']:.2f}",
            "Accuracy": f"{m['accuracy']:.2f}",
            "On Pareto Frontier?": "✓" if m["on_pareto"] else "✗",
        })

    print_results_table(
        title="Multi-Agent Substrate Efficiency (2 agents, shared field)",
        cols=["Model", "Complexity", "Accuracy", "On Pareto Frontier?"],
        rows=rows,
    )

    redundancy_str = f"{redundancy:.3f}" if redundancy is not None else "N/A"
    print_results_table(
        title="Agent Specialisation",
        cols=["Agent", "Final Patterns"],
        rows=[
            {"Agent": "efficiency_a", "Final Patterns": str(final["n_patterns_a"])},
            {"Agent": "efficiency_b", "Final Patterns": str(final["n_patterns_b"])},
            {"Agent": "Field Redundancy", "Final Patterns": redundancy_str},
        ],
    )


if __name__ == "__main__":
    main()
