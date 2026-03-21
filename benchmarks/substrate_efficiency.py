"""
Benchmark 3: Substrate Efficiency
====================================
Tests whether HPM discovers a compact representation of a redundant data stream,
and compares its compression/accuracy trade-off against a GMM baseline on a
Pareto frontier.

Data: 3 overlapping Gaussian clusters in 16-dimensional space.
Optimal model needs only 3 components.

Run:
    python benchmarks/substrate_efficiency.py
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
N_STEPS = 2000
EVAL_EVERY = 20
N_CLUSTERS = 3
CLUSTER_STD = 0.3
GMM_SAMPLE_SIZE = 500
GMM_K_VALUES = [1, 2, 3, 4, 5]
RNG_SEED = 42


def make_cluster_means(rng: np.random.Generator, n: int, dim: int) -> list[np.ndarray]:
    means = []
    for _ in range(n):
        v = rng.standard_normal(dim)
        means.append(v / np.linalg.norm(v))
    return means


def sample_clusters(rng: np.random.Generator, means: list, std: float, n: int) -> np.ndarray:
    """Sample n observations uniformly across clusters."""
    dim = means[0].shape[0]
    samples = []
    for _ in range(n):
        cluster_idx = rng.integers(0, len(means))
        obs = rng.normal(loc=means[cluster_idx], scale=std, size=dim)
        samples.append(obs)
    return np.array(samples)


def is_pareto_dominated(complexity: float, accuracy: float,
                        others_c: list, others_a: list) -> bool:
    """Return True if (complexity, accuracy) is dominated by any other model."""
    for c, a in zip(others_c, others_a):
        if c <= complexity and a >= accuracy and (c < complexity or a > accuracy):
            return True
    return False


def run() -> dict:
    rng = np.random.default_rng(RNG_SEED)
    cluster_means = make_cluster_means(rng, N_CLUSTERS, FEATURE_DIM)

    agent = make_agent(feature_dim=FEATURE_DIM, agent_id="efficiency_bench")

    eval_records = []

    for step in range(1, N_STEPS + 1):
        cluster_idx = rng.integers(0, N_CLUSTERS)
        obs = rng.normal(loc=cluster_means[cluster_idx], scale=CLUSTER_STD, size=FEATURE_DIM)
        result = agent.step(obs)

        if step % EVAL_EVERY == 0:
            eval_records.append({
                "step": step,
                "mean_accuracy": result["mean_accuracy"],
                "compress_mean": result["compress_mean"],
                "n_patterns": result["n_patterns"],
            })

    # Final HPM snapshot
    final = eval_records[-1]
    hpm_complexity = 1.0 - final["compress_mean"]
    hpm_accuracy = final["mean_accuracy"]

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
            mean_ll = float(gmm.score(gmm_samples))  # mean log-likelihood per sample
            gmm_results.append({"k": k, "bic": bic, "mean_ll": mean_ll})

    # Normalise GMM metrics to [0, 1] for Pareto comparison
    if gmm_results:
        bics = [g["bic"] for g in gmm_results]
        lls = [g["mean_ll"] for g in gmm_results]
        bic_min, bic_max = min(bics), max(bics)
        ll_min, ll_max = min(lls), max(lls)

        for g in gmm_results:
            # BIC: lower is better -> complexity proxy = normalised BIC
            g["norm_complexity"] = (
                (g["bic"] - bic_min) / (bic_max - bic_min)
                if bic_max > bic_min else 0.5
            )
            # LL: higher is better -> accuracy = normalised LL
            g["norm_accuracy"] = (
                (g["mean_ll"] - ll_min) / (ll_max - ll_min)
                if ll_max > ll_min else 0.5
            )

    # Build unified list of all models for Pareto computation
    all_complexities = [hpm_complexity] + [g["norm_complexity"] for g in gmm_results]
    all_accuracies = [hpm_accuracy] + [g["norm_accuracy"] for g in gmm_results]

    # For HPM: use raw values (not normalised) — already in [0,1] range
    # We normalise HPM accuracy too for fair Pareto comparison
    all_acc_min = min(all_accuracies)
    all_acc_max = max(all_accuracies)
    all_comp_min = min(all_complexities)
    all_comp_max = max(all_complexities)

    def norm_val(v, lo, hi):
        return (v - lo) / (hi - lo) if hi > lo else 0.5

    models = []

    # HPM entry
    hpm_nc = norm_val(hpm_complexity, all_comp_min, all_comp_max)
    hpm_na = norm_val(hpm_accuracy, all_acc_min, all_acc_max)
    models.append({
        "name": "HPM (final)",
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

    # Compute Pareto frontier membership
    all_nc = [m["norm_complexity"] for m in models]
    all_na = [m["norm_accuracy"] for m in models]

    for i, m in enumerate(models):
        others_c = all_nc[:i] + all_nc[i+1:]
        others_a = all_na[:i] + all_na[i+1:]
        dominated = is_pareto_dominated(m["norm_complexity"], m["norm_accuracy"],
                                        others_c, others_a)
        m["on_pareto"] = not dominated

    # Sanity check
    assert any(m["on_pareto"] for m in models), "Pareto frontier must contain at least 1 entry"

    return {
        "models": models,
        "hpm_final": final,
        "sklearn_available": sklearn_available,
    }


def main():
    result = run()
    models = result["models"]

    if not result["sklearn_available"]:
        print("WARNING: scikit-learn not installed. GMM comparison skipped.")
        print("Install with: pip install scikit-learn>=1.3")

    rows = []
    for m in models:
        rows.append({
            "Model": m["name"],
            "Complexity": f"{m['complexity']:.2f}",
            "Accuracy": f"{m['accuracy']:.2f}",
            "On Pareto Frontier?": "✓" if m["on_pareto"] else "✗",
        })

    print_results_table(
        title="Substrate Efficiency",
        cols=["Model", "Complexity", "Accuracy", "On Pareto Frontier?"],
        rows=rows,
    )


if __name__ == "__main__":
    main()
