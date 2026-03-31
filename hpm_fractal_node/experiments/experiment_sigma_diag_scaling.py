"""
Sigma-diag scaling experiment: memory and runtime comparison.

Runs the math arithmetic experiment in two modes:
  - Baseline: node_use_diag=False  (full D×D covariance matrices for learned nodes)
  - Diagonal: node_use_diag=True   (D-vector diagonal covariance for learned nodes)

For each mode measures wall-clock time, peak memory (via tracemalloc), final node
counts, coverage %, and mean category purity.  Prints a side-by-side comparison
table and a theoretical scaling projection for D=512, 50 000 priors.

Usage:
    PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_sigma_diag_scaling.py
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn import Observer, calibrate_tau
from hfn.tiered_forest import TieredForest
from hpm_fractal_node.math.math_loader import (
    generate_observations,
    get_category,
    CATEGORY_NAMES,
    D,
)
from hpm_fractal_node.math.math_world_model import build_math_world_model

# ---------------------------------------------------------------------------
# Configuration — identical to experiment_math.py
# ---------------------------------------------------------------------------

N_SAMPLES = 5000
N_PASSES = 4
SEED = 42

TAU_SIGMA = 1.0
TAU_MARGIN = 5.0

N_CATEGORIES = len(CATEGORY_NAMES)


def purity(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(counts.values()) / total


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def run_experiment(use_diag: bool) -> dict:
    """Run the full math observation loop and return metrics."""
    label = "diag" if use_diag else "full"
    print(f"\n{'='*60}")
    print(f"  Mode: node_use_diag={use_diag}  ({label} sigma storage)")
    print(f"{'='*60}")

    data = generate_observations(n=N_SAMPLES, seed=SEED)

    cold_dir = Path(__file__).parents[2] / "data" / "hfn_math_cold"
    cold_dir.mkdir(parents=True, exist_ok=True)

    forest, prior_ids = build_math_world_model(
        forest_cls=TieredForest,
        cold_dir=cold_dir,
        max_hot=600,
    )
    forest.set_protected(prior_ids)
    n_priors = len(prior_ids)
    print(f"  {n_priors} priors registered and protected")

    tau = calibrate_tau(D, sigma_scale=TAU_SIGMA, margin=TAU_MARGIN)

    obs = Observer(
        forest,
        tau=tau,
        protected_ids=prior_ids,
        recombination_strategy="nearest_prior",
        hausdorff_absorption_threshold=0.35,
        hausdorff_absorption_weight_floor=0.4,
        absorption_miss_threshold=20,
        persistence_guided_absorption=True,
        lacunarity_guided_creation=True,
        lacunarity_creation_radius=0.08,
        multifractal_guided_absorption=False,
        gap_query_threshold=None,
        max_expand_depth=2,
        node_use_diag=use_diag,
    )

    node_explanations: dict[str, list] = defaultdict(list)

    # -- Start timing and memory tracking -----------------------------------
    tracemalloc.start()
    t0 = time.perf_counter()

    for p in range(N_PASSES):
        n_explained = 0
        n_unexplained = 0
        rng = np.random.default_rng(SEED + p)
        order = rng.permutation(len(data))

        for i in order:
            vec, (left, op, right, result) = data[i]
            x = vec.astype(np.float64)
            obs_result = obs.observe(x)
            forest._on_observe()

            category = get_category(left, op, right, result)
            if obs_result.explanation_tree:
                best_id = max(
                    obs_result.accuracy_scores,
                    key=lambda k: obs_result.accuracy_scores[k],
                )
                node_explanations[best_id].append((left, op, right, result, category))
                n_explained += 1
            else:
                n_unexplained += 1

        n_total = len(data)
        print(f"  Pass {p+1}: explained {n_explained}/{n_total} "
              f"({100*n_explained/n_total:.1f}%)")

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # -- Compute metrics ----------------------------------------------------
    n_obs_total = N_PASSES * N_SAMPLES
    total_attributed = sum(len(v) for v in node_explanations.values())
    coverage_pct = 100.0 * total_attributed / n_obs_total

    learned_nodes = [k for k in node_explanations if k not in prior_ids]
    active_nodes = list(forest.active_nodes())
    n_active_learned = sum(1 for n in active_nodes if n.id not in prior_ids)
    final_node_count = n_priors + n_active_learned

    # Category purity over learned nodes with n >= 5
    cat_purities = []
    for nid in learned_nodes:
        labels = node_explanations[nid]
        if len(labels) < 5:
            continue
        cat_counts: dict[str, int] = defaultdict(int)
        for *_, cat in labels:
            cat_counts[cat] += 1
        cat_purities.append(purity(cat_counts))

    mean_purity = float(np.mean(cat_purities)) if cat_purities else float("nan")

    return {
        "label": label,
        "use_diag": use_diag,
        "elapsed_s": elapsed,
        "peak_mb": peak_bytes / (1024 ** 2),
        "n_priors": n_priors,
        "n_active_learned": n_active_learned,
        "final_node_count": final_node_count,
        "coverage_pct": coverage_pct,
        "mean_purity": mean_purity,
        "n_purity_nodes": len(cat_purities),
    }


# ---------------------------------------------------------------------------
# Scaling projection
# ---------------------------------------------------------------------------

def scaling_projection(D_proj: int, n_priors_proj: int) -> None:
    """Print theoretical memory for full vs diag sigma at large D."""
    bytes_per_float = 8  # float64

    # Prior nodes in both modes are stored as D-vectors (existing priors use
    # diagonal already from build_math_world_model); learned nodes differ.
    # We project memory for *learned* nodes only, assuming same ratio as seen.
    # For illustration we project the cost of storing sigma for n_priors_proj nodes.

    full_sigma_bytes = n_priors_proj * D_proj * D_proj * bytes_per_float
    diag_sigma_bytes = n_priors_proj * D_proj * bytes_per_float

    mu_bytes = n_priors_proj * D_proj * bytes_per_float  # same for both modes

    print(f"\n{'='*60}")
    print(f"  Scaling projection: D={D_proj}, {n_priors_proj:,} nodes")
    print(f"{'='*60}")
    print(f"  mu storage (identical):   {mu_bytes / (1024**2):>10.1f} MB")
    print(f"  sigma — full (D×D):       {full_sigma_bytes / (1024**2):>10.1f} MB")
    print(f"  sigma — diag (D-vec):     {diag_sigma_bytes / (1024**2):>10.1f} MB")
    ratio = full_sigma_bytes / max(diag_sigma_bytes, 1)
    print(f"  Diag saves:               {(1 - 1/ratio)*100:.1f}%  ({ratio:.0f}x smaller sigma)")
    total_full = (mu_bytes + full_sigma_bytes) / (1024**2)
    total_diag = (mu_bytes + diag_sigma_bytes) / (1024**2)
    print(f"  Total (mu+sigma) full:    {total_full:>10.1f} MB")
    print(f"  Total (mu+sigma) diag:    {total_diag:>10.1f} MB")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(baseline: dict, diag: dict) -> None:
    print(f"\n{'='*60}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*60}")
    col_w = 20

    def row(label: str, fmt_b: str, fmt_d: str) -> None:
        print(f"  {label:<28s}  {fmt_b:>{col_w}s}  {fmt_d:>{col_w}s}")

    row("Metric", "node_use_diag=False", "node_use_diag=True")
    print("  " + "-" * (28 + 2 + col_w + 2 + col_w))
    row("Wall-clock time (s)",
        f"{baseline['elapsed_s']:.2f}",
        f"{diag['elapsed_s']:.2f}")
    row("Peak memory (MB)",
        f"{baseline['peak_mb']:.1f}",
        f"{diag['peak_mb']:.1f}")
    row("Prior nodes",
        str(baseline['n_priors']),
        str(diag['n_priors']))
    row("Active learned nodes",
        str(baseline['n_active_learned']),
        str(diag['n_active_learned']))
    row("Total node count",
        str(baseline['final_node_count']),
        str(diag['final_node_count']))
    row("Coverage %",
        f"{baseline['coverage_pct']:.2f}%",
        f"{diag['coverage_pct']:.2f}%")
    row("Mean cat purity (n>=5)",
        f"{baseline['mean_purity']:.4f}" if not np.isnan(baseline['mean_purity']) else "n/a",
        f"{diag['mean_purity']:.4f}" if not np.isnan(diag['mean_purity']) else "n/a")
    row("Purity-eligible nodes",
        str(baseline['n_purity_nodes']),
        str(diag['n_purity_nodes']))

    # Derived speedup / memory saving
    speedup = baseline['elapsed_s'] / max(diag['elapsed_s'], 1e-9)
    mem_saving = (baseline['peak_mb'] - diag['peak_mb']) / max(baseline['peak_mb'], 1e-9) * 100
    cov_delta = diag['coverage_pct'] - baseline['coverage_pct']
    pur_delta = diag['mean_purity'] - baseline['mean_purity']

    print("  " + "-" * (28 + 2 + col_w + 2 + col_w))
    print(f"\n  Speedup (full/diag):      {speedup:.2f}x")
    print(f"  Peak memory saving:       {mem_saving:+.1f}%")
    print(f"  Coverage delta:           {cov_delta:+.4f}%  (should be ~0)")
    if not (np.isnan(baseline['mean_purity']) or np.isnan(diag['mean_purity'])):
        print(f"  Mean purity delta:        {pur_delta:+.4f}   (should be ~0)")

    # Sanity check
    if abs(cov_delta) > 1.0:
        print("\n  WARNING: coverage differs by more than 1 pp — check experiment setup!")
    else:
        print("\n  Sanity check: coverage identical within tolerance. OK.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Sigma-diag scaling experiment")
    print(f"  D={D}, N_SAMPLES={N_SAMPLES}, N_PASSES={N_PASSES}, SEED={SEED}")
    print(f"  Observation stream is identical between modes (same seed).")

    baseline = run_experiment(use_diag=False)
    diag_run = run_experiment(use_diag=True)

    print_comparison(baseline, diag_run)
    scaling_projection(D_proj=512, n_priors_proj=50_000)


if __name__ == "__main__":
    main()
