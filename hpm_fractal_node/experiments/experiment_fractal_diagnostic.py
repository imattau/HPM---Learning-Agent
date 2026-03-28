"""
Fractal convergence diagnostic experiment.

Tracks the box-counting dimension of the Observer's node population in
μ-space after each pass. Tests the IFS convergence hypothesis:

  If recombine() acts as a contracting affine map (IFS), the set of
  learned nodes should converge toward a fractal attractor. Dimension
  should decrease as structure emerges — high dimension = scattered,
  low dimension = coherent hierarchical clusters.

Reports dimension per pass for:
  - All nodes (priors + learned)
  - Learned nodes only (priors have fixed positions)
  - Prior nodes only (baseline — should be ~constant)

Run: python3 -m hpm_fractal_node.experiments.experiment_fractal_diagnostic
"""

import json
import glob
import numpy as np
from collections import defaultdict

from hfn import Observer, population_dimension
from hpm_fractal_node.arc.arc_world_model import build_world_model


def load_3x3_colour(data_dir: str = "data/ARC-AGI-2/data/training") -> list[tuple[str, np.ndarray]]:
    records = []
    for f in sorted(glob.glob(f"{data_dir}/*.json")):
        d = json.load(open(f))
        puzzle_id = f.split("/")[-1].replace(".json", "")
        for ex in d["train"]:
            grid = ex["input"]
            if len(grid) != 3 or len(grid[0]) != 3:
                continue
            vec = np.array([[cell / 9.0 for cell in row] for row in grid]).flatten()
            records.append((puzzle_id, vec))
    return records


def run(n_passes: int = 8, seed: int = 42, n_scales: int = 8) -> None:
    rng = np.random.default_rng(seed)
    records = load_3x3_colour()
    print(f"Loaded {len(records)} colour-encoded observations (3x3)\n")

    forest, registry = build_world_model(rows=3, cols=3)
    prior_ids = set(registry.keys())
    D = 9
    baseline = D / 2 * np.log(2 * np.pi)
    tau = baseline + 1.0

    obs = Observer(
        forest,
        tau=tau,
        budget=10,
        lambda_complexity=0.05,
        alpha_gain=0.15,
        beta_loss=0.05,
        absorption_overlap_threshold=0.6,
        absorption_miss_threshold=8,
        residual_surprise_threshold=baseline + 2.5,
        compression_cooccurrence_threshold=4,
        w_init=0.1,
        protected_ids=prior_ids,
    )

    # Measure baseline dimension (before any observations)
    all_nodes = forest.active_nodes()
    prior_nodes = [n for n in all_nodes if n.id in prior_ids]
    prior_dim_baseline = population_dimension(prior_nodes, n_scales=n_scales)
    all_dim_baseline = population_dimension(all_nodes, n_scales=n_scales)

    print(f"{'Pass':>5}  {'All nodes':>12}  {'Learned only':>14}  {'Priors (fixed)':>16}  {'N total':>8}  {'N learned':>10}")
    print(f"{'':>5}  {'(dim)':>12}  {'(dim)':>14}  {'(dim)':>16}  {'':>8}  {'':>10}")
    print(f"  --- baseline ---")
    print(f"{'  0':>5}  {all_dim_baseline:>12.4f}  {'—':>14}  {prior_dim_baseline:>16.4f}  {len(all_nodes):>8}  {'0':>10}")

    for pass_num in range(1, n_passes + 1):
        indices = rng.permutation(len(records))
        for i in indices:
            _, vec = records[i]
            obs.observe(vec)

        active = forest.active_nodes()
        learned = [n for n in active if n.id not in prior_ids]
        priors = [n for n in active if n.id in prior_ids]

        dim_all = population_dimension(active, n_scales=n_scales)
        dim_learned = population_dimension(learned, n_scales=n_scales) if learned else float('nan')
        dim_prior = population_dimension(priors, n_scales=n_scales)

        print(f"  {pass_num:>3}  {dim_all:>12.4f}  {dim_learned:>14.4f}  {dim_prior:>16.4f}  {len(active):>8}  {len(learned):>10}")

    # Final analysis
    active = forest.active_nodes()
    learned = [n for n in active if n.id not in prior_ids]

    print(f"\n=== Interpretation ===")
    print(f"Prior dimension (fixed):   {prior_dim_baseline:.4f}")
    print(f"All-node dimension (final): {population_dimension(active, n_scales):.4f}")
    if learned:
        print(f"Learned-only dimension:    {population_dimension(learned, n_scales):.4f}")

    print(f"\nIf learned dimension < prior dimension:")
    print(f"  → Observer clusters learned nodes around prior attractors (IFS convergence)")
    print(f"If learned dimension ≈ prior dimension:")
    print(f"  → Learned nodes fill observation space uniformly (no convergence)")
    print(f"If learned dimension > prior dimension:")
    print(f"  → Learned nodes are more scattered than priors (fragmentation)")

    # Show where learned nodes cluster relative to priors
    if learned:
        print(f"\n=== Nearest prior for each learned node ===")
        for node in sorted(learned, key=lambda n: -obs.get_weight(n.id))[:8]:
            nearest = min(
                (p for p in active if p.id in prior_ids),
                key=lambda p: float(np.linalg.norm(node.mu - p.mu))
            )
            dist = float(np.linalg.norm(node.mu - nearest.mu))
            w = obs.get_weight(node.id)
            print(f"  {node.id[:36]}  w={w:.3f}  nearest={nearest.id[:30]}  dist={dist:.3f}")


if __name__ == "__main__":
    run()
