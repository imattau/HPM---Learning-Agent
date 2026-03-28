"""
dSprites experiment: generative factor alignment.

Runs the Observer over dSprites 16x16 observations and measures whether
learned nodes align with the dataset's known generative factors (shape,
scale, orientation, position).

A node with low shape entropy fired predominantly on one shape type — the
Observer discovered a latent generative factor without supervision.

Usage:
    PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_dsprites.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn import Observer, calibrate_tau
from hfn.fractal import hausdorff_distance
from hpm_fractal_node.dsprites.dsprites_loader import (
    load_dsprites, factor_names, shape_name, D,
)
from hpm_fractal_node.dsprites.dsprites_world_model import build_dsprites_world_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SAMPLES = 2000
N_PASSES = 3
SEED = 42

TAU_SIGMA = 1.5
TAU_MARGIN = 3.0


def entropy(counts: dict) -> float:
    """Shannon entropy of a count distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([v / total for v in counts.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def max_entropy(n_values: int) -> float:
    return np.log2(n_values) if n_values > 1 else 0.0


def purity(counts: dict) -> float:
    """Fraction of observations matching the dominant factor value."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(counts.values()) / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading {N_SAMPLES} dSprites samples (16x16, D={D}) ...")
    images, latents = load_dsprites(n_samples=N_SAMPLES, seed=SEED)
    print(f"  images: {images.shape}  latents: {latents.shape}")
    print(f"  value range: [{images.min():.3f}, {images.max():.3f}]")

    print("\nBuilding world model ...")
    forest, prior_ids = build_dsprites_world_model()
    print(f"  {len(forest._registry)} priors, {len(prior_ids)} protected")

    tau = calibrate_tau(D, sigma_scale=TAU_SIGMA, margin=TAU_MARGIN)
    print(f"  tau = {tau:.2f}")

    obs = Observer(
        forest,
        tau=tau,
        protected_ids=prior_ids,
        recombination_strategy="nearest_prior",
        hausdorff_absorption_threshold=0.15,
        persistence_guided_absorption=True,
        lacunarity_guided_creation=True,
        lacunarity_creation_radius=0.08,
        multifractal_guided_absorption=True,
        multifractal_crowding_radius=0.12,
    )

    # ------------------------------------------------------------------
    # Multi-pass observation — track which node explains each observation
    # ------------------------------------------------------------------
    # node_id -> list of latent vectors for observations it best explained
    node_explanations: dict[str, list[np.ndarray]] = defaultdict(list)

    factor_n_values = [3, 6, 40, 32, 32]  # shape, scale, orientation, pos_x, pos_y
    fnames = factor_names()

    print(f"\nRunning {N_PASSES} passes over {N_SAMPLES} observations ...")
    for p in range(N_PASSES):
        n_explained = 0
        n_unexplained = 0
        rng = np.random.default_rng(SEED + p)
        order = rng.permutation(len(images))

        for i in order:
            x = images[i].astype(np.float64)
            result = obs.observe(x)

            if result.explanation_tree:
                # Best-explaining node = highest accuracy score
                best_id = max(result.accuracy_scores, key=lambda k: result.accuracy_scores[k])
                node_explanations[best_id].append(latents[i])
                n_explained += 1
            else:
                n_unexplained += 1

        n_total = len(images)
        print(f"  Pass {p+1}: explained {n_explained}/{n_total} "
              f"({100*n_explained/n_total:.1f}%), "
              f"unexplained {n_unexplained} "
              f"({100*n_unexplained/n_total:.1f}%)")

    # ------------------------------------------------------------------
    # Coverage summary
    # ------------------------------------------------------------------
    prior_explained = sum(
        len(v) for k, v in node_explanations.items() if k in prior_ids
    )
    learned_nodes = [k for k in node_explanations if k not in prior_ids]
    learned_explained = sum(len(node_explanations[k]) for k in learned_nodes)
    total_attributed = prior_explained + learned_explained
    n_obs_total = N_PASSES * N_SAMPLES

    print(f"\n=== Coverage (over all passes, {n_obs_total} total obs) ===")
    print(f"  Prior nodes explained:   {prior_explained:5d} ({100*prior_explained/n_obs_total:.1f}%)")
    print(f"  Learned nodes explained: {learned_explained:5d} ({100*learned_explained/n_obs_total:.1f}%)")
    print(f"  Total attributed:        {total_attributed:5d} ({100*total_attributed/n_obs_total:.1f}%)")
    print(f"  Learned node count:      {len(learned_nodes)}")

    # ------------------------------------------------------------------
    # Generative factor alignment — prior nodes
    # ------------------------------------------------------------------
    print(f"\n=== Prior node factor alignment (top priors by coverage) ===")

    prior_rows = [
        (k, node_explanations[k])
        for k in prior_ids
        if len(node_explanations.get(k, [])) > 0
    ]
    prior_rows.sort(key=lambda r: len(r[1]), reverse=True)

    for node_id, lats in prior_rows[:15]:
        lats_arr = np.array(lats)
        n = len(lats_arr)
        shape_counts: dict[int, int] = defaultdict(int)
        for row in lats_arr:
            shape_counts[int(row[0])] += 1
        dom_shape = max(shape_counts, key=lambda k: shape_counts[k])
        pur = purity(shape_counts)
        ent = entropy(shape_counts)
        shape_str = " ".join(f"{shape_name(s)}:{c}" for s, c in sorted(shape_counts.items()))
        print(f"  {node_id:<35s}  n={n:4d}  shape_purity={pur:.2f}  H={ent:.2f}  [{shape_str}]")

    # ------------------------------------------------------------------
    # Generative factor alignment — learned nodes
    # ------------------------------------------------------------------
    if not learned_nodes:
        print("\nNo learned nodes — try increasing N_PASSES or lowering tau_margin.")
        return

    print(f"\n=== Learned node factor alignment ({len(learned_nodes)} nodes) ===")

    rows = []
    for node_id in learned_nodes:
        lats_arr = np.array(node_explanations[node_id])
        n = len(lats_arr)
        row = {"id": node_id, "n": n}
        for fi, fname in enumerate(fnames):
            counts: dict[int, int] = defaultdict(int)
            for lat in lats_arr:
                counts[int(lat[fi])] += 1
            row[f"{fname}_purity"] = purity(counts)
            row[f"{fname}_entropy"] = entropy(counts)
            row[f"{fname}_dominant"] = max(counts, key=lambda k: counts[k])
        rows.append(row)

    rows.sort(key=lambda r: r["n"], reverse=True)

    header = f"  {'Node ID':<20s}  {'n':>4s}  {'shape_pur':>9s}  {'scale_pur':>9s}  {'pos_x_pur':>9s}  {'pos_y_pur':>9s}  dominant_shape"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        dom_shape = shape_name(r["shape_dominant"])
        print(
            f"  {r['id']:<20s}  {r['n']:>4d}  "
            f"{r['shape_purity']:>9.3f}  {r['scale_purity']:>9.3f}  "
            f"{r['pos_x_purity']:>9.3f}  {r['pos_y_purity']:>9.3f}  "
            f"{dom_shape}"
        )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    shape_purities = [r["shape_purity"] for r in rows if r["n"] >= 5]
    scale_purities = [r["scale_purity"] for r in rows if r["n"] >= 5]
    pos_x_purities = [r["pos_x_purity"] for r in rows if r["n"] >= 5]

    print(f"\n=== Factor purity summary (nodes with n>=5) ===")
    if shape_purities:
        print(f"  Shape:    mean={np.mean(shape_purities):.3f}  max={np.max(shape_purities):.3f}")
        print(f"  Scale:    mean={np.mean(scale_purities):.3f}  max={np.max(scale_purities):.3f}")
        print(f"  Pos X:    mean={np.mean(pos_x_purities):.3f}  max={np.max(pos_x_purities):.3f}")
        print(f"  Random baseline: shape={1/3:.3f}, scale={1/6:.3f}, pos_x={1/32:.3f}")
    else:
        print("  Insufficient learned nodes with n>=5 for statistics.")

    # ------------------------------------------------------------------
    # Fractal diagnostics
    # ------------------------------------------------------------------
    prior_mus = np.array([forest._registry[k].mu for k in prior_ids])
    learned_mus_list = [forest._registry[k].mu for k in forest._registry if k not in prior_ids]

    if learned_mus_list:
        learned_mus = np.array(learned_mus_list)
        hd = hausdorff_distance(learned_mus, prior_mus)
        print(f"\n=== Fractal diagnostics ===")
        print(f"  Learned nodes:          {len(learned_mus_list)}")
        print(f"  Hausdorff(learned, priors): {hd:.4f}")

    print(f"\n=== Absorbed nodes: {len(obs.absorbed_ids)} ===")


if __name__ == "__main__":
    main()
