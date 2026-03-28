"""
Fractal Hausdorff distance diagnostic experiment.

Tracks the Hausdorff distance between the learned node population and the
prior node population in μ-space after each observation pass.

Hypothesis (HPM prediction):
    An Observer seeded with structural priors should show DECREASING Hausdorff
    distance over passes — learned nodes converge toward the prior attractor.
    An Observer with no priors accumulates nodes in arbitrary regions of
    observation space, showing no systematic convergence.

Hausdorff(A, B) = max(max_{a∈A} min_{b∈B} d(a,b),
                      max_{b∈B} min_{a∈A} d(a,b))

For the world-model condition:
  A = learned nodes (Observer-created)
  B = prior nodes (fixed)

  Decreasing Hausdorff = learned nodes are closing in on the prior attractor.
  Stable/increasing = learned nodes are not converging.

For the no-priors condition:
  A = all nodes (pass k)
  B = all nodes (pass k-1)

  This measures how much the no-priors node population shifts between passes —
  high variance = unstable, fragmented learning.

Run:
    python3 -m hpm_fractal_node.experiments.experiment_fractal_hausdorff
"""

from __future__ import annotations

import json
import glob
import numpy as np

from hfn import HFN, Forest, Observer, calibrate_tau, hausdorff_distance
from hpm_fractal_node.arc.arc_world_model import build_world_model


def load_3x3_colour(
    data_dir: str = "data/ARC-AGI-2/data/training",
) -> list[np.ndarray]:
    records: list[np.ndarray] = []
    for f in sorted(glob.glob(f"{data_dir}/*.json")):
        d = json.load(open(f))
        for ex in d["train"]:
            grid = ex["input"]
            if len(grid) != 3 or len(grid[0]) != 3:
                continue
            vec = np.array([[cell / 9.0 for cell in row] for row in grid]).flatten()
            records.append(vec)
    return records


def run(
    n_passes: int = 8,
    seed: int = 42,
    data_dir: str = "data/ARC-AGI-2/data/training",
) -> None:
    rng = np.random.default_rng(seed)
    D = 9

    observations = load_3x3_colour(data_dir)
    if not observations:
        print("No 3×3 observations found. Check data_dir.")
        return
    print(f"Loaded {len(observations)} colour-encoded observations (3×3)\n")

    # --- World-model condition ---
    forest_wm, registry = build_world_model(rows=3, cols=3)
    prior_ids = set(registry.keys())
    tau = calibrate_tau(D, sigma_scale=2.0, margin=1.0)
    obs_wm = Observer(forest_wm, tau=tau, protected_ids=prior_ids)

    # --- No-priors condition ---
    forest_np = Forest(D=D, forest_id="no_priors")
    forest_np.register(HFN(mu=np.full(D, 0.5), sigma=np.eye(D) * 4.0, id="prior_uniform"))
    obs_np = Observer(forest_np, tau=tau, protected_ids={"prior_uniform"})

    prior_nodes = [n for n in forest_wm.active_nodes() if n.id in prior_ids]

    print(f"{'Pass':>4}  {'WM Hausdorff':>14}  {'WM learned N':>13}  "
          f"{'NP shift':>10}  {'NP nodes':>9}")
    print("-" * 60)

    prev_np_nodes: list = []

    for p in range(1, n_passes + 1):
        order = rng.permutation(len(observations))
        for idx in order:
            obs_wm.observe(observations[idx])
            obs_np.observe(observations[idx])

        # World-model: Hausdorff between learned nodes and priors
        wm_all = list(forest_wm.active_nodes())
        learned = [n for n in wm_all if n.id not in prior_ids]
        if learned:
            hd_wm = hausdorff_distance(learned, prior_nodes)
        else:
            hd_wm = float("nan")

        # No-priors: Hausdorff between current and previous node set (stability)
        np_nodes = list(forest_np.active_nodes())
        if prev_np_nodes and np_nodes:
            hd_np_shift = hausdorff_distance(np_nodes, prev_np_nodes)
        else:
            hd_np_shift = float("nan")
        prev_np_nodes = np_nodes

        def fmt(v: float) -> str:
            return f"{v:.4f}" if v == v else "nan"

        print(f"{p:>4}  {fmt(hd_wm):>14}  {len(learned):>13}  "
              f"{fmt(hd_np_shift):>10}  {len(np_nodes):>9}")

    # Final: show which learned nodes are closest / furthest from any prior
    wm_all = list(forest_wm.active_nodes())
    learned = [n for n in wm_all if n.id not in prior_ids]
    if learned:
        print(f"\n=== Learned node distances to nearest prior ===")
        rows = []
        for node in learned:
            nearest = min(prior_nodes, key=lambda p: float(np.linalg.norm(node.mu - p.mu)))
            dist = float(np.linalg.norm(node.mu - nearest.mu))
            rows.append((dist, node.id, nearest.id))
        rows.sort()
        for dist, nid, pid in rows:
            print(f"  dist={dist:.4f}  learned={nid[:36]}  nearest_prior={pid[:30]}")

    print()
    print("Interpretation:")
    print("  WM Hausdorff — distance between learned nodes and nearest prior.")
    print("    Decreasing ⟹ convergence toward prior attractor (IFS prediction).")
    print("    Stable/increasing ⟹ learned nodes exploring novel territory.")
    print()
    print("  NP shift — Hausdorff distance between consecutive no-priors node sets.")
    print("    High/variable ⟹ unstable, fragmented learning (no attractor seed).")
    print("    Low/decreasing ⟹ no-priors Observer stabilising on its own attractor.")


if __name__ == "__main__":
    run()
