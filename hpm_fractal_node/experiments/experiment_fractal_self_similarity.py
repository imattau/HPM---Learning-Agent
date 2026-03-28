"""
Fractal self-similarity diagnostic experiment.

Compares two conditions across multiple observation passes over ARC 3x3 grids:

  World-model condition  — Observer seeded with ARC structural priors
                           (primitives, relationships, colour priors, etc.)
  No-priors condition    — Observer with a single broad uninformative prior

Hypothesis (HPM prediction):
    An Observer seeded with a world model should converge to a node population
    whose μ-space distribution is more self-similar (lower CV score) than one
    with no priors, because the contracting maps implicit in the world model
    drive the IFS attractor to have coherent structure at multiple scales.

Metric:
    self_similarity_score(nodes) = CV of consecutive log-count differences.
    Lower ⟹ more self-similar; 0.0 = perfect power-law scaling.

Run:
    python3 -m hpm_fractal_node.experiments.experiment_fractal_self_similarity
"""

from __future__ import annotations

import json
import glob
import numpy as np

from hfn import HFN, Forest, Observer, calibrate_tau, self_similarity_score
from hpm_fractal_node.arc.arc_world_model import build_world_model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_3x3_inputs(
    data_dir: str = "data/ARC-AGI-2/data/training",
) -> list[np.ndarray]:
    """Return all 3×3 ARC training inputs as flat colour-encoded float vectors."""
    records: list[np.ndarray] = []
    for f in sorted(glob.glob(f"{data_dir}/*.json")):
        d = json.load(open(f))
        for ex in d["train"]:
            grid = ex["input"]
            if len(grid) != 3 or len(grid[0]) != 3:
                continue
            vec = np.array(
                [[cell / 9.0 for cell in row] for row in grid]
            ).flatten()
            records.append(vec)
    return records


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run(
    n_passes: int = 5,
    seed: int = 42,
    n_scales: int = 8,
    data_dir: str = "data/ARC-AGI-2/data/training",
) -> None:
    rng = np.random.default_rng(seed)
    D = 9  # 3×3 grid

    observations = load_3x3_inputs(data_dir)
    if not observations:
        print("No 3x3 observations found. Check data_dir.")
        return
    print(f"Loaded {len(observations)} observations\n")

    # -----------------------------------------------------------------------
    # Condition A: world model with ARC priors
    # -----------------------------------------------------------------------
    forest_wm, registry = build_world_model(rows=3, cols=3)
    prior_ids_wm = set(registry.keys())
    tau_wm = calibrate_tau(D, sigma_scale=2.0, margin=1.0)
    obs_wm = Observer(forest_wm, tau=tau_wm, protected_ids=prior_ids_wm)

    # -----------------------------------------------------------------------
    # Condition B: no priors — single broad uninformative Gaussian
    # Use same tau as world model so conditions are comparable.
    # -----------------------------------------------------------------------
    forest_np = Forest(D=D, forest_id="no_priors")
    prior_node = HFN(
        mu=np.full(D, 0.5),
        sigma=np.eye(D) * 4.0,
        id="prior_uniform",
    )
    forest_np.register(prior_node)
    obs_np = Observer(forest_np, tau=tau_wm, protected_ids={"prior_uniform"})

    # -----------------------------------------------------------------------
    # Run passes
    # -----------------------------------------------------------------------
    print(f"{'Pass':>4}  {'WM learned SS':>14}  {'WM all SS':>10}  {'NP all SS':>10}  "
          f"{'WM nodes':>9}  {'NP nodes':>9}")
    print("-" * 72)

    for p in range(1, n_passes + 1):
        order = rng.permutation(len(observations))
        for idx in order:
            obs_wm.observe(observations[idx])
            obs_np.observe(observations[idx])

        nodes_wm_all = list(forest_wm.active_nodes())
        nodes_wm_learned = [n for n in nodes_wm_all if n.id not in prior_ids_wm]
        nodes_np = list(forest_np.active_nodes())

        ss_wm_learned = self_similarity_score(nodes_wm_learned, n_scales=n_scales)
        ss_wm_all = self_similarity_score(nodes_wm_all, n_scales=n_scales)
        ss_np = self_similarity_score(nodes_np, n_scales=n_scales)

        def fmt(v: float) -> str:
            return f"{v:>10.4f}" if not (v != v) else f"{'nan':>10}"

        print(
            f"{p:>4}  {fmt(ss_wm_learned):>14}  {fmt(ss_wm_all):>10}  {fmt(ss_np):>10}  "
            f"{len(nodes_wm_all):>9}  {len(nodes_np):>9}"
        )

    print()
    print("Columns:")
    print("  WM learned SS — self-similarity of Observer-created nodes only (excl. priors)")
    print("  WM all SS     — self-similarity of all world-model nodes (priors + learned)")
    print("  NP all SS     — self-similarity of no-priors Observer nodes")
    print()
    print("Interpretation:")
    print("  Lower score ⟹ more self-similar (closer to IFS attractor; 0.0 = perfect).")
    print("  HPM prediction: WM learned nodes converge to lower scores than NP.")


if __name__ == "__main__":
    run()
