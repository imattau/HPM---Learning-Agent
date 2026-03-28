"""
ARC-AGI-2 Observer experiment.

Feeds binarized 3x3 ARC input grids (as 9-dim vectors) into an Observer
and reports what structure emerges — without any hand-crafting.

Encoding: each cell is 0 (empty) or 1 (occupied), regardless of colour.
This captures spatial shape, which is what tiling rules operate on.

Run: python3 -m hpm_fractal_node.experiment_arc_observer
"""

import json
import glob
import numpy as np
from collections import defaultdict

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_3x3_inputs(data_dir: str = "data/ARC-AGI-2/data/training") -> list[tuple[str, np.ndarray]]:
    """Load all 3x3 input grids from ARC training set, binarized."""
    records = []
    for f in sorted(glob.glob(f"{data_dir}/*.json")):
        d = json.load(open(f))
        puzzle_id = f.split("/")[-1].replace(".json", "")
        for ex in d["train"]:
            grid = ex["input"]
            if len(grid) != 3 or len(grid[0]) != 3:
                continue
            vec = np.array([[1.0 if cell != 0 else 0.0 for cell in row]
                            for row in grid]).flatten()
            records.append((puzzle_id, vec))
    return records


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run(n_passes: int = 5, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    records = load_3x3_inputs()
    D = 9
    print(f"Loaded {len(records)} observations from {len(set(r[0] for r in records))} puzzles\n")

    # Calibrate tau to the dimensionality of the space.
    # For a D-dim Gaussian with sigma=I, the minimum kl_surprise (at its own
    # mean) is D/2 * log(2π) ≈ 12.7 for D=9.  tau must be above this baseline
    # or no node will ever explain anything.  We add a margin so nodes explain
    # grids differing by ~2 cells (each cell difference adds 0.5 to surprise).
    baseline_surprise = D / 2 * np.log(2 * np.pi)   # ≈ 12.7
    tau = baseline_surprise + 1.0                     # explains grids ≤ ~2 cells away
    residual_threshold = baseline_surprise + 2.5      # new node only for truly novel grids
    print(f"D={D}, baseline_surprise={baseline_surprise:.1f}, tau={tau:.1f}, residual_threshold={residual_threshold:.1f}\n")

    forest = Forest(D=D, forest_id="arc_forest")
    obs = Observer(
        forest,
        tau=tau,
        budget=5,
        lambda_complexity=0.05,
        alpha_gain=0.15,
        beta_loss=0.05,
        absorption_overlap_threshold=0.6,
        absorption_miss_threshold=4,
        residual_surprise_threshold=residual_threshold,
        compression_cooccurrence_threshold=4,
        w_init=0.1,
    )

    # Multiple passes, shuffled each time
    for pass_num in range(n_passes):
        indices = rng.permutation(len(records))
        for i in indices:
            _, vec = records[i]
            obs.observe(vec)

    # --- Report ---
    print(f"=== Forest after {n_passes} passes ===")
    print(f"Active nodes: {len(forest)}")
    print(f"Absorbed nodes: {len(obs.absorbed_ids)}\n")

    # Sort nodes by weight descending
    nodes = sorted(forest.active_nodes(), key=lambda n: obs.get_weight(n.id), reverse=True)

    print(f"{'Node ID':<40} {'Weight':>7} {'Score':>7}  Shape (mu)")
    print("-" * 80)
    for node in nodes:
        w = obs.get_weight(node.id)
        s = obs.get_score(node.id)
        mu_str = np.array2string(node.mu.round(2), separator=",", max_line_width=60)
        print(f"{node.id:<40} {w:>7.3f} {s:>7.3f}  {mu_str}")

    # --- Cluster analysis ---
    # For each node, find which puzzle inputs it best explains
    print("\n=== Node → Puzzle mapping (top explainers per node) ===")
    for node in nodes[:10]:  # top 10 by weight
        covered = []
        for puzzle_id, vec in records:
            surprise = -node.log_prob(vec)
            if surprise < tau:
                covered.append(puzzle_id)
        puzzle_counts = defaultdict(int)
        for pid in covered:
            puzzle_counts[pid] += 1
        top = sorted(puzzle_counts.items(), key=lambda x: -x[1])[:5]
        print(f"  {node.id[:38]}: {top}")

    # --- Shared structure check ---
    print("\n=== Nodes explaining inputs from 2+ puzzles ===")
    cross_puzzle_nodes = []
    for node in nodes:
        puzzles_explained = set()
        for puzzle_id, vec in records:
            if -node.log_prob(vec) < tau:
                puzzles_explained.add(puzzle_id)
        if len(puzzles_explained) >= 2:
            cross_puzzle_nodes.append((node, puzzles_explained))

    print(f"Found {len(cross_puzzle_nodes)} cross-puzzle nodes")
    for node, puzzles in sorted(cross_puzzle_nodes, key=lambda x: -len(x[1]))[:10]:
        w = obs.get_weight(node.id)
        print(f"  {node.id[:38]} (w={w:.3f}): covers {len(puzzles)} puzzles → {sorted(puzzles)[:5]}")

    # --- Unique shapes in discovered nodes ---
    print("\n=== Discovered node shapes (mu binarized) ===")
    seen_shapes = set()
    for node in nodes:
        shape = tuple((node.mu > 0.5).astype(int))
        if shape not in seen_shapes:
            seen_shapes.add(shape)
            grid = np.array(shape).reshape(3, 3)
            print(f"  {node.id[:30]}:")
            for row in grid:
                print(f"    {''.join('X' if c else '.' for c in row)}")


if __name__ == "__main__":
    run()
