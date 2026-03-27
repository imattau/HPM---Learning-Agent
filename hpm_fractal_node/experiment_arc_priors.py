"""
ARC-AGI-2 Observer experiment with cell-position priors as structural children.

Architecture:
  - 9 cell-position nodes are pre-defined as L1 building blocks (not Forest nodes).
  - The Observer creates pattern nodes from whole-grid observations.
  - Each new pattern node automatically gets the filled cell priors attached as children.
  - The Forest contains patterns; cell priors are shared structural leaves beneath them.

This shows:
  - Cell positions as known priors shape pattern structure without competing in the Forest.
  - Patterns discovered from observations are structurally decomposed into their cells.
  - The same cell node is shared across all patterns that contain that cell.
  - Compression creates higher-order nodes whose children reference cell priors.

Run: python3 -m hpm_fractal_node.experiment_arc_priors
"""

import json
import glob
import numpy as np
from collections import defaultdict

from hpm_fractal_node.hfn import HFN
from hpm_fractal_node.forest import Forest
from hpm_fractal_node.observer import Observer


D = 9
CELL_NAMES = [f"cell_{r}{c}" for r in range(3) for c in range(3)]


# ---------------------------------------------------------------------------
# L1 cell priors — pre-defined building blocks
# Not registered in the Forest. Attached as children to pattern nodes.
# ---------------------------------------------------------------------------

def make_cell_priors() -> dict[str, HFN]:
    priors = {}
    for i, name in enumerate(CELL_NAMES):
        mu = np.zeros(D)
        mu[i] = 1.0
        priors[name] = HFN(mu=mu, sigma=np.eye(D), id=name)
    return priors


CELL_PRIORS = make_cell_priors()


# ---------------------------------------------------------------------------
# ArcObserver — extends Observer to attach cell priors on node creation
# ---------------------------------------------------------------------------

class ArcObserver(Observer):
    """
    Observer that attaches cell-position priors as structural children
    whenever a new pattern node is created from residual surprise.
    """

    def _check_residual_surprise(self, x: np.ndarray, result) -> None:
        from hpm_fractal_node.observer import ExplanationResult  # avoid circular
        should_create = (
            len(self.forest) == 0
            or result.residual_surprise >= self.residual_surprise_threshold
        )
        if not should_create:
            return

        new_node = HFN(
            mu=x.copy(),
            sigma=np.eye(D),
            id=f"pattern_{len(self.forest)}",
        )

        # Attach filled cell priors as structural children
        for i, name in enumerate(CELL_NAMES):
            if x[i] > 0.5:
                new_node._children.append(CELL_PRIORS[name])

        self.register(new_node)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_3x3_inputs(data_dir: str = "data/ARC-AGI-2/data/training") -> list[tuple[str, np.ndarray]]:
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
# Run
# ---------------------------------------------------------------------------

def run(n_passes: int = 5, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    records = load_3x3_inputs()
    print(f"Loaded {len(records)} observations from {len(set(r[0] for r in records))} puzzles")
    print(f"Pre-defined {len(CELL_PRIORS)} L1 cell priors (building blocks, not Forest nodes)\n")

    # Calibrate tau for sigma=I nodes.
    # baseline kl_surprise at own mean = D/2*log(2π) ≈ 8.27
    # margin=1.0: explains grids within ~2 cells
    baseline = D / 2 * np.log(2 * np.pi)
    tau = baseline + 1.0
    residual_threshold = baseline + 2.5
    print(f"D={D}, baseline={baseline:.2f}, tau={tau:.2f}, residual_threshold={residual_threshold:.2f}\n")

    forest = Forest(D=D, forest_id="arc_forest")
    obs = ArcObserver(
        forest,
        tau=tau,
        budget=8,
        lambda_complexity=0.05,
        alpha_gain=0.15,
        beta_loss=0.05,
        absorption_overlap_threshold=0.6,
        absorption_miss_threshold=6,
        residual_surprise_threshold=residual_threshold,
        compression_cooccurrence_threshold=4,
        w_init=0.1,
    )

    for pass_num in range(n_passes):
        indices = rng.permutation(len(records))
        for i in indices:
            _, vec = records[i]
            obs.observe(vec)

    # --- Report ---
    active = forest.active_nodes()
    pattern_nodes = [n for n in active if n.id.startswith("pattern_")]
    compressed_nodes = [n for n in active if n.id.startswith("compressed")]

    print(f"=== Forest after {n_passes} passes ===")
    print(f"Pattern nodes (from observations): {len(pattern_nodes)}")
    print(f"Compressed nodes (discovered):     {len(compressed_nodes)}")
    print(f"Total active:                      {len(active)}")
    print(f"Absorbed:                          {len(obs.absorbed_ids)}\n")

    # Show pattern nodes with their cell structure
    print("=== Pattern nodes with cell children ===")
    top_patterns = sorted(active, key=lambda n: -obs.get_weight(n.id))[:12]
    for node in top_patterns:
        w = obs.get_weight(node.id)
        shape = np.array(node.mu > 0.5, dtype=int).reshape(3, 3)
        rows = [''.join('X' if c else '.' for c in row) for row in shape]
        cell_children = [c.id for c in node.children() if c.id in CELL_PRIORS]
        # Recurse one level for compressed nodes
        if not cell_children:
            for child in node.children():
                cell_children += [c.id for c in child.children() if c.id in CELL_PRIORS]
        print(f"  {node.id[:36]}  w={w:.3f}  cells={len(cell_children)}")
        for r in rows:
            print(f"    {r}")
        if cell_children:
            print(f"    → {cell_children}")

    # Cell prior usage: how many pattern nodes reference each cell?
    print("\n=== Cell prior usage (how many patterns reference each cell) ===")
    cell_usage = defaultdict(int)
    for node in active:
        def count_cells(n, depth=0):
            if depth > 3:
                return
            for c in n.children():
                if c.id in CELL_PRIORS:
                    cell_usage[c.id] += 1
                else:
                    count_cells(c, depth + 1)
        count_cells(node)

    grid_usage = np.zeros((3, 3), dtype=int)
    for name, count in cell_usage.items():
        idx = CELL_NAMES.index(name)
        grid_usage[idx // 3, idx % 3] = count

    print("  Usage counts (3x3 grid):")
    for row in grid_usage:
        print("  " + "  ".join(f"{v:3d}" for v in row))

    # Cross-puzzle coverage
    print(f"\n=== Cross-puzzle coverage ===")
    cross = []
    for node in active:
        puzzles = set()
        for pid, vec in records:
            if -node.log_prob(vec) < tau:
                puzzles.add(pid)
        if len(puzzles) >= 3:
            cross.append((node, puzzles))
    cross.sort(key=lambda x: -len(x[1]))
    print(f"Nodes covering 3+ puzzles: {len(cross)}")
    for node, puzzles in cross[:8]:
        cell_children = [c.id for c in node.children() if c.id in CELL_PRIORS]
        print(f"  {node.id[:36]}  w={obs.get_weight(node.id):.3f}  covers {len(puzzles)} puzzles  cells={cell_children}")

    # Shared cell node identity check
    print(f"\n=== Shared cell node identity ===")
    for cell_name in ["cell_11", "cell_00", "cell_22"]:
        references = []
        for node in active:
            if CELL_PRIORS[cell_name] in node.children():
                references.append(node.id[:20])
        if references:
            print(f"  {cell_name} referenced by {len(references)} active nodes (same object)")
            print(f"    e.g. {references[:4]}")


if __name__ == "__main__":
    run()
