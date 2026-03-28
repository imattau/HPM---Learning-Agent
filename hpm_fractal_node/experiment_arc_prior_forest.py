"""
ARC Observer experiment with pre-populated prior Forest.

Runs the Observer against ARC 3x3 training data starting from a world
model that already contains structural knowledge: spatial patterns,
transformation priors, and relationship priors — all as HFN nodes.

The experiment shows:
  - Which priors gain weight (are confirmed by observations)
  - Which priors lose weight or get absorbed (don't explain the data)
  - What new nodes emerge (genuine novelty beyond the priors)
  - How far the prior hierarchy can explain observations before residual fires

Run: python3 -m hpm_fractal_node.experiment_arc_prior_forest
"""

import json
import glob
import numpy as np
from collections import defaultdict

from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hpm_fractal_node.arc_prior_forest import build_prior_forest, D, CELL_NAMES


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
    print(f"Loaded {len(records)} observations from {len(set(r[0] for r in records))} puzzles\n")

    forest, prior_registry = build_prior_forest()
    prior_ids = set(prior_registry.keys())

    print(f"Prior Forest: {len(forest)} top-level nodes")
    for n in forest.active_nodes():
        print(f"  {n.id}  children={len(n.children())}")
    print()

    baseline = D / 2 * np.log(2 * np.pi)
    tau = baseline + 1.0
    residual_threshold = baseline + 2.5
    print(f"tau={tau:.2f}, residual_threshold={residual_threshold:.2f}\n")

    obs = Observer(
        forest,
        tau=tau,
        budget=8,
        lambda_complexity=0.05,
        alpha_gain=0.15,
        beta_loss=0.05,
        absorption_overlap_threshold=0.6,
        absorption_miss_threshold=12,
        residual_surprise_threshold=residual_threshold,
        compression_cooccurrence_threshold=4,
        w_init=0.1,
        protected_ids=prior_ids,  # priors are invariant structural knowledge
    )

    # Run passes
    for _ in range(n_passes):
        indices = rng.permutation(len(records))
        for i in indices:
            _, vec = records[i]
            obs.observe(vec)

    # --- Report ---
    active = forest.active_nodes()
    surviving_priors = [n for n in active if n.id in prior_ids]
    new_nodes = [n for n in active if n.id not in prior_ids]

    print(f"=== After {n_passes} passes ===")
    print(f"Surviving priors:  {len(surviving_priors)}")
    print(f"New nodes:         {len(new_nodes)}")
    print(f"Absorbed:          {len(obs.absorbed_ids)}")
    print()

    # Prior weights
    print("=== Prior node weights ===")
    all_nodes_sorted = sorted(active, key=lambda n: -obs.get_weight(n.id))
    for node in all_nodes_sorted:
        w = obs.get_weight(node.id)
        s = obs.get_score(node.id)
        label = "PRIOR" if node.id in prior_ids else "NEW  "
        shape = np.array(node.mu > 0.5, dtype=int).reshape(3, 3)
        rows = '|'.join(''.join('X' if c else '.' for c in row) for row in shape)
        print(f"  [{label}] {node.id:<36}  w={w:.3f}  s={s:.4f}  {rows}")

    # How well did priors cover observations?
    explained_by_prior = 0
    explained_by_new = 0
    unexplained = 0
    prior_hits = defaultdict(int)

    for _, vec in records:
        best_prior = None
        best_surprise = float('inf')
        for node in active:
            surprise = -node.log_prob(vec)
            if surprise < tau and node.id in prior_ids:
                if surprise < best_surprise:
                    best_surprise = surprise
                    best_prior = node
        if best_prior:
            explained_by_prior += 1
            prior_hits[best_prior.id] += 1
        elif any(-n.log_prob(vec) < tau for n in active if n.id not in prior_ids):
            explained_by_new += 1
        else:
            unexplained += 1

    total = len(records)
    print(f"\n=== Explanation coverage ({total} observations) ===")
    print(f"  By surviving priors:  {explained_by_prior} ({100*explained_by_prior/total:.0f}%)")
    print(f"  By new nodes:         {explained_by_new} ({100*explained_by_new/total:.0f}%)")
    print(f"  Unexplained:          {unexplained} ({100*unexplained/total:.0f}%)")

    if prior_hits:
        print(f"\n  Most-used priors:")
        for pid, count in sorted(prior_hits.items(), key=lambda x: -x[1])[:5]:
            print(f"    {pid:<36}  explains {count} observations")

    # Which priors got absorbed / lost?
    absorbed_priors = [nid for nid in obs.absorbed_ids if nid in prior_ids]
    print(f"\n=== Absorbed priors: {len(absorbed_priors)} ===")
    for nid in absorbed_priors:
        print(f"  {nid}")

    # New nodes that emerged
    if new_nodes:
        print(f"\n=== New nodes (emerged from observations) ===")
        for node in sorted(new_nodes, key=lambda n: -obs.get_weight(n.id)):
            w = obs.get_weight(node.id)
            shape = np.array(node.mu > 0.5, dtype=int).reshape(3, 3)
            rows = [''.join('X' if c else '.' for c in row) for row in shape]
            puzzles = set(pid for pid, vec in records if -node.log_prob(vec) < tau)
            print(f"  {node.id[:36]}  w={w:.3f}  covers {len(puzzles)} puzzles")
            for r in rows:
                print(f"    {r}")


if __name__ == "__main__":
    run()
