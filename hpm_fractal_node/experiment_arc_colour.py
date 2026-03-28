"""
ARC colour-encoded Observer experiment.

Uses value encoding (cell_value / 9.0) instead of binary, so colour
identity is preserved in the observation vector. Tests whether the
colour priors improve coverage over binary encoding.

Run: python3 -m hpm_fractal_node.experiment_arc_colour
"""

import json
import glob
import numpy as np
from collections import defaultdict

from hpm_fractal_node.observer import Observer
from hpm_fractal_node.arc_world_model import build_world_model


def load_3x3_colour(data_dir: str = "data/ARC-AGI-2/data/training") -> list[tuple[str, np.ndarray]]:
    """Load 3x3 grids with value encoding: cell_value / 9.0."""
    records = []
    for f in sorted(glob.glob(f"{data_dir}/*.json")):
        d = json.load(open(f))
        puzzle_id = f.split("/")[-1].replace(".json", "")
        for ex in d["train"]:
            grid = ex["input"]
            if len(grid) != 3 or len(grid[0]) != 3:
                continue
            vec = np.array([[cell / 9.0 for cell in row]
                            for row in grid]).flatten()
            records.append((puzzle_id, vec))
    return records


def run(n_passes: int = 5, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    records = load_3x3_colour()
    print(f"Loaded {len(records)} colour-encoded observations\n")

    # Show colour distribution
    all_values = np.array([vec for _, vec in records]).flatten()
    unique_vals, counts = np.unique((all_values * 9).round().astype(int), return_counts=True)
    print("Colour distribution (value: count):")
    for v, c in zip(unique_vals, counts):
        print(f"  {v}: {c:5d}  {'█' * (c // 20)}")
    print()

    forest, registry = build_world_model(rows=3, cols=3)
    prior_ids = set(registry.keys())

    colour_nodes = [n for n in forest.active_nodes() if "colour" in n.id]
    print(f"World model: {len(forest)} nodes ({len(colour_nodes)} colour priors)")

    D = 9
    baseline = D / 2 * np.log(2 * np.pi)
    tau = baseline + 1.0
    residual_threshold = baseline + 2.5

    obs = Observer(
        forest,
        tau=tau,
        budget=10,
        lambda_complexity=0.05,
        alpha_gain=0.15,
        beta_loss=0.05,
        absorption_overlap_threshold=0.6,
        absorption_miss_threshold=8,
        residual_surprise_threshold=residual_threshold,
        compression_cooccurrence_threshold=4,
        w_init=0.1,
        protected_ids=prior_ids,
    )

    for _ in range(n_passes):
        indices = rng.permutation(len(records))
        for i in indices:
            _, vec = records[i]
            obs.observe(vec)

    active = forest.active_nodes()
    surviving_priors = [n for n in active if n.id in prior_ids]
    new_nodes = [n for n in active if n.id not in prior_ids]

    print(f"\n=== After {n_passes} passes ===")
    print(f"Surviving priors: {len(surviving_priors)}")
    print(f"New nodes:        {len(new_nodes)}")
    print(f"Absorbed:         {len(obs.absorbed_ids)}")

    # Coverage
    ep = en = un = 0
    prior_hits: dict[str, int] = defaultdict(int)
    layer_hits: dict[str, int] = defaultdict(int)

    for _, vec in records:
        best = None
        bs = float("inf")
        for node in active:
            s = -node.log_prob(vec)
            if s < tau and node.id in prior_ids and s < bs:
                bs = s
                best = node
        if best:
            ep += 1
            prior_hits[best.id] += 1
            if "colour" in best.id:
                layer_hits["colour"] += 1
            elif "primitive" in best.id:
                layer_hits["primitives"] += 1
            elif "relationship" in best.id or best.id.startswith("prim_"):
                layer_hits["relationships"] += 1
            elif any(x in best.id for x in ["object", "scene", "rule"]):
                layer_hits["semantic"] += 1
            elif "encoder" in best.id:
                layer_hits["encoder"] += 1
            else:
                layer_hits["structural_priors"] += 1
        elif any(-n.log_prob(vec) < tau for n in active if n.id not in prior_ids):
            en += 1
        else:
            un += 1

    total = len(records)
    print(f"\n=== Explanation coverage ({total} observations) ===")
    print(f"  By priors:    {ep} ({100*ep/total:.0f}%)")
    print(f"  By new nodes: {en} ({100*en/total:.0f}%)")
    print(f"  Unexplained:  {un} ({100*un/total:.0f}%)")
    print(f"  Absorbed priors: {sum(1 for nid in obs.absorbed_ids if nid in prior_ids)}")

    print(f"\n=== Coverage by layer ===")
    for layer, count in sorted(layer_hits.items(), key=lambda x: -x[1]):
        print(f"  {layer:<20} {count} observations")

    print(f"\n=== Most-used priors ===")
    for pid, count in sorted(prior_hits.items(), key=lambda x: -x[1])[:12]:
        print(f"  {pid:<40}  {count}")

    # Show top new nodes with their colour profile
    if new_nodes:
        print(f"\n=== Top new nodes (colour profile) ===")
        for node in sorted(new_nodes, key=lambda n: -obs.get_weight(n.id))[:6]:
            w = obs.get_weight(node.id)
            vals = (node.mu * 9).round().astype(int).reshape(3, 3)
            rows_str = ["".join(f"{v}" for v in row) for row in vals]
            print(f"  {node.id[:36]}  w={w:.3f}")
            for r in rows_str:
                print(f"    {r}")


if __name__ == "__main__":
    run()
