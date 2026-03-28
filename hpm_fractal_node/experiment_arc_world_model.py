"""
ARC World Model experiment.

Tests the full world model (primitives + relationships + priors + encoder)
against 3x3 ARC training data.

Run: python3 -m hpm_fractal_node.experiment_arc_world_model
"""

import json
import glob
import numpy as np
from collections import defaultdict

from hpm_fractal_node.observer import Observer
from hpm_fractal_node.arc_world_model import build_world_model


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


def run(n_passes: int = 5, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    records = load_3x3_inputs()
    print(f"Loaded {len(records)} observations from {len(set(r[0] for r in records))} puzzles\n")

    forest, registry = build_world_model(rows=3, cols=3)
    prior_ids = set(registry.keys())

    print(f"World model: {len(forest)} top-level nodes")
    print(f"  Primitives:    {sum(1 for n in forest.active_nodes() if 'primitive' in n.id)}")
    print(f"  Relationships: {sum(1 for n in forest.active_nodes() if 'relationship' in n.id or 'prim_' in n.id)}")
    print(f"  Priors:        {sum(1 for n in forest.active_nodes() if n.id.startswith('prior_'))}")
    print(f"  Encoder:       {sum(1 for n in forest.active_nodes() if 'encoder' in n.id)}")
    print()

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

    print(f"=== After {n_passes} passes ===")
    print(f"Surviving priors: {len(surviving_priors)}")
    print(f"New nodes:        {len(new_nodes)}")
    print(f"Absorbed:         {len(obs.absorbed_ids)}")
    print()

    # Coverage
    explained_by_prior = 0
    explained_by_new = 0
    unexplained = 0
    prior_hits: dict[str, int] = defaultdict(int)

    for _, vec in records:
        best_prior = None
        best_surprise = float("inf")
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
    print(f"=== Explanation coverage ({total} observations) ===")
    print(f"  By priors:    {explained_by_prior} ({100*explained_by_prior/total:.0f}%)")
    print(f"  By new nodes: {explained_by_new} ({100*explained_by_new/total:.0f}%)")
    print(f"  Unexplained:  {unexplained} ({100*unexplained/total:.0f}%)")

    print(f"\n  Most-used priors:")
    for pid, count in sorted(prior_hits.items(), key=lambda x: -x[1])[:10]:
        print(f"    {pid:<40}  explains {count} observations")

    print(f"\n=== Absorbed priors: {sum(1 for nid in obs.absorbed_ids if nid in prior_ids)} ===")

    # Which layer explains most?
    print(f"\n=== Coverage by layer ===")
    layer_hits: dict[str, int] = defaultdict(int)
    for pid, count in prior_hits.items():
        if "primitive" in pid:
            layer_hits["primitives"] += count
        elif "relationship" in pid or pid.startswith("prim_"):
            layer_hits["relationships"] += count
        elif "encoder" in pid:
            layer_hits["encoder"] += count
        else:
            layer_hits["priors"] += count
    for layer, count in sorted(layer_hits.items(), key=lambda x: -x[1]):
        print(f"  {layer:<16} {count} observations")

    # Show new nodes that emerged
    if new_nodes:
        print(f"\n=== Top new nodes ===")
        for node in sorted(new_nodes, key=lambda n: -obs.get_weight(n.id))[:8]:
            w = obs.get_weight(node.id)
            shape = np.array(node.mu > 0.5, dtype=int).reshape(3, 3)
            rows_str = "|".join("".join("X" if c else "." for c in row) for row in shape)
            print(f"  {node.id[:40]}  w={w:.3f}  {rows_str}")


if __name__ == "__main__":
    run()
