"""
ARC-AGI-2 Observer experiment on 10x10 grids — full world model, colour encoding.

Uses value encoding (cell_value / 9.0) and build_world_model(10, 10) so all
7 layers of priors (perception, primitives, relationships, structural, colour,
semantic, encoder) are active.

Tau calibration for D=100, sigma=I:
  baseline = D/2 * log(2π) ≈ 91.9
  tau = baseline + 5.0  → tolerates ~10 cells difference (~10% variation)
  residual_threshold = baseline + 12.5  → new node for >25 cells novel

Run: python3 -m hpm_fractal_node.experiment_arc_10x10
"""

import json
import glob
import numpy as np
from collections import defaultdict

from hpm_fractal_node.observer import Observer
from hpm_fractal_node.arc_world_model import build_world_model


D = 100  # 10x10


def load_10x10_inputs(data_dir: str = "data/ARC-AGI-2/data/training") -> list[tuple[str, np.ndarray]]:
    records = []
    for f in sorted(glob.glob(f"{data_dir}/*.json")):
        d = json.load(open(f))
        puzzle_id = f.split("/")[-1].replace(".json", "")
        for ex in d["train"]:
            grid = ex["input"]
            if len(grid) != 10 or len(grid[0]) != 10:
                continue
            vec = np.array([[cell / 9.0 for cell in row]
                            for row in grid]).flatten()
            records.append((puzzle_id, vec))
    return records


def vec_to_grid(vec: np.ndarray) -> list[str]:
    grid = (vec > 0.5).astype(int).reshape(10, 10)
    return ["".join("X" if c else "." for c in row) for row in grid]


def run(n_passes: int = 2, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    records = load_10x10_inputs()
    n_puzzles = len(set(r[0] for r in records))
    print(f"Loaded {len(records)} observations from {n_puzzles} puzzles\n")

    baseline = D / 2 * np.log(2 * np.pi)   # ≈ 91.9
    tau = baseline + 5.0                     # ~10 cells tolerance
    residual_threshold = baseline + 12.5     # new node for >25 cells novel
    print(f"D={D}, baseline={baseline:.1f}, tau={tau:.1f}, residual_threshold={residual_threshold:.1f}\n")

    forest, prior_registry = build_world_model(rows=10, cols=10)
    prior_ids = set(prior_registry.keys())

    colour_nodes = [n for n in forest.active_nodes() if "colour" in n.id]
    print(f"World model: {len(forest)} nodes ({len(colour_nodes)} colour priors)")
    print()

    obs = Observer(
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
        protected_ids=prior_ids,
    )

    for pass_num in range(n_passes):
        indices = rng.permutation(len(records))
        for i in indices:
            _, vec = records[i]
            obs.observe(vec)
        print(f"Pass {pass_num + 1}: {len(forest)} nodes, {len(obs.absorbed_ids)} absorbed")

    # --- Report ---
    active = forest.active_nodes()
    nodes = sorted(active, key=lambda n: -obs.get_weight(n.id))

    print(f"\n=== Forest after {n_passes} passes ===")
    print(f"Active nodes:   {len(active)}")
    print(f"Absorbed nodes: {len(obs.absorbed_ids)}\n")

    # Density distribution of discovered nodes
    print("=== Discovered node densities ===")
    density_counts = defaultdict(int)
    for node in active:
        filled = int((node.mu > 0.5).sum())
        bucket = f"{(filled // 10) * 10}-{(filled // 10) * 10 + 9}"
        density_counts[bucket] += 1
    for bucket in sorted(density_counts):
        print(f"  {bucket:6s} filled cells: {density_counts[bucket]} nodes")

    # Top nodes by weight
    print(f"\n=== Top 10 nodes by weight ===")
    for node in nodes[:10]:
        w = obs.get_weight(node.id)
        filled = int((node.mu > 0.5).sum())
        rows = vec_to_grid(node.mu)
        print(f"  {node.id[:36]}  w={w:.3f}  filled={filled}/100")
        for r in rows[:3]:
            print(f"    {r}")
        print(f"    ...")

    # Row/col density profiles for top node
    if nodes:
        top = nodes[0]
        print(f"\n=== Row/col profile of top node ({top.id[:30]}) ===")
        grid = (top.mu > 0.5).astype(int).reshape(10, 10)
        print("  Row sums:", list(grid.sum(axis=1)))
        print("  Col sums:", list(grid.sum(axis=0)))

    # Coverage: how much do priors explain vs new nodes vs unexplained
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
    print(f"\n=== Explanation coverage ({total} observations) ===")
    print(f"  By surviving priors:  {explained_by_prior} ({100*explained_by_prior/total:.0f}%)")
    print(f"  By new nodes:         {explained_by_new} ({100*explained_by_new/total:.0f}%)")
    print(f"  Unexplained:          {unexplained} ({100*unexplained/total:.0f}%)")

    if prior_hits:
        print(f"\n  Most-used priors:")
        for pid, count in sorted(prior_hits.items(), key=lambda x: -x[1])[:8]:
            print(f"    {pid:<36}  explains {count} observations")

    absorbed_priors = [nid for nid in obs.absorbed_ids if nid in prior_ids]
    print(f"\n=== Absorbed priors: {len(absorbed_priors)} ===")


if __name__ == "__main__":
    run()
