"""
Math arithmetic experiment: algebraic rule discovery.

Runs the Observer over arithmetic observations (left, op, right, result)
encoded as pure one-hot vectors in R^109. No semantic flags are given to
the agent — all structure must emerge from geometric compression of the
observation stream.

Evaluation measures whether learned nodes align with known mathematical
categories (carry, divisibility, identity laws, prime results, etc.).

Usage:
    PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_math.py
"""
from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn import Observer, calibrate_tau
from hfn.fractal import hausdorff_distance
from hfn.tiered_forest import TieredForest
from hpm_fractal_node.math.math_loader import (
    generate_observations, get_category, CATEGORY_NAMES, D,
    all_observations,
)
from hpm_fractal_node.math.math_world_model import build_math_world_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SAMPLES = 5000
N_PASSES = 4
SEED = 42

TAU_SIGMA = 1.0
TAU_MARGIN = 5.0

N_CATEGORIES = len(CATEGORY_NAMES)
RANDOM_BASELINE = 1.0 / N_CATEGORIES


def purity(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(counts.values()) / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n_unique = len(all_observations())
    print(f"Loading {N_SAMPLES} math observations (D={D}) ...")
    print(f"  {n_unique} unique valid (left, op, right, result) tuples in pool")
    data = generate_observations(n=N_SAMPLES, seed=SEED)
    print(f"  {len(data)} sampled observations (with replacement)")
    print(f"  {N_CATEGORIES} evaluation categories: {CATEGORY_NAMES[:5]}...")

    print("\nBuilding math world model ...")
    _cold_dir = Path(__file__).parents[2] / "data" / "hfn_math_cold"
    _cold_dir.mkdir(parents=True, exist_ok=True)
    forest, prior_ids = build_math_world_model(
        forest_cls=TieredForest,
        cold_dir=_cold_dir,
        max_hot=600,
    )
    forest.set_protected(prior_ids)
    print(f"  {len(prior_ids)} priors registered and protected")

    tau = calibrate_tau(D, sigma_scale=TAU_SIGMA, margin=TAU_MARGIN)
    print(f"  tau = {tau:.2f}")

    obs = Observer(
        forest,
        tau=tau,
        protected_ids=prior_ids,
        recombination_strategy="nearest_prior",
        hausdorff_absorption_threshold=0.35,
        hausdorff_absorption_weight_floor=0.4,
        absorption_miss_threshold=20,
        persistence_guided_absorption=True,
        lacunarity_guided_creation=True,
        lacunarity_creation_radius=0.08,
        multifractal_guided_absorption=False,
        gap_query_threshold=None,   # no LLM — pure geometric learning
        max_expand_depth=2,
    )

    # ------------------------------------------------------------------
    # Multi-pass observation
    # ------------------------------------------------------------------
    node_explanations: dict[str, list[tuple]] = defaultdict(list)

    print(f"\nRunning {N_PASSES} passes over {N_SAMPLES} observations ...")
    for p in range(N_PASSES):
        n_explained = 0
        n_unexplained = 0
        rng = np.random.default_rng(SEED + p)
        order = rng.permutation(len(data))

        for i in order:
            vec, (left, op, right, result) = data[i]
            x = vec.astype(np.float64)
            obs_result = obs.observe(x)
            forest._on_observe()
            if (n_explained + n_unexplained) % 1000 == 0:
                print(f"    {n_explained + n_unexplained}/{N_SAMPLES} ...", flush=True)

            category = get_category(left, op, right, result)
            if obs_result.explanation_tree:
                best_id = max(
                    obs_result.accuracy_scores,
                    key=lambda k: obs_result.accuracy_scores[k],
                )
                node_explanations[best_id].append((left, op, right, result, category))
                n_explained += 1
            else:
                n_unexplained += 1

        n_total = len(data)
        print(f"  Pass {p+1}: explained {n_explained}/{n_total} "
              f"({100*n_explained/n_total:.1f}%), "
              f"unexplained {n_unexplained} "
              f"({100*n_unexplained/n_total:.1f}%)")

    # ------------------------------------------------------------------
    # Coverage summary
    # ------------------------------------------------------------------
    n_obs_total = N_PASSES * N_SAMPLES
    prior_explained = sum(len(v) for k, v in node_explanations.items() if k in prior_ids)
    learned_nodes = [k for k in node_explanations if k not in prior_ids]
    learned_explained = sum(len(node_explanations[k]) for k in learned_nodes)
    total_attributed = prior_explained + learned_explained

    print(f"\n=== Coverage (over all passes, {n_obs_total} total obs) ===")
    print(f"  Prior nodes explained:   {prior_explained:5d} ({100*prior_explained/n_obs_total:.1f}%)")
    print(f"  Learned nodes explained: {learned_explained:5d} ({100*learned_explained/n_obs_total:.1f}%)")
    print(f"  Total attributed:        {total_attributed:5d} ({100*total_attributed/n_obs_total:.1f}%)")
    print(f"  Learned node count:      {len(learned_nodes)}")

    # ------------------------------------------------------------------
    # Top prior alignment
    # ------------------------------------------------------------------
    print(f"\n=== Top prior nodes by coverage ===")
    prior_rows = [
        (k, node_explanations[k])
        for k in prior_ids
        if len(node_explanations.get(k, [])) > 0
    ]
    prior_rows.sort(key=lambda r: len(r[1]), reverse=True)
    for node_id, labels in prior_rows[:20]:
        n = len(labels)
        cat_counts: dict[str, int] = defaultdict(int)
        for *_, cat in labels:
            cat_counts[cat] += 1
        pur = purity(cat_counts)
        dom = max(cat_counts, key=lambda k: cat_counts[k])
        print(f"  {node_id:<35s}  n={n:5d}  purity={pur:.2f}  dom={dom}")

    # ------------------------------------------------------------------
    # Learned node alignment
    # ------------------------------------------------------------------
    if not learned_nodes:
        print("\nNo learned nodes formed.")
        return

    print(f"\n=== Learned node category alignment ({len(learned_nodes)} nodes) ===")
    rows = []
    for node_id in learned_nodes:
        if forest.get(node_id) is None:
            continue
        labels = node_explanations[node_id]
        n = len(labels)
        cat_counts: dict[str, int] = defaultdict(int)
        op_counts: dict[str, int] = defaultdict(int)
        for left, op, right, result, cat in labels:
            cat_counts[cat] += 1
            op_counts[op] += 1
        cat_pur = purity(cat_counts)
        op_pur = purity(op_counts)
        dom_cat = max(cat_counts, key=lambda k: cat_counts[k]) if cat_counts else "none"
        dom_op = max(op_counts, key=lambda k: op_counts[k]) if op_counts else "none"

        node_mu = forest.get(node_id).mu
        best_prior, best_dist = None, float("inf")
        for pid in prior_ids:
            pn = forest.get(pid)
            if pn is None:
                continue
            d = float(np.linalg.norm(node_mu - pn.mu))
            if d < best_dist:
                best_dist, best_prior = d, pid

        rows.append({
            "id": node_id, "n": n,
            "cat_purity": cat_pur, "op_purity": op_pur,
            "dom_cat": dom_cat, "dom_op": dom_op,
            "nearest_prior": best_prior, "prior_dist": best_dist,
        })

    rows.sort(key=lambda r: r["n"], reverse=True)
    header = (f"  {'Node ID':<20s}  {'n':>4s}  {'cat_pur':>7s}  "
              f"{'op_pur':>6s}  {'dom_cat':<22s}  {'dom_op':<6s}  {'nearest_prior':<35s}  dist")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows[:30]:
        print(
            f"  {r['id']:<20s}  {r['n']:>4d}  {r['cat_purity']:>7.3f}  "
            f"{r['op_purity']:>6.3f}  {r['dom_cat']:<22s}  {r['dom_op']:<6s}  "
            f"{r['nearest_prior'] or 'none':<35s}  {r['prior_dist']:.3f}"
        )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    cat_purities = [r["cat_purity"] for r in rows if r["n"] >= 5]
    print(f"\n=== Category purity summary (nodes with n>=5) ===")
    if cat_purities:
        print(f"  Category purity:  mean={np.mean(cat_purities):.3f}  max={np.max(cat_purities):.3f}")
        print(f"  Random baseline:  {RANDOM_BASELINE:.3f}  (1/{N_CATEGORIES})")
        n_above_baseline = sum(1 for p in cat_purities if p > RANDOM_BASELINE * 2)
        print(f"  Nodes > 2x baseline: {n_above_baseline}/{len(cat_purities)}")
    else:
        print("  Insufficient learned nodes with n>=5.")

    # ------------------------------------------------------------------
    # Fractal diagnostics
    # ------------------------------------------------------------------
    prior_node_list = [forest.get(k) for k in prior_ids if forest.get(k) is not None]
    learned_node_list = [n for n in forest.active_nodes() if n.id not in prior_ids]
    if learned_node_list:
        hd = hausdorff_distance(learned_node_list, prior_node_list)
        print(f"\n=== Fractal diagnostics ===")
        print(f"  Active learned nodes:       {len(learned_node_list)}")
        print(f"  Hausdorff(learned, priors): {hd:.4f}")
    print(f"\n=== Absorbed nodes: {len(obs.absorbed_ids)} ===")

    # ------------------------------------------------------------------
    # Abstraction candidates
    # ------------------------------------------------------------------
    word_ids = prior_ids
    depth_map: dict[str, int] = {pid: 0 for pid in word_ids}
    queue: deque = deque(word_ids)
    visited = set(word_ids)
    while queue:
        nid = queue.popleft()
        node = forest.get(nid)
        if node is None:
            continue
        try:
            children = list(node.children()) if callable(node.children) else list(node.children)
        except Exception:
            children = []
        for child in children:
            if child.id not in visited:
                depth_map[child.id] = depth_map[nid] + 1
                visited.add(child.id)
                queue.append(child.id)

    active_non_prior = [n for n in forest.active_nodes() if n.id not in prior_ids]
    n_depth2 = sum(1 for n in active_non_prior if depth_map.get(n.id, 0) >= 2)
    n_cross_cat = sum(
        1 for n in active_non_prior
        if len({cat for *_, cat in node_explanations.get(n.id, [])}) >= 2
    )
    n_stable = sum(1 for n in active_non_prior if n.id in node_explanations)

    print(f"\n=== Abstraction candidates ({len(active_non_prior)} non-prior nodes) ===")
    print(f"  depth >= 2:           {n_depth2}")
    print(f"  cross-category (>=2): {n_cross_cat}")
    print(f"  stable:               {n_stable}")

    # ------------------------------------------------------------------
    # Rule discovery summary: which mathematical rules emerged?
    # ------------------------------------------------------------------
    print(f"\n=== Rule discovery summary ===")
    rule_hits: dict[str, int] = defaultdict(int)
    for node_id in learned_nodes:
        labels = node_explanations.get(node_id, [])
        for *_, cat in labels:
            rule_hits[cat] += 1

    if rule_hits:
        sorted_rules = sorted(rule_hits.items(), key=lambda x: x[1], reverse=True)
        print("  Category              learned_node_obs")
        for cat, cnt in sorted_rules[:15]:
            print(f"  {cat:<30s}  {cnt:5d}")
    else:
        print("  No learned node observations to summarise.")


if __name__ == "__main__":
    main()
