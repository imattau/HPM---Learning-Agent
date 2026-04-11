"""
SP55: Experiment 38 — Grounded Perceptual Transfer

Tests whether the Observer can form transferable attractors from raw pixel
input (no pre-encoded semantic dimensions, no world model priors).

Setup:
1. Generate 8x8 binary images (64 dims) of 4 pattern types with Gaussian noise.
2. Train Observer cold-start on 120 unlabelled noisy images.
3. Measure attractor quality: do nodes cluster by pattern type?
4. Test retrieval transfer: given novel noisy test images, does the Decoder
   retrieve nodes of the correct pattern type?
5. Bonus: probe a novel composition (cross = H_STRIPE + V_STRIPE).

Metrics:
- Node Purity: fraction of activations from the dominant pattern type per node.
- Retrieval Accuracy: % of test images decoded to the correct pattern type.

Success: retrieval_accuracy > 0.75 AND distinct pattern types covered >= 3.
"""
from __future__ import annotations

import numpy as np
from collections import Counter, defaultdict

from hfn import Forest, Observer, calibrate_tau
from hfn.hfn import HFN
from hfn.decoder import Decoder

# --- Pattern types ---
PATTERNS = ["H_STRIPE", "V_STRIPE", "DIAGONAL", "CHECKERBOARD"]
DIM = 64  # 8x8 flattened


def make_pattern(pattern_type: str) -> np.ndarray:
    """Generate a clean 8x8 binary image for the given pattern type."""
    img = np.zeros((8, 8))
    if pattern_type == "H_STRIPE":
        img[2:6, :] = 1.0
    elif pattern_type == "V_STRIPE":
        img[:, 2:6] = 1.0
    elif pattern_type == "DIAGONAL":
        for i in range(8):
            img[i, i] = 1.0
    elif pattern_type == "CHECKERBOARD":
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 == 0:
                    img[r, c] = 1.0
    return img.flatten()


def generate_dataset(n_per_type: int, noise: float, seed: int) -> list[tuple[str, np.ndarray]]:
    """Generate noisy instances of all pattern types."""
    rng = np.random.RandomState(seed)
    data = []
    for ptype in PATTERNS:
        base = make_pattern(ptype)
        for _ in range(n_per_type):
            noisy = np.clip(base + rng.normal(0, noise, size=DIM), 0.0, 1.0)
            data.append((ptype, noisy))
    rng.shuffle(data)
    return data


def run_experiment():
    print("--- SP55: Experiment 38 — Grounded Perceptual Transfer ---\n")

    N_TRAIN = 30   # per pattern type
    N_TEST = 10    # per pattern type
    NOISE = 0.2

    train_data = generate_dataset(N_TRAIN, NOISE, seed=42)
    test_data = generate_dataset(N_TEST, NOISE, seed=99)

    # Cold-start Forest and Observer — no priors, no world model
    # tau=1.0 matches other experiments; calibrate_tau gives ~D which is too permissive
    # for structured pixel patterns with clear inter-type distance
    forest = Forest()
    tau = 1.0
    observer = Observer(forest=forest, tau=tau, residual_surprise_threshold=1.0, node_use_diag=True, budget=30)
    print(f"PHASE 1: Unsupervised Training on raw 8x8 pixels (tau={tau:.2f})...")

    # Track which pattern type activates each node
    node_activations: dict[str, Counter] = defaultdict(Counter)

    for ptype, vec in train_data:
        observer.observe(vec)
        # Record which node best explains this observation
        result = observer.expand(vec)
        if result.accuracy_scores:
            best_id = max(result.accuracy_scores, key=result.accuracy_scores.get)
            node_activations[best_id][ptype] += 1

    n_nodes = len(list(forest.active_nodes()))
    print(f"  Forest size after training: {n_nodes} nodes")

    # --- Phase 2: Attractor Quality ---
    print("\nPHASE 2: Attractor Quality...")

    # Only count nodes that were actually activated during training
    active_tracked = {nid: counts for nid, counts in node_activations.items()
                      if sum(counts.values()) > 0}

    if not active_tracked:
        print("  [WARN] No node activation data — using all active nodes")
        for ptype, vec in train_data:
            result = observer.expand(vec)
            if result.accuracy_scores:
                best_id = max(result.accuracy_scores, key=result.accuracy_scores.get)
                node_activations[best_id][ptype] += 1
        active_tracked = dict(node_activations)

    purities = []
    node_dominant_type: dict[str, str] = {}
    types_covered = set()

    for nid, counts in active_tracked.items():
        total = sum(counts.values())
        dominant = max(counts, key=counts.get)
        p = counts[dominant] / total
        purities.append(p)
        node_dominant_type[nid] = dominant
        types_covered.add(dominant)

    mean_purity = np.mean(purities) if purities else 0.0
    n_distinct = len(types_covered)

    print(f"  Tracked nodes: {len(active_tracked)}")
    print(f"  Mean node purity: {mean_purity:.3f}")
    print(f"  Distinct pattern types covered: {n_distinct} / {len(PATTERNS)}")
    for ptype in PATTERNS:
        count = sum(1 for d in node_dominant_type.values() if d == ptype)
        print(f"    {ptype}: {count} dominant node(s)")

    # --- Phase 3: Retrieval Transfer ---
    print("\nPHASE 3: Retrieval Transfer...")
    decoder = Decoder(target_forest=forest)

    correct = 0
    total = len(test_data)

    for ptype, vec in test_data:
        goal = HFN(mu=vec, sigma=np.ones(DIM) * 2.0, id="probe", use_diag=True)
        dec_res = decoder.decode(goal)
        if isinstance(dec_res, list) and dec_res:
            retrieved_id = dec_res[0].id
            # Find dominant type of retrieved node (check by mu proximity to training nodes)
            best_match = None
            best_dist = float("inf")
            for nid, counts in active_tracked.items():
                node = forest.get(nid)
                if node is not None:
                    dist = np.linalg.norm(dec_res[0].mu - node.mu)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = nid
            if best_match and node_dominant_type.get(best_match) == ptype:
                correct += 1

    retrieval_accuracy = correct / total
    print(f"  Retrieval Accuracy: {retrieval_accuracy*100:.1f}% ({correct}/{total})")

    # --- Phase 4: Novel Composition Probe ---
    print("\nPHASE 4: Novel Composition Probe (Cross = H_STRIPE + V_STRIPE)...")
    cross = np.clip(make_pattern("H_STRIPE") + make_pattern("V_STRIPE"), 0.0, 1.0)
    goal_cross = HFN(mu=cross, sigma=np.ones(DIM) * 2.0, id="cross_probe", use_diag=True)
    cross_res = decoder.decode(goal_cross)

    if isinstance(cross_res, list) and cross_res:
        retrieved_id = cross_res[0].id
        best_match = None
        best_dist = float("inf")
        for nid in active_tracked:
            node = forest.get(nid)
            if node is not None:
                dist = np.linalg.norm(cross_res[0].mu - node.mu)
                if dist < best_dist:
                    best_dist = dist
                    best_match = nid
        if best_match:
            cross_type = node_dominant_type.get(best_match, "UNKNOWN")
            print(f"  Cross pattern decoded to: {cross_type} node")
        else:
            print("  Cross pattern: no matching node found")
    else:
        print("  Cross pattern: decoder returned no result")

    # --- Results ---
    print("\n--- RESULTS ---")
    print(f"Node Purity (mean):        {mean_purity:.3f}")
    print(f"Distinct Types Covered:    {n_distinct} / {len(PATTERNS)}")
    print(f"Retrieval Accuracy:        {retrieval_accuracy*100:.1f}%")

    if retrieval_accuracy > 0.75 and n_distinct >= 3:
        print("\n[SUCCESS] Grounded Perceptual Transfer Achieved!")
        print("The Observer formed transferable attractors from raw pixels without")
        print("any pre-encoded semantic dimensions or world model priors.")
    else:
        print("\n[FAIL] Observer failed to form transferable attractors from raw pixels.")
        if n_distinct < 3:
            print(f"  Only {n_distinct} pattern type(s) represented — forest too sparse or collapsed.")
        if retrieval_accuracy <= 0.75:
            print(f"  Retrieval accuracy {retrieval_accuracy*100:.1f}% below 75% threshold.")


if __name__ == "__main__":
    run_experiment()
