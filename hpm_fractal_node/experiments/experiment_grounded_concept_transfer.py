"""
SP56: Experiment 39 — Grounded Concept Transfer

Tests whether the system can learn what an operation *is* from visual examples
of it being applied, then predict outcomes for novel inputs — with no explicit
concept labels, no symbolic preprocessing, and no separation between perceptual
and planning layers.

Key insight: operations are encoded as [icon(64D), delta(1D)] where delta =
after - before. For operations with constant delta (ADD_1: +1, SUB_1: -1,
ADD_3: +3, IDENTITY: 0), the concept is fully captured in the icon-delta pair,
independent of the specific before_value. This allows clean attractor formation
and generalisation to novel values.

Phases:
1. Grounded learning: Observer sees icon+delta pairs, no concept names
2. Predictive transfer: Decoder predicts delta from noisy icon → after = before + delta
3. Icon invariance: shifted icon variants at test time
4. Novel value transfer: any before_value works since delta is operation-specific

Success: prediction_accuracy > 0.85 AND novel_value_accuracy > 0.70
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict

from hfn.forest import Forest
from hfn.observer import Observer
from hfn.decoder import Decoder
from hfn.hfn import HFN

# --- Constants ---
ICON_DIM = 64   # 8x8 flattened
TOTAL_DIM = ICON_DIM + 1  # [icon(64D), delta(1D)] = 65D

# Operations with constant deltas (after - before)
OPERATIONS = ["ADD_1", "SUB_1", "ADD_3", "IDENTITY"]
OP_DELTA = {"ADD_1": 1.0, "SUB_1": -1.0, "ADD_3": 3.0, "IDENTITY": 0.0}


# --- Operation icons (8x8 binary patterns) ---

def make_icon(op: str) -> np.ndarray:
    """Generate maximally-distinct 8x8 binary icons (orthogonal quadrant patterns)."""
    img = np.zeros((8, 8))
    if op == "ADD_1":
        # Horizontal stripe: rows 2-5 filled
        img[2:6, :] = 1.0
    elif op == "SUB_1":
        # Vertical stripe: cols 2-5 filled
        img[:, 2:6] = 1.0
    elif op == "ADD_3":
        # Diagonal: main diagonal pixels
        for i in range(8):
            img[i, i] = 1.0
    elif op == "IDENTITY":
        # Checkerboard: (row+col) % 2 == 0
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 == 0:
                    img[r, c] = 1.0
    return img.flatten()


# --- Data generation ---

def generate_training_data(n_per_op: int, noise: float,
                           seed: int) -> list[tuple[str, np.ndarray]]:
    """Training vectors: [icon(64D), delta(1D)] — no before/after values."""
    rng = np.random.RandomState(seed)
    data = []
    for op in OPERATIONS:
        base_icon = make_icon(op)
        delta = OP_DELTA[op]
        for _ in range(n_per_op):
            noisy_icon = np.clip(base_icon + rng.normal(0, noise, ICON_DIM), 0, 1)
            vec = np.zeros(TOTAL_DIM)
            vec[:ICON_DIM] = noisy_icon
            vec[ICON_DIM] = delta
            data.append((op, vec))
    rng.shuffle(data)
    return data


def generate_test_data(values: list[float], icon_noise: float,
                       icon_shift: int, seed: int) -> list[tuple[str, float, np.ndarray]]:
    """Test data: (op, before_value, noisy_icon) — after = before + OP_DELTA[op]."""
    rng = np.random.RandomState(seed)
    data = []
    for op in OPERATIONS:
        base_icon = make_icon(op)
        if icon_shift > 0:
            img = base_icon.reshape(8, 8)
            base_icon = np.roll(img, icon_shift, axis=1).flatten()
        for val in values:
            noisy_icon = np.clip(base_icon + rng.normal(0, icon_noise, ICON_DIM), 0, 1)
            data.append((op, val, noisy_icon))
    return data


# --- Experiment ---

def run_experiment():
    print("--- SP56: Experiment 39 — Grounded Concept Transfer ---\n")

    TRAIN_VALUES = list(range(1, 9))    # 1–8 (for test only)
    NOVEL_VALUES = [9.0, 10.0]
    N_PER_OP = 5                       # 5 examples per op = 20 total (avoids compression cascade)
    ICON_NOISE = 0.15
    TAU = 1.0

    # Phase 1: Grounded learning
    print("PHASE 1: Grounded Learning (no concept labels)...")
    train_data = generate_training_data(N_PER_OP, ICON_NOISE, seed=42)
    print(f"  Training examples: {len(train_data)}")

    forest = Forest()
    observer = Observer(
        forest=forest,
        tau=TAU,
        residual_surprise_threshold=1.0,
        node_use_diag=True,
        budget=30,
    )

    node_op_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for op_label, vec in train_data:
        observer.observe(vec)
        result = observer.expand(vec)
        if result.accuracy_scores:
            best_id = max(result.accuracy_scores, key=result.accuracy_scores.get)
            node_op_counts[best_id][op_label] += 1

    n_nodes = len(list(forest.active_nodes()))
    print(f"  Forest size: {n_nodes} nodes")

    node_dominant: dict[str, str] = {}
    ops_covered = set()
    purities = []
    for nid, counts in node_op_counts.items():
        total = sum(counts.values())
        if total == 0:
            continue
        dominant = max(counts, key=counts.get)
        purities.append(counts[dominant] / total)
        node_dominant[nid] = dominant
        ops_covered.add(dominant)

    mean_purity = np.mean(purities) if purities else 0.0
    print(f"  Node purity (mean): {mean_purity:.3f}")
    print(f"  Operations covered: {len(ops_covered)} / {len(OPERATIONS)}")

    # Phase 2: Predictive transfer (seen values, matched icons)
    print("\nPHASE 2: Predictive Transfer (seen values, matched icons)...")
    decoder = Decoder(target_forest=forest)

    def predict_after(before: float, icon: np.ndarray) -> float | None:
        """Query icon only; retrieve predicted delta; return before + delta."""
        query_mu = np.zeros(TOTAL_DIM)
        query_mu[:ICON_DIM] = icon
        query_sigma = np.ones(TOTAL_DIM) * 0.5
        query_sigma[ICON_DIM] = 100.0   # delta is unknown — high sigma
        goal = HFN(mu=query_mu, sigma=query_sigma, id="query", use_diag=True)
        res = decoder.decode(goal)
        if isinstance(res, list) and res:
            predicted_delta = float(res[0].mu[ICON_DIM])
            return before + predicted_delta
        return None

    test_seen = generate_test_data(TRAIN_VALUES, ICON_NOISE, icon_shift=0, seed=99)
    correct_seen = 0
    for op, val, icon in test_seen:
        pred = predict_after(val, icon)
        true = val + OP_DELTA[op]
        if pred is not None and abs(pred - true) < 0.5:
            correct_seen += 1

    acc_seen = correct_seen / len(test_seen)
    print(f"  Accuracy (seen values): {acc_seen*100:.1f}% ({correct_seen}/{len(test_seen)})")

    # Phase 3: Icon invariance (shifted icons)
    print("\nPHASE 3: Icon Invariance (shifted icon variants)...")
    test_shifted = generate_test_data(TRAIN_VALUES, ICON_NOISE, icon_shift=2, seed=99)
    correct_shifted = 0
    for op, val, icon in test_shifted:
        pred = predict_after(val, icon)
        true = val + OP_DELTA[op]
        if pred is not None and abs(pred - true) < 0.5:
            correct_shifted += 1

    acc_shifted = correct_shifted / len(test_shifted)
    print(f"  Accuracy (shifted icons, shift=2): {acc_shifted*100:.1f}% ({correct_shifted}/{len(test_shifted)})")

    # Phase 4: Novel value transfer
    print("\nPHASE 4: Novel Value Transfer (values 9-10, unseen)...")
    test_novel = generate_test_data(NOVEL_VALUES, ICON_NOISE, icon_shift=0, seed=99)
    correct_novel = 0
    for op, val, icon in test_novel:
        pred = predict_after(val, icon)
        true = val + OP_DELTA[op]
        if pred is not None and abs(pred - true) < 0.5:
            correct_novel += 1

    acc_novel = correct_novel / len(test_novel)
    print(f"  Accuracy (novel values 9-10): {acc_novel*100:.1f}% ({correct_novel}/{len(test_novel)})")

    # Summary
    print("\n--- RESULTS ---")
    print(f"Node Purity (mean):            {mean_purity:.3f}")
    print(f"Operations Covered:            {len(ops_covered)} / {len(OPERATIONS)}")
    print(f"Predictive Accuracy (seen):    {acc_seen*100:.1f}%")
    print(f"Icon Invariance (shifted):     {acc_shifted*100:.1f}%")
    print(f"Novel Value Transfer:          {acc_novel*100:.1f}%")

    if acc_seen > 0.85 and acc_novel > 0.70:
        print("\n[SUCCESS] Grounded Concept Transfer Achieved!")
        print("The system learned operation semantics from visual co-occurrence alone")
        print("and generalised to novel inputs without explicit concept labels.")
    else:
        print("\n[FAIL] Grounded concept transfer did not reach threshold.")
        if acc_seen <= 0.85:
            print(f"  Basic prediction failed ({acc_seen*100:.1f}% < 85%)")
        if acc_novel <= 0.70:
            print(f"  Novel value transfer failed ({acc_novel*100:.1f}% < 70%)")


if __name__ == "__main__":
    run_experiment()
