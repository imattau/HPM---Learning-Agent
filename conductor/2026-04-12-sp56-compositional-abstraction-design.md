# SP56: Compositional Abstraction and Meta-Relational Transfer (Design)

## 1. Objective
To empirically validate the **Hierarchical Pattern Stack** of the HPM framework by demonstrating **Compositional Abstraction**. The agent will discover relations (Level 2), abstract the *geometric trajectories* of those relations (Level 3), and use this Level 3 meta-pattern to predict sequence continuations in a strictly novel, unseen domain (Zero-Shot Cross-Domain Transfer).

This experiment discards the anthropomorphic "Manager/Worker" concept, framing the stack purely as a **nested geometry of constraints** within a Multi-Polygraph HFN.

---

## 2. The Factorized Manifold (The Representation)
To enable multi-level abstraction, the HFN's D-dimensional space must be factorized to allow relations to be treated as observable data.

We define a 90-Dimensional Manifold:
*   **`[0:30]` Base Content (Level 1)**: The raw state (e.g., encoded numbers, boolean vectors, spatial coordinates).
*   **`[30:60]` Relational Delta (Level 2)**: The transformation vector between $State_t$ and $State_{t+1}$. This represents the "Rule" (e.g., `+1`, `Rotate`, `Invert`).
*   **`[60:90]` Meta-Relational Delta (Level 3)**: The transformation vector between $Relation_t$ and $Relation_{t+1}$. This represents the "Pattern of the Rule" (e.g., `Constant`, `Alternating`, `Accelerating`).

### The Stack Mechanism:
1.  When observing a sequence, the `EmpiricalOracle` populates L1 and L2.
2.  The `Observer` tracks the trajectory of the L2 activations over time.
3.  If the L2 trajectory is highly variant (e.g., `+1`, then `-1`, then `+1`), it triggers a high **Complexity Penalty** in the HPM Utility equation.
4.  To maximize utility, the system absorbs this trajectory into a single **L3 Meta-Node** (e.g., an `Alternator` polygraph). The L3 node sits "on top" of the L2 nodes, constraining their activation sequence.

---

## 3. The Experimental Curriculum

### Phase 1: Pre-training the Substrate (L2 Formation)
The agent is exposed to parallel streams of simple sequence pairs to stabilize basic L2 relations.
*   **Domain A (Numeric)**: `1 -> 2`, `5 -> 6` (Discovers L2: `Add_1`)
*   **Domain B (Spatial 1D)**: `[0, 1, 0] -> [0, 0, 1]` (Discovers L2: `Shift_Right`)

### Phase 2: Meta-Pattern Discovery (L3 Formation)
The agent is exposed to prolonged, complex sequences in the pre-trained domains. It must compress the sequence of L2 relations into L3 meta-patterns to minimize description length.
*   **Pattern X (Constant)**: 
    *   Math: `[1, 2, 3, 4, 5]` (L2 sequence: `Add_1, Add_1, Add_1...`)
    *   L3 Discovered: **`Identity_Meta`** (The rule does not change).
*   **Pattern Y (Alternating)**:
    *   Math: `[0, 1, 0, 1, 0]` (L2 sequence: `Add_1, Sub_1, Add_1, Sub_1...`)
    *   Spatial: `[Left, Right, Left, Right]`
    *   L3 Discovered: **`Oscillator_Meta`** (The rule flips between two states).
*   **Pattern Z (Accelerating)**:
    *   Math: `[1, 2, 4, 7, 11]` (L2 sequence: `Add_1, Add_2, Add_3...`)
    *   L3 Discovered: **`Accumulator_Meta`** (The rule itself grows by a constant L2 delta).

### Phase 3: Zero-Shot Transfer (The "Leap")
The agent is introduced to **Domain C (Boolean Logic)**, which it has *never* seen.
*   **Input Sequence**: `[True, False, True, False, ...]`
*   **Step 1**: The agent observes `True -> False` and `False -> True`. It quickly forms new L2 relations: `Negate_T` and `Negate_F`.
*   **Step 2 (The Constraint)**: The sequence of L2 rules (`Negate_T`, `Negate_F`) perfectly matches the geometric signature of the **L3 `Oscillator_Meta`** discovered in Phase 2.
*   **Step 3 (Prediction)**: The L3 node is activated. It acts as a top-down mathematical constraint (via Gaussian product) on the L2 manifold. The agent predicts the next 10 items in the boolean sequence with zero additional learning.

---

## 4. Agent Architecture & Optimizations (Leveraging Past Learnings)

1.  **Multi-CPU Asynchronous Processing**: 
    *   Leveraging the `AsyncHFNController` from recent refactors, Phase 1 and 2 will feed sequences through parallel worker threads. 
    *   The `TieredForest` will serve as the thread-safe shared memory where L2 and L3 nodes are probabilistically merged and updated via replicator dynamics.
2.  **Dense Embedding Oracle**:
    *   Building on SP55, we will use a fast hash-based dense projection (e.g., `zlib.adler32` or a lightweight `numpy` projection) to map discrete states into the continuous `[0:30]` L1 manifold, avoiding symbolic lookup tables.
3.  **Utility-Driven Compression**:
    *   Instead of hardcoding when to form an L3 node, we rely strictly on the `Observer._check_compression_candidates()` logic. The L3 `Oscillator` node will only form if the penalty of storing alternating L2 rules exceeds the cost of forming a new L3 meta-node.

---

## 5. Success Metrics

| Metric | Target | Description |
| :--- | :--- | :--- |
| **L3 Emergence** | $\ge 3$ distinct L3 nodes | The forest autonomously generates identifiable L3 nodes for Constant, Alternating, and Accelerating patterns. |
| **Compression Ratio** | $> 2.0x$ | The description length of Phase 2 sequences drops by at least 50% after L3 nodes form, proving utility-driven abstraction. |
| **Zero-Shot Transfer** | $100\%$ Accuracy | The agent correctly predicts $t_5 \dots t_{10}$ in the unseen Boolean domain using the L3 `Oscillator_Meta` node, without updating its L2 weights after $t_4$. |
| **Manifold Separation** | Distinct Clusters | PCA/t-SNE of the forest shows L1, L2, and L3 nodes occupying distinct dimensional frequencies within the shared 90D space. |
