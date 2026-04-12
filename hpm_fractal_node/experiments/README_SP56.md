# Experiment 46 (Refactored): Compositional Abstraction and Meta-Relational Transfer (SP56)

## 1. Objective
This experiment provides rigorous empirical validation of the **Hierarchical Pattern Stack** in the HPM framework. It demonstrates that second-order patterns (Level 3 Meta-Relations) are not just descriptive clusters, but **causal constraints** necessary for accurate prediction in non-stationary sequences across novel domains.

---

## 2. Refactored Methodology: The 90D Manifold
To ensure geometric consistency without handcrafted "oracle leakage," the manifold now uses a dedicated axis-mapping embedding:
- **Level 1 (Content)**: Maps numeric, boolean, and spatial dimensions to continuous axes.
- **Level 2 (Relation)**: $L2 = L1_t - L1_{t-1}$ (First-order derivative).
- **Level 3 (Meta-Relation)**: $L3 = L2_t - L2_{t-1}$ (Second-order derivative).

This additive manifold allows the system to discover structural trajectories (e.g., constant growth) purely through relative geometry.

---

## 3. Experimental Curriculum

### Phase 1: Relation Stabilization (L2)
Trains basic operators (e.g., `Add_1`) in simple Numeric and Spatial 1D domains. **Boolean logic is strictly excluded** to ensure zero-shot integrity.

### Phase 2: Meta-Pattern Discovery (L3)
Trains the agent on prolonged sequences with non-stationary rules (Accelerating sequences). The system abstracts the constant acceleration as a Level 3 "Accumulator" meta-pattern.

### Phase 3: Zero-Shot Transfer & Active Prediction (The Test)
The agent is presented with a **Spatial 2D Accumulator sequence** (a truly unseen domain using multiple axes).
1.  **Priming**: The agent observes $t=0 \dots 3$. It dynamically forms new L1, L2, and L3 nodes for this unseen domain.
2.  **Constraint**: The agent retrieves the newly formed L3 meta-node.
3.  **Autoregressive Prediction**: The agent predicts steps $t=4 \dots 9$ by feeding its own predictions back into the state loop, using the L3 node as a top-down constraint ($L2_{pred} = L2_{curr} + L3_{node}$).

---

## 4. Results: Strict Empirical Validation

| Test Condition | Mean L1 Prediction Error (t=4..9) | Result |
| :--- | :--- | :--- |
| **L2-Only Baseline** | **1.3199** | **FAIL (Diverged)** |
| **Random L3 Baseline** | **6.2585** | **FAIL (Diverged)** |
| **Full HPM (L3 Constraint)** | **0.0000** | **SUCCESS (Perfect Tracking)** |

### Analysis:
- **Causal Utility**: The L2-only baseline diverged because it assumed the transition rule was constant ($L3=0$). Only the L3-constrained HPM was able to "understand" how the rule was changing and maintain perfect accuracy.
- **Dynamic Abstraction**: The system successfully partitioned the latent space into 52 nodes, with `leaf_51` being the critical meta-pattern that enabled the zero-shot transfer.
- **Zero Oracle Leakage**: Success was achieved purely through the additive physics of the manifold, not through hashing or symbolic lookup.

---

## 5. Conclusion
SP56 (Refactored) confirms that the HPM Pattern Stack enables **Compositional Abstraction**. The agent does not just "recognize" a pattern; it uses the hierarchical geometry of that pattern to **reason about the future** of a system it has never seen before. This moves HPM from a theory of classification to a theory of **causal structural foresight**.
