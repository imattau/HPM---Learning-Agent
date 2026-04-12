# SP56 Refactor Plan: True Compositional Abstraction

## 1. Objective
Refactor the SP56 Compositional Abstraction experiment to address critical flaws in empirical validation, ensuring it demonstrates true zero-shot reasoning, top-down L3 constraint, and avoids oracle leakage.

## 2. Issues Addressed
1.  **"Zero-Shot" Claim Overstated**: The original experiment included Boolean domains in Phase 1, meaning the domain was not truly unseen in Phase 3.
2.  **No Causal Use of L3 (No Prediction)**: L3 was only used for retrieval scoring. It did not actively constrain L2 to generate the next state in a prediction loop.
3.  **Weak Success Metric**: The success condition merely checked if an L3 node was retrieved, rather than measuring prediction accuracy.
4.  **No Baseline Comparisons**: There were no ablations (e.g., L2-only) to prove that L3 was necessary for generalization.
5.  **Oracle Leakage Risk**: The previous hashing oracle destroyed continuous geometry, forcing the system to memorize pseudo-random trajectories rather than abstracting them.

## 3. Implementation Steps

### Step 1: Fix Oracle Geometry (Remove Leakage)
- Rewrite `StatefulOracleSP56.encode_l1` to map different data types (int, float, bool, spatial coordinates) to dedicated continuous axes in the 30D L1 manifold. 
- This replaces the `zlib` hashing, restoring the continuous additive physics required for genuine L2 and L3 delta formation without embedding handcrafted meta-relations.

### Step 2: Ensure True Unseen Domain
- Remove `([True, False], "negate")` from the Phase 1 curriculum in `experiment_compositional_abstraction.py`.
- Introduce a novel domain in Phase 3 (e.g., a Spatial 2D Accumulator sequence) that has never been exposed to the agent during Phase 1 or 2.

### Step 3: Implement Prediction Loop (L3 -> L2 Influence)
- Implement a `predict_next(current_vec, l3_constraint_vec)` function in the experiment script.
- The function will enforce top-down constraint mathematically:
  - $L2_{predicted} = L2_{current} + L3_{constraint}$
  - $L1_{predicted} = L1_{current} + L2_{predicted}$

### Step 4: Add Ablations and Strict Metrics
- Replace the passive recognition test in Phase 3 with an active autoregressive prediction task (e.g., predicting steps $t=4 \dots 9$ after priming on $t=0 \dots 3$).
- Implement three test conditions:
  1.  **L2-Only Baseline**: Assumes no meta-relational change ($L3 = 0$).
  2.  **Random L3 Baseline**: Uses random noise for the L3 constraint.
  3.  **Full HPM**: Uses the `[60:90]` slice of the retrieved L3 Meta-Node to actively constrain the prediction.
- Success is defined solely by high prediction accuracy (low Mean L1 Error) in the Full HPM condition compared to the baselines.

## 4. Verification
After the refactor, running the experiment must output clear ablation comparisons showing that the L3 top-down constraint prevents prediction divergence in the unseen sequence.
