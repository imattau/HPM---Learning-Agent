# SP56 Final Refactor Plan: Removing Explicit L3 Access

## 1. Objective
To address the final feedback on the SP56 Compositional Abstraction experiment. The goal is to remove the last dependency on explicit, ground-truth L3 access during inference, proving that the system can actively infer a noisy meta-relational trajectory and use a retrieved, stabilized L3 constraint to improve autoregressive prediction.

## 2. Identified Weaknesses Addressed
1.  **Explicit L3 Access (Oracle Leakage)**: The previous retrieval step used the oracle's perfectly computed L3 at $t=3$. We must mask this and infer L3 purely from the noisy L1/L2 sequence history.
2.  **Linear Zero-Error**: The perfect `0.0000` error was an artifact of noiseless arithmetic. Introducing perceptual noise to L1 will ensure prediction errors accumulate realistically unless stabilized by a top-down L3 constraint.

## 3. Implementation Steps

### Step 1: Add Perceptual Noise to Phase 3 Priming
- During Phase 3 (Zero-Shot Transfer), add Gaussian noise (e.g., $\sigma=0.01$) to the Level 1 (Content) vectors of the priming sequence ($t=0 \dots 3$).
- The agent will recompute L2 and L3 online from this noisy L1 history, representing true bottom-up perception.

### Step 2: Infer Noisy L3 from History
- At $t=3$, the agent computes an `inferred_l3` vector based entirely on the noisy $L2_3 - L2_2$ transition.
- **Critical Change**: The query to the Retriever will *only* use this `inferred_l3` vector, not the clean ground truth. This forces the system to perform a retrieval based on an estimated, noisy trajectory.

### Step 3: Top-Down Stabilization
- The Retriever will match the noisy `inferred_l3` against the clean, stable L3 Meta-Nodes discovered during Phase 2 pre-training.
- The retrieved clean L3 node will be used as the top-down mathematical constraint for the autoregressive prediction loop ($t=4 \dots 9$).

### Step 4: Strict Baseline Comparisons
Implement three specific baselines:
1.  **L2-Only**: Assumes constant rule (diverges due to missing the accumulator pattern).
2.  **Noisy Bottom-Up**: Uses the `inferred_l3` without retrieval. Error will accumulate due to the noise in the estimate.
3.  **Full HPM**: Uses the stable retrieved L3 node. Error will be minimal, demonstrating that top-down retrieval successfully stabilizes noisy bottom-up perception.

## 4. Verification
The experiment will be deemed a complete success if the **Full HPM** condition achieves a significantly lower Mean L1 Prediction Error than both the **L2-Only** and **Noisy Bottom-Up** baselines.
