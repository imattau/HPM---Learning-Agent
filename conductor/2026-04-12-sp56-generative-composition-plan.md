# SP56 Final-Final Refactor Plan: True Generative Composition

## 1. Objective
To address the final feedback by moving from "Fractal Reuse via Shared Geometry" (retrieving a single matching vector) to **True Generative Composition**. The agent must construct a novel L3 constraint by mathematically combining multiple L2 primitives that it has learned, because no single stored vector matches the inferred L3 trajectory.

## 2. Identified Weaknesses Addressed
- **Retrieval-Dominant (Single Match)**: Previously, the agent found an exact match (`Add_1` = Accumulator +1). This is nearest-neighbor lookup, not composition.
- **No Compositional Synthesis**: The agent must now combine primitives (e.g., $L3 = L2_a + L2_b$) to approximate the novel target dynamic.

## 3. Implementation Steps

### Step 1: Adjust Curriculum for Composition
- **Phase 1 (L2 Primitives)**: Train the agent on `Add_1` and `Add_2`.
- **Phase 3 (Zero-Shot Test)**: Present an Accumulator that accelerates by **+3** each step ($0, 0, 3, 9, 18, 30 \dots$).
- The agent will infer an L3 trajectory of `+3`. Since no `+3` exists in any slice in Long-Term Memory, nearest-neighbor retrieval of a single node will yield a poor approximation (`+2` is the closest).

### Step 2: Implement Generative Composition Logic
- Update the retrieval loop in Phase 3 to allow **Linear Combination of Priors**.
- The agent will evaluate single nodes, but also pairs of nodes across slices: $L3_{constraint} = NodeA_{sliceX} + NodeB_{sliceY}$.
- It will discover that adding the L2 slice of `Add_1` and the L2 slice of `Add_2` perfectly constructs the required `+3` L3 trajectory.

### Step 3: Evaluate Stabilization
- Run the autoregressive prediction loop ($t=4 \dots 9$).
- Compare the composed $L3$ constraint against baselines, demonstrating that the dynamically synthesized meta-pattern drastically reduces prediction error compared to using the single best retrieved node.

## 4. Verification
If the system correctly identifies that `Add_1 + Add_2` forms the necessary L3 constraint and achieves near-zero error on the `+3` Accumulator, it demonstrates definitive **Generative Compositional Abstraction**.
