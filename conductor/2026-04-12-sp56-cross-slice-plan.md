# SP56 Final Refactor Plan: Cross-Slice Compositional Abstraction

## 1. Objective
To elevate the SP56 experiment from "Hierarchical Stabilization" to **True Compositional Abstraction** by demonstrating Cross-Structure and Cross-Slice Generalization. The agent will invent a novel meta-pattern (Accumulator) zero-shot by reusing a first-order relation (Add_1) as a second-order constraint.

## 2. Identified Weaknesses Addressed
- **Single-Trajectory Bias**: The previous experiment trained on Accumulators and tested on Accumulators, meaning it merely recognized a known meta-pattern.
- **Retrieval vs. Composition**: The system was retrieving an existing L3 node rather than composing a new dynamic.

## 3. Implementation Steps

### Step 1: Remove Accumulators from Pre-training
- Modify Phase 2 to strictly train on **Constant** and **Oscillator** sequences. The agent will *never* see an Accumulator sequence during training, meaning it will have no L3 node for it.

### Step 2: Implement Cross-Slice Retrieval
- Since the HPM manifold is fractal, an Accumulator's L3 trajectory (e.g., constant +1 change to the rule) is geometrically identical to the L2 relation of "Add 1".
- Modify the Phase 3 retrieval logic to perform **Cross-Slice Retrieval**. The agent will take its noisy `inferred_l3` estimate and search the entire forest to find the closest 30D sub-vector across *all* slices (L1, L2, L3) of all nodes.

### Step 3: Zero-Shot Cross-Structure Transfer
- Test on the Spatial 2D Accumulator.
- The agent should discover that its noisy L3 trajectory best matches the `[30:60]` (L2) slice of the `num_add_1` node from Phase 1.
- It will then use this retrieved L2 relation as its top-down L3 constraint, dynamically composing the "Accumulator" meta-pattern on the fly.

## 4. Verification
If the Full HPM condition achieves near-zero error using a cross-slice retrieved constraint, it provides definitive proof of HPM's core thesis: **Fractal Compositional Abstraction**.
