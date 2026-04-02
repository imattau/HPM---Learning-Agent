# Thinking ARC Solver Experiment (SP28)

This experiment upgrades the Sovereign ARC Solver with a **Hypothesis Testing Loop** and expanded **30x30 Grid Support**. It transitions the system from "Pattern Matching" to "Iterative Reasoning."

## Core Mechanism: The "Thinking" Loop

Instead of producing a single guess, the system performs internal simulation:

1.  **Top-K Induction**: The perceptual specialists return their top 3 candidates for each training example.
2.  **Hypothesis Permutation**: The Governor generates a list of candidate rules based on the intersection of these winners.
3.  **Internal Simulation**: For each rule, the **Spatial Decoder** attempts to transform a training input grid.
4.  **Verification**: The Governor compares the simulation result to the actual training output.
    *   **Match**: The rule is validated and applied to the test case.
    *   **Mismatch**: The rule is falsified and recorded as "Negative Knowledge."

## 30x30 Sovereign Manifold

To handle the full range of ARC-AGI-2 tasks, the latent space has been expanded to **950 Dimensions**:
- **Spatial Slice (900D)**: 30x30 pixel transformation delta.
- **Symbolic Slice (30D)**: Numerical invariants.
- **Structural Slice (20D)**: Topological features.

## Key Insights: Negative Knowledge

A major focus of SP28 is **Negative Anchoring**. When a hypothesis fails validation, it isn't simply discarded. It is registered as an HFN node in a `Failure_Manifold`. 
- This allows the system to measure its **Geometric Distance to Solution**.
- Future iterations can use these negative anchors to prune the search space, avoiding regions of the manifold that have already been proven incorrect.

## Key Findings (20 Tasks)

- **Falsification in Action**: The "Thinking" loop successfully identified and rejected dozens of incorrect hypotheses that would have otherwise resulted in "hallucinated" answers.
- **Hypothesis Validation**: Several tasks reached the "Validated" state, where a rule perfectly explained the training examples, even if it later failed to generalize to the test case.
- **Sovereign Scaling**: Verified that the multi-process architecture and the Agnostic Decoder handle 900D vectors efficiently.

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_thinking_arc_solver.py
```
