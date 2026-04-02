# SP28: The Iterative "Thinking" Solver — Design Specification

## 1. Overview and Rationale

The **Iterative "Thinking" Solver** (SP28) upgrades the Sovereign ARC Solver from a zero-shot predictor to a multi-shot reasoning engine. It implements an internal validation loop where the system tests its induced rules against the training examples before committing to a test output. If a mismatch is detected, the system "thinks" by iterating through candidate rules or triggering demand-driven learning.

Additionally, this experiment expands the **Sovereign Manifold** from 10x10 (100D) to **30x30 (900D)** to support the full range of ARC-AGI-2 tasks.

## 2. Expanded Manifold (30x30)

To handle larger grids, the common latent space is expanded:
*   **Spatial Slice ($S$): 900D** (30x30 grid delta).
*   **Symbolic Slice ($M$): 30D** (Numerical invariants).
*   **Structural Slice ($C$): 20D** (Topological features).
*   **Total $D$: 950D**.

All workers and the Agnostic Decoder will be updated to operate in this 950D manifold.

## 3. The "Thinking" Loop (Iterative Refinement)

The Governor implements a **Hypothesis Testing** loop for each ARC task:

### Phase 1: Candidate Induction
1.  Governor broadcasts training examples to the Perceptual Cluster.
2.  Instead of just the top winner, each specialist returns its **Top $K$ candidates**.
3.  The Governor generates a ranked list of **Rule Hypotheses** $\{H_1, H_2, ..., H_m\}$ based on the intersections and permutations of specialist winners.

### Phase 2: Internal Validation (The "Thinking" Phase)
For each hypothesis $H_i$:
1.  **Simulation**: The Governor asks the **Spatial Decoder** to express $H_i$ for a **Training Input**.
2.  **Verification**: The Governor compares the decoded output to the **Training Output**.
3.  **Signal**: 
    *   If **Match**: Hypothesis $H_i$ is validated. Proceed to Phase 3.
    *   If **Mismatch**: Hypothesis $H_i$ is falsified. **Negative Anchoring**: Register $H_i$ as a "Negative Concept" node. This allows the system to measure how "far" it is from the truth and avoid similar failures in subsequent iterations.
4.  **Exhaustion**: If all $H_i$ fail, trigger **Demand-Driven Learning** (SP24) to discover a novel rule node specifically for the residuals of the training set.

## 4. Negative Knowledge: The Failure Manifold
Failed hypotheses are not discarded but stored as HFN nodes with negative valence (low weight/high penalty). This provides the system with:
*   **Error Geometry**: Measuring the distance between a failed $\mu$ and the target $\mu$.
*   **Search Pruning**: The Observer can use these negative anchors to suppress "boring" or already-falsified regions of the latent space.

### Phase 3: Test Execution
1.  The validated Rule $H^*$ is used to construct the Goal for the Test Input.
2.  The Decoder resolves the Goal into the final Test Output grid.

## 4. Evaluation Metrics

1.  **Thinking Depth**: Average number of iterations (hypotheses tested) per successful solve.
2.  **Large Grid Accuracy**: Solve rate for tasks with grids > 10x10.
3.  **False Positive Rate**: How often a rule passes training validation but fails the test case.
4.  **Recovery Rate**: Percentage of tasks solved only after the first hypothesis was falsified.

## 5. Implementation Roadmap

1.  **Loader Update**: Modify `arc_sovereign_loader.py` to extract 900D spatial vectors (30x30).
2.  **Induction Update**: Update the Governor to collect and permute the top 3 winners from each specialist.
3.  **Validation Loop**: Implement the `for hypothesis in hypotheses:` loop in the experiment script.
4.  **Worker Scaling**: Ensure all workers handle the 950D vector efficiently.
