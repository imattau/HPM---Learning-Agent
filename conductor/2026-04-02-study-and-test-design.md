# SP29: Sovereign Study-and-Test — Design Specification

## 1. Overview and Rationale

The **Sovereign Study-and-Test** experiment (SP29) evaluates the HPM AI's capacity for **Meta-Transfer Learning**. In previous solver experiments, the system's world model was either static (priors only) or reset between each task to prevent interference.

SP29 implements a two-phase cognitive lifecycle:
1.  **Study Phase**: The system processes a set of "Study Tasks." It is allowed to learn, stabilize, and retain new HFN nodes representing emergent structural motifs.
2.  **Test Phase**: The system is given a set of unseen "Test Tasks." Crucially, it retains all nodes learned during the Study Phase.

The central hypothesis is that structural motifs learned in the Study Phase (e.g., a specific object-counting rule or a complex geometric symmetry) will act as **Ad-hoc Priors** that accelerate and improve the solving of related Test Tasks.

## 2. Multi-Process Architecture

The system uses the 4-process Sovereign Cluster (Spatial, Symbolic, Explorer, Decoder) with **Persistent State**.

### 2.1 Persistent World Models
Unlike SP28, the `TieredForest` cold directories are **not wiped** between tasks. Each worker maintains a growing library of:
*   **Axiomatic Priors**: The initial world model.
*   **Study Nodes**: Nodes discovered and stabilized during the Study Phase.
*   **Failure Manifold**: Negative knowledge (rejected hypotheses) from previous tasks.

### 2.2 The "Study" Governor
The Governor manages the curriculum:
1.  **Curriculum Load**: Loads two distinct task blocks (Study Set and Test Set).
2.  **Phase Transition**: Signals the specialists when the Study Phase is complete.
3.  **Measurement**: Tracks which "Study Nodes" are retrieved and used as "Explanation Winners" during the Test Phase.

## 3. The "Say and Point" Task (Extended)

We will use ARC-AGI-2 tasks grouped by concept (e.g., all "Object-Persistence" tasks).

*   **Study Set**: 10 tasks involving object movement and counting.
*   **Test Set**: 10 tasks involving more complex variations of the same underlying movement/counting rules.

## 4. Evaluation Metrics

1.  **Transfer Utility**: Percentage of Test Task observations explained by nodes born in the Study Phase.
2.  **Solve Rate Delta**: Comparison of solve rate on the Test Set *with* vs *without* the Study Phase.
3.  **Maturation Rate**: Do nodes from the Study Phase gain higher predictive weight when reused in the Test Phase?
4.  **Interference Tracking**: Do "Negative Anchors" from the Study Phase successfully prevent the system from testing known-false hypotheses in the Test Phase?

## 5. Implementation Roadmap

1.  **Persistence Layer**: Update `SovereignARCWorker` to support cumulative learning across multiple task calls.
2.  **Curriculum Generator**: Define the Study and Test task sets.
3.  **Node Genealogy**: Implement a tracking mechanism to tag nodes with their "Birth Task ID."
4.  **Experiment Script**: Implement `experiment_study_and_test.py`.
    *   Execute Study Phase (Tasks 1-10).
    *   Execute Test Phase (Tasks 11-20).
    *   Output the "Transfer Matrix" showing cross-task node reuse.
