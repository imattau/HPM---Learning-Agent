# SP19: Sovereign Meta-Hierarchy — Design Specification

## 1. Overview and Rationale

The **Sovereign Meta-Hierarchy** experiment (SP19) advances the HPM AI from **Parallel Recognition** (SP18) to **Hierarchical Synthesis**. In SP18, the Governor identified "Stereo Vision" events when multiple Level 1 (L1) specialists explained the same observation. However, that insight was ephemeral.

SP19 introduces a **Level 2 (L2) Relational Specialist**. Its job is to "observe the observers" — to ingest the explanation states of the L1 specialists and stabilize recurring "Joint Identities" (cross-domain analogies) as permanent HFN nodes in a higher-order manifold.

This represents the birth of **Generative Reasoning**: formally learning that a geometric move is governed by a numerical property.

## 2. The Multi-Tier Architecture

The system consists of a **Governor**, two **L1 Specialists**, and one **L2 Bridge Specialist**. All run in independent processes.

### 2.1 Level 1: The Perceptual Specialists
*   **Worker 1: Spatial Specialist (L1)**
    *   **Input**: Spatial feature vector (e.g., pixel delta, rotation angle).
    *   **Output to L2**: The $\mu$ vector of its best-matching prior (the "Explanation Anchor").
*   **Worker 2: Symbolic Specialist (L1)**
    *   **Input**: Symbolic feature vector (e.g., object count, parity).
    *   **Output to L2**: The $\mu$ vector of its best-matching prior.

### 2.2 Level 2: The Relational Bridge
*   **Worker 3: L2 Synthesizer**
    *   **The L2 Manifold**: A new latent space $D_{L2} = D_{Spatial} + D_{Symbolic}$.
    *   **Input**: The concatenated $\mu$ vectors from the L1 winners: $[\mu_{Spatial} \,\|\, \mu_{Symbolic}]$.
    *   **Role**: To discover and stabilize "Joint Identity Nodes".
    *   **Plasticity**: High ($\tau$ is large). It starts with zero priors and learns pure correlations (e.g., "Whenever L1-Spatial says 'Rotate 90', L1-Symbolic says 'Count = 3'").

### 2.3 The Governor (Orchestrator)
1.  Routes raw data to L1 specialists.
2.  Waits for L1 explanations.
3.  Assembles the L2 message vector.
4.  Routes the message to the L2 Bridge.
5.  Tracks the emergence of stabilized L2 nodes.

## 3. The "Rosetta Grounding" Dataset

To test this, we need synthetic data where the spatial rule is strictly dependent on the symbolic state.

**Task: "Count-Governed Rotation"**
*   **Input**: A 3x3 ARC-style grid with $N$ active pixels ($N \in \{1, 2, 3, 4\}$).
*   **Rule**: The grid rotates by $N \times 90^\circ$.
*   **L1 Spatial** sees: "Rotation(90)", "Rotation(180)", etc.
*   **L1 Symbolic** sees: "Count(1)", "Count(2)", etc.
*   **L2 Goal**: Discover 4 distinct "Law Nodes" that bind Count(N) to Rotation(N*90).

## 4. Evaluation Metrics

The experiment will track:
1.  **L1 Recognition Accuracy**: Can the L1 specialists correctly identify the isolated transformations and counts?
2.  **L2 Convergence**: Does the L2 Bridge discover exactly 4 dominant nodes (one for each rule permutation)?
3.  **Cross-Domain Purity**: Do the L2 nodes maintain high purity (e.g., does the L2 node for "Count=3" only fire when "Rotation=270" is observed)?
4.  **Generative Potential**: (Optional) Can the L2 node predict the L1 Spatial $\mu$ given only the L1 Symbolic $\mu$?

## 5. Implementation Roadmap

1.  **Dataset Generator**: Implement `generate_rosetta_tasks()` to create the Count-Governed Rotation dataset.
2.  **L2 Message Protocol**: Update the IPC queue logic so L1 workers return their explanation `mu` vectors, not just a boolean success flag.
3.  **L2 Worker Config**: Define the `WorkerConfig` for the higher-order L2 Bridge, initializing it as a "Wild Specialist" (Degree 0.0) that rapidly absorbs recurring messages.
4.  **Experiment Loop**: Implement `experiment_sovereign_meta.py` with the two-tier routing Governor.
