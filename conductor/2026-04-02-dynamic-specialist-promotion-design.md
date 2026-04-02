# SP25: Dynamic Specialist Promotion — Design Specification

## 1. Overview and Rationale

The **Dynamic Specialist Promotion** experiment (SP25) bridges Curriculum Learning, Multi-Process Architecture, and Demand-Driven Synthesis. In previous experiments, the multi-process architecture was hardcoded: we predefined a Math worker, a Spatial worker, etc. 

SP25 tests the HPM AI's ability to achieve **Emergent Sovereignty**. The system starts as a single monolithic Generalist. As it processes a mixed curriculum, it must autonomously identify dense, high-utility sub-graphs within its world model and "promote" them by spawning dedicated specialist processes (Worker nodes). This prevents catastrophic interference and optimizes computational resources.

## 2. The Core Mechanism: "Structural Clustering & Promotion"

The Governor monitors the Generalist's `TieredForest`. It looks for regions of the latent space that exhibit high structural stability.

### 2.1 The Promotion Criteria
A sub-graph (a collection of connected HFN nodes) is eligible for promotion to a new Specialist Process if it meets the following criteria:
1.  **Density (Volume)**: The cluster contains at least $N_{min}$ active nodes.
2.  **Persistence (Utility)**: The average predictive weight of the nodes in the cluster exceeds $W_{min}$.
3.  **Isolation (Modularity)**: The cluster is topologically distinct from other dense regions (minimal edge crossings out of the cluster).

### 2.2 The Promotion Event
When the Governor detects a qualifying cluster:
1.  **Extraction**: It extracts the nodes and their internal edges from the Generalist's forest.
2.  **Spawning**: It spawns a new `multiprocessing.Process` (a `SovereignWorker`) and initializes its `TieredForest` with the extracted nodes.
3.  **Deregistration**: It deletes the extracted nodes from the Generalist to free up capacity.
4.  **Routing Update**: The Governor updates its internal routing table. Future observations that fall within the bounding geometry ($\mu \pm \Sigma$) of the promoted cluster are routed to the new Specialist instead of the Generalist.

## 3. The Experiment: "The Signal in the Noise"

To test this without requiring the full NLP or ARC stack, we will use a synthetic 2D continuous space.

*   **The World**: A 2D coordinate space.
*   **The Data Stream**: A curriculum containing two types of observations:
    *   *The Signal (Domain A)*: A highly structured, tight cluster of points (e.g., points sampled from a Gaussian centered at `[5.0, 5.0]` with low variance).
    *   *The Noise (Domain B)*: Uniformly distributed random points across the entire `[-10, 10]` space.
*   **The System**: Starts with 1 Governor and 1 Generalist worker (`Explorer`, Degree 0.0, high plasticity).

### 3.1 The Expected Lifecycle
1.  **Phase 1 (Monolithic)**: The Explorer absorbs both the Signal and the Noise. Because the Signal is structured and repetitive, those nodes gain high weight and density. The Noise nodes remain sparse and low-weight.
2.  **Phase 2 (Detection)**: The Governor scans the Explorer's active nodes, runs a simple clustering algorithm (e.g., distance-based grouping of high-weight nodes), and identifies the "Signal Cluster."
3.  **Phase 3 (Promotion)**: The Governor spawns `Specialist_1`, seeding it with the Signal nodes, and removes them from the Explorer.
4.  **Phase 4 (Distributed)**: The data stream continues. 
    *   When a new Signal point arrives, the Governor routes it to `Specialist_1`.
    *   When a new Noise point arrives, the Governor routes it to the `Explorer`.

## 4. Evaluation Metrics

1.  **Autonomous Spawning**: Does the system successfully detect the cluster and spawn exactly one new specialist process during the run?
2.  **Topological Extraction**: Are the nodes cleanly moved from the Generalist to the Specialist without data loss or duplication?
3.  **Routing Accuracy**: After promotion, what percentage of subsequent "Signal" observations are correctly routed to the new Specialist?
4.  **Generalist Relief**: Does the removal of the dense cluster improve the Generalist's ability to handle the remaining noise (measured by a drop in total active nodes due to better garbage collection of sparse data)?

## 5. Implementation Roadmap

1.  **Clustering Algorithm**: Implement a lightweight clustering heuristic in the Governor to group nodes by $\mu$-distance and filter by weight.
2.  **Worker Lifecycle**: Update the Governor to support dynamic `Process.start()` during the main execution loop.
3.  **Routing Table**: Implement a mechanism in the Governor to check new observations against the bounding boxes of active Specialists before defaulting to the Generalist.
4.  **Experiment Script**: Implement `experiment_dynamic_promotion.py`.
