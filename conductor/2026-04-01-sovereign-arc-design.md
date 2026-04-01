# Sovereign ARC Cluster — Design Specification

## 1. Overview and Rationale

The **Sovereign ARC Cluster** applies the multi-process "Sovereign AI" architecture to the Abstraction and Reasoning Corpus (ARC). Unlike previous ARC experiments that used a monolithic observer (e.g., `experiment_arc_world_model.py`), this system decomposes ARC reasoning into specialized, parallel domains.

The central hypothesis is that ARC puzzles are solved through **"Stereo Vision"**: simultaneous recognition of patterns across different cognitive domains (e.g., seeing a spatial rotation while simultaneously counting the objects).

## 2. The Multi-Process Architecture

The system consists of a central **Governor** and three specialized **Worker Processes**.

### 2.1 The Common Latent Space (The "Sovereign Manifold")
To enable multi-process observation, we must project ARC examples into a shared vector space where different specialists can read different slices.
Let $X$ be an ARC example (Input Grid $I$, Output Grid $O$). The governor computes a multi-modal feature vector $v(X) \in \mathbb{R}^D$ composed of three slices:

1.  **Spatial Slice ($\mathbb{R}^{S}$)**: Encodes the pixel-wise delta $(O - I)$, flattened.
2.  **Symbolic Slice ($\mathbb{R}^{M}$)**: Encodes numerical invariants (e.g., $\Delta$ active pixels, unique color count, bounding box area).
3.  **Structural Slice ($\mathbb{R}^{C}$)**: Encodes topological features (e.g., number of connected components, Euler characteristic, symmetry scores).

$D = S + M + C$. The Governor broadcasts the full vector $v(X)$ to all workers.

### 2.2 The Sovereign Workers
Each worker is an isolated `multiprocessing.Process` running an `Observer` over its own `TieredForest`.

#### Worker 1: The Spatial Specialist (Degree: 1.0, Rigid)
*   **Focus**: Pure geometry and translation.
*   **Observation Slice**: Only reads the Spatial Slice; ignores Symbol/Structure.
*   **Priors**: Pre-populated with spatial primitives (rotations, reflections, shifts) imported from `hpm_fractal_node.arc.arc_world_model`.
*   **Plasticity**: Very low. It expects the world to obey rigid spatial laws.

#### Worker 2: The Symbolic Specialist (Degree: 1.0, Rigid)
*   **Focus**: Counting, arithmetic, and logic.
*   **Observation Slice**: Only reads the Symbolic Slice.
*   **Priors**: Seeded with the Math World Model (`hpm_fractal_node.math.math_world_model`), focusing on small integers, primes, and parity.
*   **Plasticity**: Very low. 2+2 is always 4.

#### Worker 3: The Explorer (Degree: 0.0, Wild)
*   **Focus**: Novelty capture and un-modeled phenomena.
*   **Observation Slice**: Reads the entire space $D$.
*   **Priors**: None.
*   **Plasticity**: Very high ($\tau$ is large, creation radius is wide). Acts as the "nursery" for puzzle mechanics that the rigid specialists cannot explain.

## 3. The Governor Logic: "Stereo Vision Synthesis"

The Governor does not just route data; it looks for **Cross-Domain Intersections**.

For an ARC task $T$ containing multiple train examples $\{X_1, X_2, ..., X_n\}$:
1.  **Broadcast**: Send $v(X_i)$ to all workers.
2.  **Collect**: Receive explanations (the best-matching HFN node) from each worker.
3.  **Analyze**:
    *   If **Spatial** explains $X_i$ but **Symbolic** does not $\rightarrow$ Pure spatial task.
    *   If **Symbolic** explains $X_i$ but **Spatial** does not $\rightarrow$ Pure counting/logic task.
    *   If **BOTH** explain $X_i$ $\rightarrow$ **Stereo Vision Event**. The puzzle involves a mapping between geometry and math (e.g., "Rotate by $N$ degrees where $N$ is the number of blue pixels").
    *   If **NEITHER** explains $X_i$ $\rightarrow$ Route to **Explorer** to create a new joint pattern.

## 4. Evaluation Metrics

The experiment will track:
1.  **Domain Dominance**: What percentage of ARC tasks are explained by Spatial vs. Symbolic vs. Explorer?
2.  **Stereo Vision Frequency**: How often do tasks require simultaneous explanation from multiple specialists?
3.  **IPC Overhead**: Time spent serializing vectors vs. time spent in HFN `log_prob` calculations.

## 5. Implementation Roadmap

1.  **Data Loader**: Extend `arc_loader.py` to extract Symbolic and Structural features alongside the raw Spatial grids.
2.  **Worker Config**: Define the `WorkerConfig` for the ARC context.
3.  **Multi-Process Loop**: Implement the batch-routing logic (similar to `experiment_sovereign_cluster.py` but optimized for ARC tasks).
4.  **Reporting**: Output a puzzle-by-puzzle breakdown of which specialist "solved" the transformation.
