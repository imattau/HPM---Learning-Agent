# Sovereign Cluster Experiment

This experiment demonstrates the **Sovereign AI** architecture: a distributed ecology of specialized HFN observers running in parallel. It tests the system's ability to model disparate domains (Math, Text, Spatial) simultaneously while managing novelty through an "Explorer" process.

## Architecture: The "Foundational Five" Cluster

The experiment implements a 5-process cluster based on the theoretical "Systemic Octet" taxonomy. Each process is an isolated `multiprocessing.Process` with its own `TieredForest` and `Observer`.

1.  **Math Specialist (Degree 1.0, Fixed)**:
    *   **Priors**: Full 306-node Math World Model.
    *   **Behavior**: Rigidly protects priors; rejects any signal that doesn't fit mathematical axioms.
2.  **Text Specialist (Degree 0.5, Learning)**:
    *   **Priors**: 195-node NLP World Model.
    *   **Behavior**: Adaptive; protects some semantic roots but allows drift and new node creation for linguistic shifts.
3.  **Spatial Specialist (Degree 1.0, Fixed)**:
    *   **Priors**: Synthetic 2D spatial primitives.
    *   **Behavior**: Rigidly protects geometric grounding.
4.  **Explorer (Degree 0.0, Wild)**:
    *   **Priors**: None.
    *   **Behavior**: High-plasticity ($\tau = 2.5$). Catches "Residual Surprise" rejected by specialists. Acts as a "nursery" for new domain stabilization.
5.  **Governor (Systemic/Controller)**:
    *   **Role**: Orchestrates batch routing, tracks "Stereo Vision" events, and monitors global topography.

## Theoretical Core: The Sovereignty Spectrum

A key concept tested is the **Degree of Specialization**:
*   **Degree 1.0**: Absolute prior protection. Functions as the system's "Ground Truth" or "Immune System."
*   **Degree 0.5**: Mixed protection. Functions as "Adaptive Modeling" for fluid environments.
*   **Degree 0.0**: Zero protection. Functions as "Pure Discovery."

## Experiment Stages

1.  **Domain Grounding**: Independent streams of Math, Text, and Spatial data. Verifies that specialists can effectively "tile" the common latent space.
2.  **The Novelty Shock**: Exposure to high-variance, cross-domain vectors that specialists cannot explain.
3.  **Cross-Domain Synthesis**: A return to mixed streams to check for interference and Explorer stabilization.

## Key Insights

- **Stereo Vision as Analogy**: The system identifies when multiple specialists "claim" the same input. This overlap is the mathematical foundation for cross-domain analogy discovery in HPM AI.
- **Structural Immunity**: Rigid specialists (Math/Spatial) are immune to the "Novelty Shock," correctly rejecting noise rather than corrupting their internal models.
- **Autonomous Expansion**: The Explorer successfully created 100+ new nodes during the shock stage, proving the system can autonomously expand its modeling scope without human intervention.
- **Global Topography**: The final system manages ~900 nodes across 4 processes, demonstrating that a "Global Brain" can be achieved through distributed, lean worker processes rather than a monolithic model.

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_sovereign_cluster.py
```
