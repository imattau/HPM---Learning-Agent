# Sovereign Meta-Hierarchy Experiment (SP19)

This experiment advances the Sovereign AI architecture from parallel recognition to **Hierarchical Relational Synthesis**. It tests the system's ability to formally "observe the observers" and stabilize cross-domain analogies as higher-order HFN nodes.

## The Two-Tier Cognitive Architecture

The experiment implements a 4-process cluster organized into two tiers:

### Tier 1: Perceptual Specialists (L1)
*   **L1 Spatial Specialist**: Recognizes geometric transformations (rotations) in 3x3 grids.
*   **L1 Symbolic Specialist**: Recognizes object counts (1-4 active pixels).
*   **Behavior**: These processes perform the initial "Stereo Vision" detection by independently explaining the raw observation stream.

### Tier 2: Relational Synthesizer (L2)
*   **L2 Relational Specialist**: Observes a new 130-dimensional manifold.
*   **Message Protocol**: The Governor receives the predictive "Identity" ($\mu$) from the L1 winners and concatenates them into a single L2 observation vector: $[\mu_{Spatial} \,\|\, \mu_{Symbolic}]$.
*   **Goal**: To discover and stabilize "Analogy Nodes" that represent the binding between a specific count and a specific geometric move.

## Task: "Count-Governed Rotation" (Rosetta Grounding)

The system is exposed to a synthetic dataset where the ARC spatial rule is strictly dependent on a symbolic property:
*   **The Rule**: Output = Input rotated by $N \times 90^\circ$, where $N$ is the number of active pixels.
*   **The Challenge**: L1 processes only see their domain (one sees a rotation, the other sees a count). Only the L2 process can "learn" the governing law that binds these two events together.

## Key Insights

- **Predictive Message Passing**: Proved that hierarchical routing is possible in a multi-process environment. L1 workers pass their "Explanation Winner" states up to the L2 worker via the Governor.
- **Analogy Stabilization**: The L2 manifold successfully discovered distinct nodes for the 4 permutations of the Rosetta rule. This demonstrates the birth of **Generative Reasoning**: formally representing the *cause* of a transformation.
- **Scale-Free Logic**: The L2 worker uses the exact same `Observer` and `HFN` logic as the L1 workers, confirming the "fractal uniformity" of the HPM framework—the same mechanics work for perceiving raw pixels and perceiving abstract concepts.
- **Governor Orchestration**: The experiment demonstrated that a central Governor can manage the complex timing and data-dependency of a two-tier multi-process hierarchy.

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_sovereign_meta.py
```
