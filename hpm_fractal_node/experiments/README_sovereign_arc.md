# Sovereign ARC Cluster Experiment (SP18)

This experiment applies the multi-process **Sovereign AI** architecture to the Abstraction and Reasoning Corpus (ARC). It tests the hypothesis that ARC puzzles are solved through **"Stereo Vision"**: the simultaneous pattern recognition across independent cognitive domains (Geometry + Logic).

## Multi-Modal Feature Manifold

Observations (Input/Output grid pairs) are projected into a 150-dimensional common latent space composed of three specialized slices:

1.  **Spatial Slice (100D)**: Encodes the pixel-wise transformation delta $(Output - Input)$, normalized and flattened to 10x10.
2.  **Symbolic Slice (30D)**: Encodes numerical invariants, including:
    *   Grid dimensions and their parities.
    *   Normalized color distribution counts (0-9).
    *   $\Delta$ Unique colors and $\Delta$ Active pixels.
    *   Presence of mathematical properties (primes, identity mapping).
3.  **Structural Slice (20D)**: Encodes topological features extracted via `scipy.ndimage`:
    *   Connected component counts.
    *   Vertical and Horizontal symmetry scores.
    *   Euler Characteristic (components - holes).
    *   Area of the largest connected object.

## Architecture

The experiment utilizes a 4-process cluster coordinated by a central **Governor**:

*   **Spatial Specialist**: A rigid observer (Degree 1.0) seeded with ARC spatial priors (rotation, symmetry, connectivity). It only observes the Spatial Slice.
*   **Symbolic Specialist**: A rigid observer (Degree 1.0) seeded with the 306-prior Math World Model. It only observes the Symbolic Slice.
*   **Explorer**: A wild observer (Degree 0.0) with high plasticity. It observes the entire 150D vector to capture novelty that specialists reject.
*   **Governor**: Manages multi-process communication via `multiprocessing.Queue`. It routes batches of tasks and aggregates "Stereo Vision" events.

## Key Findings (200 ARC Tasks)

- **Stereo Vision Frequency (~68%)**: A majority of ARC tasks trigger simultaneous explanations from both Spatial and Symbolic domains. This confirms that ARC rules are frequently mappings between geometric transformations and numerical constraints.
- **Symbolic Dominance (~30%)**: A significant portion of tasks are purely logical or counting-based, requiring no spatial delta understanding to explain.
- **The Explorer Catch (~2%)**: Exclusive Explorer "claims" identify genuine "Novelty Shocks"—tasks where the built-in world model is insufficient and the system must autonomously stabilize new pattern concepts.
- **True Multi-Core Scaling**: By running 4 independent `TieredForest` instances, the system utilizes multiple CPU cores simultaneously, bypassing the Global Interpreter Lock (GIL) for CPU-bound HFN logic.

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_sovereign_arc.py
```
