# Experiment 7: Closed-Loop Learning (World Model Refinement)

**Script:** `experiment_closed_loop.py`

## Objective
To validate the fundamental HPM cycle: `observe → explain → fail → create → re-observe`. This experiment tests the system's ability to refine its world model over time without explicit supervision or task-specific reasoning scaffolds, tracking residual surprise and structural complexity over continuous observation epochs.

## Setup
- **Curriculum:** A sequence of 5 distinct, sharp Gaussian patterns (100D vectors with a strong localized signal).
- **Environment:** A single `Observer` processing the curriculum over 10 epochs.
- **Configurations:** 
  - `tau=1.0` (Surprise threshold)
  - `residual_surprise_threshold=1.5`
  - `adaptive_compression=True`
  - `compression_cooccurrence_threshold=2`

## Results & Analysis
The experiment successfully demonstrated the classic "S-curve" of learning and autonomous model refinement:

1. **Discovery Phase (Epochs 1-6):** The system encountered novel patterns, yielding high residual surprise. It rapidly expanded its hypothesis space, creating new nodes (Forest size grew from 5 to 80 nodes).
2. **Breakthrough (Epoch 7):** The structural search finally synthesized an arrangement of nodes that perfectly explained the curriculum. Residual surprise collapsed to **0.000000**.
3. **Refinement & Compression (Epochs 8-10):** With the patterns fully explained, the system's adaptive compression dynamics engaged. Redundant, overlapping, and obsolete nodes were pruned and absorbed. The forest size shrank from a peak of 136 nodes down to a stable 106 nodes, while maintaining zero surprise.

### Key Takeaway
The HFN system actively improves its own world model. It creates structure to minimize surprise, and then compresses that structure to maximize efficiency, proving the core HPM tenet: learning is structural self-organization under pressure.
