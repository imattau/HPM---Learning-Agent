# Dynamic Specialist Promotion Experiment (SP25)

This experiment demonstrates **Emergent Sovereignty** in the HPM AI architecture. It tests the system's ability to autonomously detect mature cognitive structures in a generalist process and "promote" them into dedicated, specialized worker cores.

## Architecture: The Maturation Loop

The system starts with a single monolithic **Explorer** process. As it processes a curriculum of structured signal and random noise, the system follows a 3-phase maturation lifecycle:

1.  **Local Maturation**: The Explorer process uses `adaptive_compression` to locally discover parent HFN nodes that explain recurring co-occurrences in the signal.
2.  **Recognition & Extraction**: The central Governor monitors the Explorer's forest. When a parent node achieves high predictive weight, the Governor identifies it as a "Mature Concept." It uses the **Agnostic Decoder** logic to unfold the node into its full sub-tree (identifying all necessary child ingredients).
3.  **Sovereign Promotion**: The Governor spawns a new dedicated worker process (**Spec_Signal**) and seeds its forest with the extracted sub-tree.
4.  **Specialized Routing**: The Governor updates its routing table. Signal observations are now sent directly to the new Specialist core, while noise remains with the Explorer.

## Theoretical Insight: Natural Forgetting

A key feature of this experiment is **Natural Forgetting via Signal Redirection**. 

Unlike traditional database management, the Governor does not explicitly delete nodes from the Explorer. Instead, once the signal is redirected to the Specialist, the redundant nodes in the Explorer's forest naturally lose weight due to lack of reinforcement. They are then **Garbage Collected** by the standard `TieredForest` sweep mechanism.

## Key Findings

- **Autonomous Domain Discovery**: The system successfully transitioned from a 1-process monolithic state to a 2-process specialized state based on purely geometric and topological triggers.
- **Hierarchical Extraction**: Proved that the `hfn.Decoder` logic can be used for structural housekeeping—unfolding a complex concept into its component parts for relocation.
- **Systemic Efficiency**: By offloading the dense signal to a specialized core, the Explorer is freed to continue its high-plasticity discovery role in the remaining noise.

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_dynamic_promotion.py
```
