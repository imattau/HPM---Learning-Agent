# Sovereign Decoder Experiment (SP23)

This experiment evaluates the **Sovereign Decoder**, the generative (top-down) counterpart to the multi-process Sovereign AI architecture. It tests if the system can parallelize "Variance Collapse" across independent specialist processes, achieving **Stereo Action** (mixed-modal synthesis).

## Multi-Process Generative Architecture

The system uses a 3-process cluster coordinated by a Generative Governor:

1. **L2_Narrative (The Planner)**: Operates in the domain of high-level rules and events. It holds abstract goals (high variance) that need resolving.
2. **L1_Lexical (The Speaker)**: Operates in the domain of vocabulary tokens (Strings). Resolves abstract identity nodes into specific concrete tokens.
3. **L1_Motor (The Actor)**: Operates in the domain of physical coordinates (1D or 2D). Resolves spatial/action nodes into specific numerical coordinates.

## Task: "Say and Point"

We test the system with a multi-modal goal: `Goal_Identify_Red` (Name the object and point to it).
The Governor performs the following:
1. **Unfolds** the goal into two sub-goals: `[SubGoal_Say]` and `[SubGoal_Point]`.
2. **Dispatches** the sub-goals in parallel to the `L1_Lexical` and `L1_Motor` decoders.
3. **Constraint Resolution**: Both sub-goals share a topological edge to a central `Concept_Red` node. The L1 decoders must use this topological constraint to filter their candidates, ensuring they both target the correct entity in their respective domains.
4. **Synchronizes** the results into a mixed-modal execution script.

## Key Insights

- **Stereo Action**: Verified that a single complex goal can be decomposed and executed simultaneously by multiple sovereign domains.
- **Cross-Domain Constraint Resolution**: Proved that topological edges (`REFERS_TO`, `LOCATED_AT`) can synchronize independent decoders without requiring a monolithic world model. The Lexical process outputs "red", and the Motor process outputs coordinate "2.0".
- **Parallel Variance Collapse**: The experiment demonstrates that the Agnostic Decoder algorithm (`hfn/decoder.py`) is fully compatible with the multi-process worker architecture, enabling fast, asynchronous generation.

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_sovereign_decoder.py
```
