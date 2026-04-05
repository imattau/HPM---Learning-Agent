# Local Density Stress Test (Experiment 4)

This experiment probes the lacunarity-guided creation path in `Observer`.

## Setup

- Seed a dense cluster of similar nodes.
- Seed a sparse region elsewhere.
- Inject new inputs inside the dense region that are locally novel.

## What it tests

The failure mode is **global density suppresses learning**:
if the local-density suppression is too aggressive, the dense region never
creates new nodes, even when residual surprise is high.

Expected outcomes:

- **Working**: the dense region still differentiates locally (at least one new node).
- **Broken**: the dense region never creates nodes (stagnation) or creates too many (explosion).

## Output summary

The script reports:

- dense-phase creations
- sparse-phase creations
- mean density ratios in dense vs sparse regions
- residual surprise mean
- verdict (`WORKING`, `BROKEN_STAGNATION`, `BROKEN_EXPLOSION`)

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_local_density_stress.py
```
