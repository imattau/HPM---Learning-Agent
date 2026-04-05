# Competing Explanations Experiment (Experiment 3)

This experiment creates a controlled conflict between two explanations for the same stream:

- `Option A`: reuse a known node (`cheap_known`) that is cheap but slightly wrong
- `Option B`: create a new precise node that is expensive but more correct

The policy is local to the experiment harness. Core HFN modules are unchanged.

## What it tests

The experiment measures a three-way tradeoff:

- fit quality
- complexity/structure penalty
- creation cost

Creation cost starts high and is reduced by repeated mismatch pressure. This
implements the desired behavior:

- early: cheap reuse should dominate
- under sustained pressure: building a new structure should eventually win

## Failure modes

- **Stagnation**: the run never creates a new structure despite persistent mismatch.
- **Explosion**: the run keeps creating duplicate new structures instead of converging.

## Output summary

The script reports:

- cheap reuse count
- new-structure creation step
- post-creation reuse count
- mean error before/after creation
- stagnation and explosion flags
- final verdict (`BALANCED`, `STAGNATION`, `EXPLOSION`)

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_competing_explanations.py
```
