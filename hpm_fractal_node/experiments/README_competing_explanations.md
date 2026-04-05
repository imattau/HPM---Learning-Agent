# Competing Explanations Experiment (Experiment 4)

This experiment creates a controlled conflict between two explanations for the same stream:

- `Option A`: reuse a simple incorrect node (`cheap_wrong`)
- `Option B`: choose a complex but correct structure (`complex_correct`)

The policy is local to the experiment harness. Core HFN modules are unchanged.

## What it tests

The experiment measures a three-way tradeoff:

- fit quality
- complexity penalty
- accumulated weight

The expected behavior:

- early: cheap reuse should dominate
- under sustained evidence: the complex correct structure should win

## Failure modes

- **Stagnation**: the run never shifts to the complex structure.
- **Premature complexity**: the complex structure dominates immediately.

## Output summary

The script reports:

- transition step
- weight traces for both explanations
- early cheap rate and late complex rate
- stagnation and premature complexity flags
- final verdict

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_competing_explanations.py
```
