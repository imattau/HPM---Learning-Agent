# Compression vs Memorisation (Experiment 1)

This experiment checks whether HFN naturally discovers compositional structure from a repeated sequence.

## Setup

Phase 1 stream:

```
A B A B A B ...
```

Phase 2 stream:

```
A B C A B C ...
```

The experiment feeds composite observations using a sliding window so that A and B (and later A, B, C) co-occur in the same explanation pass, enabling co-occurrence compression.

## Expected progression

- Create and reinforce leaves `A`, `B`, `C`
- Compress repeated co-occurrence into `AB`
- Reuse `AB` in later explanations
- Compress `AB` with `C` into `ABC` (directly or via AB + C)

## What we measure

- node count over time
- first step when `AB` appears
- first step when `ABC` appears
- reuse rate of `AB` and `ABC` in explanation trees

## Failure modes

- **No compression**: leaves dominate, composites never appear
- **Over-compression**: composite appears too early and dominates with little reuse stability
- **Instability**: repeated create/compress cycles cause oscillating node counts

## Usage

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_compression_vs_memorisation.py
```
