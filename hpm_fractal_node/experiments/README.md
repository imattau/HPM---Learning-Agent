# HPM Fractal Node — Experiment Suite

This directory contains experiments that test the Hierarchical Fractal Node (HFN) system against
real and synthetic datasets. Each experiment exercises a different combination of HPM components
(world model, Observer dynamics, Query/Converter gap-filling) and measures how well the emerging
node population reflects the structure of the domain.

## Background

The HFN system implements the HPM (Hierarchical Pattern Modelling) framework as a computational
substrate. The core components are:

- **HFN (Hierarchical Fractal Node)**: A Gaussian node that encodes a pattern as a mean vector
  (mu) and covariance (sigma). Nodes can have children, forming a hierarchy.
- **Forest**: A collection of HFNs forming the world model — the agent's prior knowledge about
  structure in a domain.
- **Observer**: Drives pattern dynamics. For each observation vector it (1) finds which nodes
  explain it (log-probability gating by tau), (2) updates weights by a gain/loss rule, (3)
  absorbs redundant nodes, (4) compresses co-occurring nodes, and (5) creates new nodes for
  genuinely novel observations (residual surprise).
- **Query / Converter**: Gap-driven knowledge injection. When no node explains an observation
  well enough, a "gap" query is fired to an external source (Python stdlib, LLM), and the
  response is converted into new HFN nodes seeded near the gap.
- **Fractal metrics**: Box-counting dimension, Hausdorff distance, and self-similarity score
  measure whether the learned node population converges toward an attractor with coherent
  hierarchical structure — the HPM prediction for a well-seeded world model.

---

## Experiment Table

| File | Domain | What it tests | Status |
|---|---|---|---|
| `experiment_arc_observer.py` | ARC-AGI-2 (3x3 binary) | Bare Observer with no priors — what structure emerges from scratch? | Working |
| `experiment_arc_priors.py` | ARC-AGI-2 (3x3 binary) | Cell-position priors as structural children of pattern nodes; shared leaf identity | Working |
| `experiment_arc_prior_forest.py` | ARC-AGI-2 (3x3 binary) | Pre-populated prior forest (spatial, transformation, relationship priors); prior weight dynamics | Working |
| `experiment_arc_world_model.py` | ARC-AGI-2 (3x3 binary) | Full layered world model (primitives, relationships, priors, encoder); coverage by layer | Working |
| `experiment_arc_colour.py` | ARC-AGI-2 (3x3 colour) | Value encoding vs binary encoding; whether colour priors improve coverage | Working |
| `experiment_arc_10x10.py` | ARC-AGI-2 (10x10 colour) | Full world model on 10x10 grids; density distribution of discovered nodes | Working |
| `experiment_fractal_diagnostic.py` | ARC-AGI-2 (3x3 colour) | Box-counting dimension of node population per pass; IFS convergence hypothesis | Working |
| `experiment_fractal_hausdorff.py` | ARC-AGI-2 (3x3 colour) | Hausdorff distance (learned nodes vs priors) per pass; world-model vs no-priors | Working |
| `experiment_fractal_self_similarity.py` | ARC-AGI-2 (3x3 colour) | Self-similarity score (CV of log-count differences) per pass; world-model vs no-priors | Working |
| `experiment_dsprites.py` | dSprites (16x16 binary) | Generative factor alignment: do learned nodes align with shape/scale/position? | Working |
| `experiment_nlp.py` | NLP / child language | Semantic category alignment; QueryLLM gap-filling; TieredForest | Working |
| `experiment_code.py` | Python code tokens | Category purity (control_flow, functions, builtins, data); QueryStdlib gap-filling | Working |

> The ARC experiments require the ARC-AGI-2 dataset at `data/ARC-AGI-2/data/training/`.
> The dSprites experiment requires the dSprites `.npz` file (see `hpm_fractal_node/dsprites/`).
> The NLP experiment downloads Peter Rabbit automatically on first run.
> The code experiment builds a world model on first run and caches it to `data/code_world_model.*`.

---

## How to Run

All experiments are runnable from the repository root with the `PYTHONPATH` set:

```bash
cd /path/to/HPM---Learning-Agent
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_<name>.py
```

Alternatively, most experiments support module invocation (check each file's docstring for the
exact command):

```bash
python3 -m hpm_fractal_node.experiments.experiment_fractal_diagnostic
```

To run all ARC experiments in sequence:

```bash
for name in arc_observer arc_priors arc_prior_forest arc_world_model arc_colour arc_10x10; do
    PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_${name}.py
done
```

---

## Dependencies

The experiments share a common dependency set. From the repository root:

```bash
pip install -r requirements.txt
```

Key libraries used across experiments:

| Library | Used by |
|---|---|
| `numpy` | All experiments |
| `hfn` (local) | All experiments — the core HFN/Forest/Observer implementation |
| `ollama` (via HTTP) | `experiment_nlp.py` — requires a running ollama server with `tinyllama:latest` |
| `requests` | `experiment_nlp.py` (QueryLLM) |

The `hfn` package lives at the repository root and is imported directly via `PYTHONPATH`.

---

## Experiment Groups

### ARC-AGI-2 Experiments

Six experiments on the ARC-AGI-2 dataset, progressing from the simplest (no priors, bare
Observer) to the richest (full layered world model, colour encoding, 10x10 grids). These form
a natural progression for understanding how prior knowledge shapes what the Observer learns.

The ARC domain is well-suited for testing HPM because ARC puzzles exhibit clear spatial
regularities (symmetry, repetition, colour rules) that a structured world model should be
able to represent. The key question in each experiment is: does the prior forest help the
Observer explain more observations, and does it reduce the number of new nodes that need to
be created?

### Fractal Geometry Experiments

Three experiments that treat the learned node population as a point set in mu-space and measure
its geometric properties across passes. These directly test the IFS (Iterated Function System)
convergence hypothesis: if the Observer's recombination acts as a contracting affine map, the
node population should converge to a fractal attractor shaped by the prior world model.

The three metrics are complementary:
- **Box-counting dimension** (`experiment_fractal_diagnostic.py`): is the node distribution
  getting more or less space-filling over time?
- **Hausdorff distance** (`experiment_fractal_hausdorff.py`): are learned nodes closing in on
  the prior nodes, or drifting away?
- **Self-similarity score** (`experiment_fractal_self_similarity.py`): does the node population
  exhibit power-law scaling (a signature of fractal structure)?

All three compare a world-model-seeded Observer against a no-priors baseline.

### Generative Factor Experiments

`experiment_dsprites.py` and `experiment_nlp.py` test whether the Observer discovers latent
structure that corresponds to known ground-truth categories. dSprites provides clean generative
factors (shape, scale, orientation, position); the NLP experiment provides semantic word
categories. These experiments measure *purity*: the fraction of a node's attributed observations
that share the same ground-truth label.

### Gap-Filling Experiments

`experiment_code.py` and `experiment_nlp.py` both activate the Query/Converter pipeline.
`experiment_code.py` uses `QueryStdlib` (searches Python stdlib source for token signatures);
`experiment_nlp.py` uses `QueryLLM` (queries TinyLlama via ollama for semantic neighbours of
unknown words). These experiments test whether externally injected knowledge can bridge gaps
in the world model when observations fall outside the current node coverage.

---

## Key Concepts Tested

| HPM concept | Experiments that test it |
|---|---|
| Pattern substrates (HFN nodes as encodings) | All |
| Prior knowledge shapes what is learned | `arc_prior_forest`, `arc_world_model`, fractal trio |
| Hierarchical decomposition (children) | `arc_priors` |
| Weight dynamics (gain on match, loss on miss) | All Observer experiments |
| Absorption (redundant nodes removed) | All Observer experiments |
| Compression (co-occurrence creates higher nodes) | All Observer experiments |
| Gap queries (external knowledge injection) | `experiment_code`, `experiment_nlp` |
| Fractal attractor convergence | Fractal trio |
| Unsupervised category discovery | `experiment_dsprites`, `experiment_nlp` |
| Latent generative factor alignment | `experiment_dsprites` |

---

For detailed documentation of the two most developed experiments, see:

- [`README_code.md`](README_code.md) — Python code token experiment
- [`README_nlp.md`](README_nlp.md) — NLP semantic category experiment
