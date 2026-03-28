# HFN / HPM — Future Directions

Notes on possible next steps. Not plans — observations and open questions
captured while the ideas are fresh.

---

## 1. LLM as Knowledge Oracle for HPM

### The integration point

HPM's residual surprise is an explicit signal: "I observed something my world
model cannot explain." This is a targeted query waiting to happen.

Rather than the Observer creating a blank Gaussian leaf node when residual
surprise is high, it could query an LLM: "I'm observing this pattern and my
world model can't explain it — what concept does this correspond to?" The LLM's
response seeds a new prior node with semantically informed μ and σ, rather than
a blank leaf.

```
Observation → Observer → residual surprise high
                              ↓
                    query LLM: "what is this?"
                              ↓
              LLM returns: concept name + description
                              ↓
        new HFN node with LLM-informed μ added to Forest
```

### The reverse direction

HPM's current best explanation becomes a scaffold for LLM reasoning. Rather
than open-ended LLM generation, HPM's explanation tree determines which part of
the LLM's knowledge to activate. "I've identified this as a spatial
transformation at Layer 3 — what do you know about spatial transformations?"
Grounded, targeted retrieval rather than unconstrained generation.

### The structural asymmetry

| | LLM | HFN/HPM |
|---|---|---|
| Knowledge | Vast, flat, frozen | Sparse, structured, online |
| Priors | Implicit in weights | Explicit, protected nodes |
| Self-model | None — can't tell grounded from interpolated | Explicit — knows prior vs learned |
| Novelty signal | None | Residual surprise |
| Learning | Requires retraining | Continuous, per-observation |

LLM knowledge is a very powerful but undifferentiated prior. HPM's explicit
structure — protected priors vs learned nodes, hierarchy of abstraction levels,
residual surprise — gives it the tools to use LLM knowledge selectively rather
than wholesale. The LLM fills HPM's gaps on demand; HPM tells the LLM exactly
where the gaps are.

### Where this belongs

This is an hpm_ai_v2 concern, not hfn. The hfn library stays domain-agnostic.
The HPM AI evaluator layer (see observer.py boundary note) would mediate between
the Observer's residual surprise signal and the LLM oracle. The Observer exposes
`absorption_candidates` and `residual_nodes`; the HPM AI evaluator decides
whether to query the LLM or let the Observer create a blank node.

---

## 2. Observer Evaluator Layer (hpm_ai_v2 integration)

The Observer currently makes structural decisions (absorption, compression)
internally. See `hfn/observer.py` module docstring for the full boundary note.

The intended future interface:
```
HPM AI Evaluator (uses fractal diagnostics + LLM oracle + domain knowledge)
    ↓ directives: obs.absorb(node), obs.compress(A, B)
Observer (exposes candidates, executes directives)
    ↓
Forest
```

When integrating hfn into hpm_ai_v2, the fractal strategies currently in the
Observer (`recombination_strategy`, `hausdorff_absorption_threshold`, etc.)
should migrate to the HPM AI evaluator layer. The Observer becomes pure
mechanics; the evaluator brings domain knowledge.

---

## 3. Fractal Diagnostics as HPM AI Instruments

The fractal toolkit (`hfn/fractal.py`) currently drives Observer decisions
via strategy parameters. In the HPM AI:

- `population_dimension` / `correlation_dimension` — convergence signal for
  the evaluator: is the world model building coherent structure?
- `information_dimension` — which nodes are doing real work vs dormant?
- `hausdorff_distance(learned, priors)` — which priors have coverage gaps?
- `intrinsic_dimensionality(observations)` — how complex is the domain?
  Informs how many priors are needed.
- `multifractal_spectrum` — which prior layers are geometrically dominant?

These become the HPM AI's instruments for self-assessment, not just
post-hoc diagnostics.

---

## 4. Tau Scaling by Layer

Currently `tau` is fixed across all layers. A prior calibrated for
`prior_signal` (very broad, σ=4) uses the same tolerance as `primitive_cell_23`
(tight, σ=1). This means broad priors compete unfairly with tight ones.

Fractal tau scaling: `tau_layer_n = tau_0 × r^n` where r < 1 gives each
layer an appropriate tolerance. This would matter more as the hierarchy deepens
beyond 7 layers.

---

## 5. Cross-Domain Transfer

The hfn library is domain-agnostic by design. The same Observer + Forest
machinery used for ARC-AGI-2 could be applied to:

- Time-series data (replace spatial primitives with temporal ones)
- Natural language tokens (replace pixel priors with phoneme/morpheme priors)
- Sensor streams (robotics, continuous perception)

The fractal diagnostics apply identically across domains — `intrinsic_dimensionality`
of the observation space tells you how many priors you need before you start.
