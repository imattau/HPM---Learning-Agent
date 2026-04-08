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
- **Meta-Forest**: A second-order learning layer `TieredForest(D=4)` that tracks node performance (weight, score, hit/miss counts, co-occurrence, recurrence) as HFN nodes themselves.
- **Observer (Dynamics + Control + Policy)**: Drives all pattern dynamics. Features include:
  - *Cost-aware attention*: `priority = surprise - weight` (attention is learned, trusted nodes are explored last).
  - *Stability mechanisms*: active pruning, global weight decay, and absorption thresholds to prevent structural explosion.
  - *Density-aware creation*: incorporates lacunarity and multifractal behavior to suppress redundant node creation.
  - *Dynamic Priors*: Prior nodes are not entirely static; they can undergo plasticity (drift) under consistent failure.
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
| `experiment_lexical_semantic_forest.py` | WordNet lexical ontology + Peter Rabbit corpus | Several-thousand-node external prior library grounded in real text; lemma, synset, relation, and abstraction roots; compact vs large prior comparison | Working |
| `experiment_lexical_transfer.py` | WordNet lexical transfer + mixed corpus | Compares in-domain Peter Rabbit/repo text against out-of-domain vocabulary over the WordNet ontology; measures coverage, learned-node recovery, and abstraction transfer | Working |
| `experiment_lexical_curriculum.py` | WordNet lexical curriculum + persistent forest | Reuses the same WordNet forest across easy, medium, and hard text streams to test whether stretching the model leaves behind reusable learned structure | Working |
| `experiment_lexical_consolidation.py` | WordNet lexical consolidation + persistent forest | Replays the original seed stream after stretching the same forest through harder lexical stages to test retention, reuse, and consolidation | Working |
| `experiment_lexical_pressure_revisit.py` | WordNet lexical pressure + revisit | Applies an explicit hot-cache shock before replaying the seed stream to test whether learned nodes survive actual forgetting pressure | Working |
| `experiment_lexical_cross_domain_replay.py` | WordNet lexical cross-domain replay | Switches from lexical pressure into code-like observations, then selectively replays high-utility seed nodes before revisiting the original lexical stream | Working |
| `experiment_lexical_math_cross_stream_replay.py` | WordNet lexical cross-stream replay + math stream | Switches from lexical pressure into a math-text observation stream, then selectively replays high-utility seed nodes before revisiting the original lexical stream | Working |
| `experiment_code.py` | Python code tokens | Category purity (control_flow, functions, builtins, data); QueryStdlib gap-filling | Working |
| `experiment_math.py` | Integer arithmetic | Algebraic rule discovery; 306-prior library across 6 abstraction levels; no LLM | Working |
| `experiment_math_throughput.py` | Integer arithmetic throughput sweep | Measures observations/sec, warm-pass slowdown, and full vs diagonal sigma storage across sample sizes | Working |
| `experiment_math_controller.py` | Integer arithmetic controller adapter | Compares direct HFN math execution against the async controller layer with ingest, replay, prefetch, and snapshot/export | Working |
| `experiment_multi_observer_lifecycle.py` | Math + Text (216D) | Parallelized multi-process observers (Math, Text, and Mixed) across a 6-stage lifecycle | Working |
| `experiment_sovereign_arc.py` | ARC-AGI-2 (10x10) | Multi-process "Stereo Vision" cluster (Spatial, Symbolic, Explorer) for ARC tasks | Working |
| `experiment_sovereign_meta.py` | Rosetta Grounding | Two-tier hierarchy (L1 Perceptual -> L2 Relational) for cross-domain analogy discovery | Working |
| `experiment_thematic_anchor.py` | Peter Rabbit (2000 tokens) | Structural summarization and narrative motif discovery via multi-tier synthesis | Working |
| `experiment_toddler_generator.py` | Top-Down Synthesis | Generates simple sentences ("Mum ate apple") by coordinating an 8-domain "Structural Octet" | Working |
| `experiment_agnostic_decoder.py` | Variance Collapse | Universal, domain-agnostic top-down resolution via geometric variance and HFN topology | Working |
| `experiment_sovereign_decoder.py` | Stereo Action | Multi-process, mixed-modal (Lexical + Motor) top-down synthesis of a "Say and Point" goal | Working |
| `experiment_demand_driven_learning.py` | Fail-Learn-Retry | Active learning triggered by generative gaps (Curiosity Engine) with hallucination guarding | Working |
| `experiment_competing_explanations.py` | Competing Explanations | Cheap-reuse vs expensive-correct structure under repeated ambiguity pressure | Working |
| `experiment_compression_vs_memorisation.py` | Compression vs Memorisation | AB then ABC streams to test composite emergence and reuse | Working |
| `experiment_absorption_as_generalisation.py` | Absorption as Generalisation | ABC/ABD/ABE variants under shared AB pressure; tests absorption as abstraction | Working |
| `experiment_near_miss_learning.py` | Near-Miss Learning | Train on ABC then probe AB? to test predictive completion vs creation | Working |
| `experiment_local_density_stress.py` | Local Density Stress | Dense cluster vs sparse region; tests local differentiation under lacunarity suppression | Working |
| `experiment_multifractal_learning.py` | Multifractal Learning | Dense inputs vs sparse uniques; compression dominance vs creation dominance | Working |
| `experiment_forgetting_vs_persistence.py` | Forgetting vs Persistence | Phase1 ABC then Phase2 XYZ then replay; weight decay + pruning test continual learning | Working |
| `experiment_dynamic_promotion.py` | Emergent Sovereignty | Autonomous specialist process spawning via decoder-led sub-tree extraction | Working |
| `experiment_emergent_routing.py` | Decentralized Sovereignty | Multi-process broadcast and claim model using HFN competence gates | Working |
| `experiment_thinking_arc_solver.py` | Thinking Solver | Iterative hypothesis testing and negative anchoring for ARC 30x30 | Working |
| `experiment_study_and_test.py` | Study-and-Test | Meta-transfer learning across a curriculum of persistent ARC tasks | Working |
| `experiment_closed_loop.py` | Closed-Loop Learning | `observe → explain → fail → create → re-observe` cycle; Tracks surprise reduction and structural compression over time | Working |
| `experiment_meta_hfn.py` | Meta-HFN Utilisation | A/B tests self-representation (`meta_forest`) vs ablated baseline under resource pressure to measure adaptation speed and structural efficiency | Working |
| `experiment_goal_reasoning.py` | Goal-Conditioned Reasoning | First step to agency: `goal + input → plan → execute → evaluate` loop using GoalConditionedRetriever | Working |
| `experiment_multi_step_reasoning.py` | Multi-Step Reasoning | Continuous "Chain-of-Thought": sequences multiple atomic HFN rules to reach a distal goal state | Working |
| `experiment_true_cross_domain.py` | True Cross-Domain Transfer | Achieves 100% structural reuse across orthogonal domains (Math -> Symbolic) without manual alignment | Working |
| `experiment_self_curiosity.py` | Self-Curiosity | Autonomous learning loop (`generate → observe → evaluate → expand`) without external data stream | Working |
| `experiment_belief_revision.py` | Competing Hypotheses | Tests belief revision and falsification dynamics in ambiguous environments; tracks weight trajectories | Working |
| `experiment_world_model_simulation.py` | World Model Simulation | "Imagination Test": iteratively simulates future trajectories via relational [State, Delta] encoding | Working |
| `experiment_hierarchical_abstraction.py` | Hierarchical Abstraction | Validates core HPM claim: builds multi-layered DAGs and reuses components (e.g. letters -> words -> sentences) | Working |
| `experiment_multi_agent_social.py` | Social Learning | Knowledge transfer between sovereign agents via dream broadcasting and social refinement | Working |
| `experiment_unified_cognitive_loop.py` | The Core Agent | "Capstone": autonomous Plan -> Act -> Fail -> Explore -> Re-Plan loop with belief revision | Working |
| `experiment_long_horizon_reasoning.py` | Long-Horizon Reasoning | Depth Test: stability and scalability of reasoning chains up to 20 steps with distractors | Working |
| `experiment_adversarial_belief_revision.py` | Adversarial Belief Revision | Truth Under Conflict: unlearning entrenched, high-confidence incorrect beliefs | Working |
| `experiment_developmental_cognitive_system.py` | Developmental System | **Nested Composition**: hierarchical knowledge accumulation rendered into Python | Working |
| `experiment_recursive_scaling.py` | Recursive Complexity Scaling | Algorithmic curriculum building nested abstraction graphs (maps/loops) | Working |
| `experiment_autonomous_pruning.py` | Autonomous Graph Pruning | Pruning combinatorial search space via internal simulation dreams | Working |

> The ARC experiments require the ARC-AGI-2 dataset at `data/ARC-AGI-2/data/training/`.
> The dSprites experiment requires the dSprites `.npz` file (see `hpm_fractal_node/dsprites/`).
> The NLP experiment downloads Peter Rabbit automatically on first run.
> The lexical-semantic experiment reuses the same Peter Rabbit corpus as real text observations over the WordNet ontology.
> The lexical-transfer experiment compares in-domain Peter Rabbit/repo text with out-of-domain vocabulary over the WordNet ontology.
> The lexical-curriculum experiment runs the same WordNet forest across progressively harder stages to test cumulative improvement over time.
> The lexical-consolidation experiment replays the original seed stream after stretching the same forest to test whether earlier learned structure is retained and reused.
> The lexical-pressure experiment applies a hot-cache shock before the replay to test whether learned nodes survive actual forgetting pressure.
> The lexical cross-domain replay experiment switches to code-like observations and selectively replays high-utility seed nodes before returning to the lexical stream.
> The lexical math cross-stream replay experiment switches to a math-text stream and then selectively replays high-utility seed nodes before returning to the lexical stream.
> The code experiment builds a world model on first run and caches it to `data/code_world_model.*`.
> The math throughput experiment measures observations/sec and warm-pass slowdown across sample sizes, with full vs diagonal sigma storage.
> The math controller experiment compares the direct math loop against the new async controller layer.

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
| Prior knowledge shapes what is learned | `arc_prior_forest`, `arc_world_model`, fractal trio, `experiment_math` |
| Hierarchical decomposition (children) | `arc_priors` |
| Weight dynamics (gain on match, loss on miss) | All Observer experiments |
| Absorption (redundant nodes removed) | All Observer experiments |
| Compression (co-occurrence creates higher nodes) | All Observer experiments |
| Gap queries (external knowledge injection) | `experiment_code`, `experiment_nlp` |
| Fractal attractor convergence | Fractal trio |
| Unsupervised category discovery | `experiment_dsprites`, `experiment_nlp`, `experiment_math` |
| Latent generative factor alignment | `experiment_dsprites` |
| Algebraic rule discovery (pure geometry) | `experiment_math` |
| Large structured prior library (6 levels) | `experiment_math` |
| Large lexical-semantic prior library (WordNet external ontology + corpus) | `experiment_lexical_semantic_forest` |
| Lexical transfer across in-domain and out-of-domain vocabularies | `experiment_lexical_transfer` |
| Curriculum stretching over progressively harder lexical stages | `experiment_lexical_curriculum` |
| Multi-process parallelization (one core per observer) | `experiment_multi_observer_lifecycle` |
| Cross-domain orchestration and on-demand specialist creation | `experiment_multi_observer_lifecycle` |
| Stereo Vision (cross-domain synthesis) | `experiment_sovereign_arc` |
| Hierarchical message passing (L1 -> L2) | `experiment_sovereign_meta` |
| Analogy stabilization (Joint Identities) | `experiment_sovereign_meta` |
| Negative Selection via Priors | `experiment_thematic_anchor` |
| Structural Summarization (Information Bottlenecking) | `experiment_thematic_anchor` |
| Top-Down Synthesis (Generation) | `experiment_toddler_generator` |
| Multi-Domain Constraint Resolution | `experiment_toddler_generator` |
| Variance Collapse (abstraction as variance) | `experiment_agnostic_decoder` |
| Universal HFN Decoding (domain-agnostic) | `experiment_agnostic_decoder` |
| Stereo Action (mixed-modal synthesis) | `experiment_sovereign_decoder` |
| Parallel Variance Collapse | `experiment_sovereign_decoder` |
| Demand-Driven Learning (Active Curiosity) | `experiment_demand_driven_learning` |
| Generative Anti-Hallucination Guarding | `experiment_demand_driven_learning` |
| Emergent Sovereignty (autonomous scaling) | `experiment_dynamic_promotion` |
| Natural Forgetting via Redirection | `experiment_dynamic_promotion` |
| Decentralized Sovereignty (broadcast and claim) | `experiment_emergent_routing` |
| HFN Competence Gating (global typicality) | `experiment_emergent_routing` |
| Iterative Hypothesis Testing (Thinking) | `experiment_thinking_arc_solver` |
| Negative Anchoring (falsified knowledge) | `experiment_thinking_arc_solver` |
| Meta-Transfer Learning (Study and Test) | `experiment_study_and_test` |
| Structural motif persistence across tasks | `experiment_study_and_test` |
| Closed-Loop Learning & Adaptive Compression | `experiment_closed_loop` |
| Meta-HFN Self-Representation under Pressure | `experiment_meta_hfn` |
| Goal-Conditioned Reasoning (Agency) | `experiment_goal_reasoning` |
| Intent-Driven Retrieval & Planning | `experiment_goal_reasoning` |
| Stateful Sequential Composition | `experiment_multi_step_reasoning` |
| Continuous Chain-of-Thought | `experiment_multi_step_reasoning` |
| True Abstraction (Rule vs Surface) | `experiment_true_cross_domain` |
| Surface-Structure Separation | `experiment_true_cross_domain` |
| Autonomous Learning Trajectory | `experiment_self_curiosity` |
| Generative-Perceptual Loops (Play) | `experiment_self_curiosity` |
| Belief Revision & Falsification | `experiment_belief_revision` |
| Confirmation Bias Resilience | `experiment_belief_revision` |
| Relational World Modeling | `experiment_world_model_simulation` |
| Stable Imagination (Dreaming) | `experiment_world_model_simulation` |
| Zero-Drift Extrapolation | `experiment_world_model_simulation` |
| Multi-layered DAG Emergence | `experiment_hierarchical_abstraction` |
| Compositional Generalization | `experiment_hierarchical_abstraction` |
| Transmissible Knowledge (Social Learning) | `experiment_multi_agent_social` |
| Distributed World-Model Discovery | `experiment_multi_agent_social` |

---

For detailed documentation of the most developed experiments, see:

- [`README_code.md`](README_code.md) — Python code token experiment
- [`README_nlp.md`](README_nlp.md) — NLP semantic category experiment
- [`README_math.md`](README_math.md) — Math arithmetic / algebraic rule discovery experiment
- [`README_math_throughput.md`](README_math_throughput.md) — math observations/sec benchmark
- [`README_math_controller.md`](README_math_controller.md) — async controller adapter comparison
- [`README_competing_explanations.md`](README_competing_explanations.md) — cheap reuse vs expensive correct structure tradeoff
- [`README_compression_vs_memorisation.md`](README_compression_vs_memorisation.md) — AB/ABC compression vs memorisation test
- [`README_local_density_stress.md`](README_local_density_stress.md) — local density suppression stress test

For detailed documentation of the lexical experiments, see:

- [`README_lexical_semantic_forest.md`](README_lexical_semantic_forest.md) — large WordNet-backed prior forest
- [`README_lexical_transfer.md`](README_lexical_transfer.md) — in-domain vs out-of-domain lexical transfer
- [`README_lexical_curriculum.md`](README_lexical_curriculum.md) — staged lexical stretching over time
- [`README_lexical_consolidation.md`](README_lexical_consolidation.md) — revisit after stretching to test retention and reuse
- [`README_lexical_pressure_revisit.md`](README_lexical_pressure_revisit.md) — explicit memory shock before replay
- [`README_lexical_cross_domain_replay.md`](README_lexical_cross_domain_replay.md) — code-domain shift with selective replay
- [`README_lexical_math_cross_stream_replay.md`](README_lexical_math_cross_stream_replay.md) — math-text stream with selective replay

For detailed documentation of the Sovereign AI (multi-process) experiments, see:

- [`README_lifecycle.md`](README_lifecycle.md) — multi-observer lifecycle and multi-core orchestration
- [`README_sovereign_cluster.md`](README_sovereign_cluster.md) — the "Foundational Five" parallel cluster and Sovereignty Spectrum
- [`README_sovereign_arc.md`](README_sovereign_arc.md) — ARC "Stereo Vision" multi-process experiment
- [`README_sovereign_meta.md`](README_sovereign_meta.md) — hierarchical synthesis and cross-domain analogy stabilization
- [`README_agnostic_decoder.md`](README_agnostic_decoder.md) — universal Variance Collapse decoding
- [`README_sovereign_decoder.md`](README_sovereign_decoder.md) — multi-process "Stereo Action" generation
- [`README_demand_driven_learning.md`](README_demand_driven_learning.md) — fail-learn-retry curiosity loop
- [`README_dynamic_promotion.md`](README_dynamic_promotion.md) — emergent sovereignty and autonomous process scaling
- [`README_emergent_routing.md`](README_emergent_routing.md) — decentralized broadcast and claim model
- [`README_thinking_arc_solver.md`](README_thinking_arc_solver.md) — iterative hypothesis testing and negative anchoring
- [`README_study_and_test.md`](README_study_and_test.md) — meta-transfer learning across persistent tasks
- [`README_closed_loop.md`](README_closed_loop.md) — `observe → explain → fail → create → re-observe` cycle and adaptive compression
- [`README_meta_hfn.md`](README_meta_hfn.md) — A/B testing self-representation vs ablated baseline under resource pressure
- [`README_goal_reasoning.md`](README_goal_reasoning.md) — intent-driven retrieval and planning loop
- [`README_multi_step_reasoning.md`](README_multi_step_reasoning.md) — multi-step intent-driven retrieval and planning loop
- [`README_true_cross_domain.md`](README_true_cross_domain.md) — autonomous structural transfer across orthogonal domains
- [`README_self_curiosity.md`](README_self_curiosity.md) — autonomous generative exploration loop
- [`README_belief_revision.md`](README_belief_revision.md) — falsification and belief correction dynamics
- [`README_world_model_simulation.md`](README_world_model_simulation.md) — iterative future trajectory simulation via relational encoding
- [`README_hierarchical_abstraction.md`](README_hierarchical_abstraction.md) — multi-layered compositional DAG generation
- [`README_multi_agent_social.md`](README_multi_agent_social.md) — social knowledge transfer between sovereign agents
- [`README_unified_cognitive_loop.md`](README_unified_cognitive_loop.md) — capstone agentic loop with adaptation and belief revision

- [`README_long_horizon_reasoning.md`](README_long_horizon_reasoning.md) — depth test for stability of long reasoning chains
- [`README_adversarial_belief_revision.md`](README_adversarial_belief_revision.md) — truth under conflict: unlearning entrenched beliefs
- [`README_recursive_scaling.md`](README_recursive_scaling.md) — nested composition for algorithmic tasks (maps and loops)
- [`README_autonomous_pruning.md`](README_autonomous_pruning.md) — autonomous search space pruning via simulation dreams
