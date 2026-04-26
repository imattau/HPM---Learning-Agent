# HPM AI v4

HPM AI v4 is the current discrete, stacked implementation of the Hierarchical Pattern Modelling framework in this repository. It is a research prototype for learning, retrieval, and control over structured sequences.

It is not a general-purpose AGI system. The current scope is narrower and explicit:

- text continuation and repair
- structured text
- code/DSL transformation
- discrete environment-state control
- tool/action planning
- self-curriculum selection
- reloadable library and transfer experiments

## Summary

v4 is a stacked HPM system that learns from discrete streams, stores episodes, retrieves them through a multi-polygraph, and uses a learned meta-policy to choose output modes.

The key idea is:

- the core learner stays discrete and population-based
- the reasoner turns episodic structure into retrieval and planning support
- the decoder layer turns that support into task-specific output
- thin adapters expose new domains without changing the core

In practice, v4 is best understood as a general discrete predictive-control stack rather than a single text model.

## Design

### Control flow

The main data path is:

1. input arrives through a thin adapter
2. the stacked learner updates `L1 -> L3`
3. the reasoner records the episode into the polygraph
4. higher levels summarize control and plan quality
5. the meta-policy selects a decoder family / mode
6. the decoder renders the output
7. the outcome feeds back into the same ecology

### Representation flow

- `L1` compresses raw character streams into character-class structure.
- `L2` learns over `L1` latent state.
- `L3` learns over a soft summary of `L2`.
- `L4` learns over structured reasoner and plan-quality observations.
- `L5` learns decoder / policy selection.

### Memory flow

Episodes are stored once, then viewed through multiple relational graphs:

- context
- action
- outcome
- stage
- policy
- summary/community

This means memory is not a separate subsystem. It is pattern structure over episodes.

### Output flow

The output side is split from the learning core:

- `WordDecoder` for readable text
- `CharDecoder` for character-level continuation
- `TargetDecoder` for target-conditioned continuation
- `ConstrainedDecoder` for dictionary / grammar constrained text
- `ExplanationDecoder` for inspection

The meta-policy learns which output family works best in a given context. Bootstrap priors help cold start, but they are not intended to dominate long-term selection.

## What v4 is built around

The v4 stack is centered on the same HPM roles, but applied to a smaller and more testable substrate:

- **Patterns** live in `HPMAgent` populations.
- **Dynamics** are handled by the pattern operators and the replicator-style update loop.
- **Evaluation** comes from prediction, compression, plausibility, execution, and transfer signals.
- **Fields** are the discrete streams and task environments the system trains against.

The main orchestration pieces are:

- [`hpm_ai_v4/agents/agent.py`](./agents/agent.py) — core pattern population and online learning
- [`hpm_ai_v4/agents/reasoning.py`](./agents/reasoning.py) — retrieval, episodic polygraph, planning, and multi-graph support
- [`hpm_ai_v4/agents/decoders.py`](./agents/decoders.py) — task-specific output adapters
- [`hpm_ai_v4/agents/meta_decoder_policy.py`](./agents/meta_decoder_policy.py) — learned decoder/control policy
- [`hpm_ai_v4/simulations/layered_agent.py`](./simulations/layered_agent.py) — stacked `L1 -> L5` orchestration

## Current stack

### L1 to L3

The lower stack is a discrete hierarchy over character streams:

- `L1` learns character-class structure from raw text.
- `L2` learns over `L1` latent state.
- `L3` learns over a soft summary of `L2`.

### L4 and L5

The upper stack is control-oriented:

- `L4` learns over structured reasoner outputs and plan-quality signals.
- `L5` is the learned decoder/meta-policy layer.

This means the system can separate:

- content structure
- generative planning
- output-policy selection

### Episodic multi-polygraph

The reasoner stores episodes and retrieves them through multiple relational views:

- context
- action
- outcome
- stage
- policy
- summary/community

This is the current long-context memory mechanism. It is episode-based, not a separate memory engine.

## Supported domains

v4 currently supports multiple discrete domains through thin I/O adapters:

- plain text
- structured text / canonical JSON
- code/DSL
- environment state
- tool/action sequences
- curriculum selection

The adapters live in [`hpm_ai_v4/io/adapters.py`](./io/adapters.py). They translate domain-specific inputs and outputs into the same discrete HPM substrate.

## Output and control

The output layer is modular:

- `WordDecoder` for readable text
- `CharDecoder` for character-level continuation
- `TargetDecoder` for target-conditioned continuation
- `ConstrainedDecoder` for dictionary/grammar-filtered text
- `ExplanationDecoder` for stack/state inspection

Decoder choice is not hard-coded. It is learned through `MetaDecoderPolicy` and biased by the episodic polygraph.

## Library workflow

v4 includes an explicit library discipline:

- build seed libraries
- register them in a JSON registry
- validate held-out transfer
- promote reusable bundles

Relevant files:

- [`hpm_ai_v4/simulations/build_library.py`](./simulations/build_library.py)
- [`hpm_ai_v4/simulations/bootstrap_libraries.py`](./simulations/bootstrap_libraries.py)
- [`hpm_ai_v4/tools/library_registry.py`](./tools/library_registry.py)

The bootstrap pass populates the initial registry entries for:

- `text_seed`
- `structured_text_seed`
- `code_dsl_seed`
- `environment_seed`
- `tool_seed`
- `curriculum_seed`

## Simulations

### Text

- [`full_simulation.py`](./simulations/full_simulation.py) — stacked HPM over a Wikipedia-like character stream
- [`text_full_simulation.py`](./simulations/text_full_simulation.py) — target-conditioned text loop
- [`text_generalization_simulation.py`](./simulations/text_generalization_simulation.py) — held-out corpus validation
- [`text_repair_simulation.py`](./simulations/text_repair_simulation.py) — corruption/reconstruction baseline
- [`experiment_generative_output.py`](./simulations/experiment_generative_output.py) — readable output inspection

### Structured text and code/DSL

- [`structured_text_simulation.py`](./simulations/structured_text_simulation.py)
- [`code_dsl_simulation.py`](./simulations/code_dsl_simulation.py)

These use parser/canonicalization/execution feedback rather than dictionary/grammar alone.

### Environment, tool, curriculum

- [`hpm_environment_simulation.py`](./simulations/hpm_environment_simulation.py)
- [`hpm_tool_simulation.py`](./simulations/hpm_tool_simulation.py)
- [`hpm_curriculum_simulation.py`](./simulations/hpm_curriculum_simulation.py)

These are the current non-text control benchmarks.

### Transfer and function-level validation

- [`control_transfer_simulation.py`](./simulations/control_transfer_simulation.py)
- [`hpm_functional_simulation.py`](./simulations/hpm_functional_simulation.py)
- [`bootstrap_libraries.py`](./simulations/bootstrap_libraries.py)

## Quick start

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the main stack

```bash
PYTHONPATH=. python3 hpm_ai_v4/simulations/full_simulation.py \
  --corpus hpm_ai_v4/simulations/data/wiki_sample.txt \
  --steps 100000 \
  --checkpoint-dir /tmp/hpm_full
```

### Run the text repair benchmark

```bash
PYTHONPATH=. python3 hpm_ai_v4/simulations/text_repair_simulation.py \
  --corpus hpm_ai_v4/simulations/data/wiki_sample.txt
```

### Run the bootstrap library pass

```bash
PYTHONPATH=. python3 hpm_ai_v4/simulations/bootstrap_libraries.py \
  --registry library_bootstrap/registry.json \
  --root-dir library_bootstrap
```

### Run the focused tests

```bash
PYTHONPATH=. pytest -q \
  hpm_ai_v4/tests/test_reasoning.py \
  hpm_ai_v4/tests/test_layered_agent.py \
  hpm_ai_v4/tests/test_meta_decoder_policy.py \
  hpm_ai_v4/tests/test_text_full_simulation.py
```

## What the current system can do

At the moment v4 can:

- learn from discrete streams online
- generate and repair short text continuations
- run held-out text transfer checks
- use dictionary and grammar signals for text
- use parser and execution signals for code/DSL
- plan short action sequences in toy environments
- select curriculum families through learned control
- persist and reload the learned stack

## What it cannot do yet

The current implementation does not prove:

- open-ended general intelligence
- robust long-horizon autonomy
- broad multimodal grounding
- theorem proving or general mathematical reasoning
- strong real-world planning across arbitrary tasks

## Design constraints

The current v4 code follows a few strict rules:

- keep the core discrete
- use thin adapters for new domains
- avoid hard-coded decoder controllers
- let bootstrap priors decay
- validate transfer before promotion
- keep memory episode-based and relation-based

## Status

The repository includes tests covering the core learning stack, decoders, polygraph memory, text simulations, structured text, code/DSL, environment, tool, curriculum, and library registry paths.

If you are extending v4, the safest next step is usually:

1. add a thin adapter for the new domain
2. add a simulation
3. add a transfer test
4. register and validate the resulting bundle
