# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This project builds AI learning agents grounded in the **Hierarchical Pattern Modelling (HPM)** framework — a unifying hypothesis about human learning developed by Matt Thomson (see the working paper PDF in this repo). The aim is not to replicate existing architectures but to implement agents whose learning behaviour reflects HPM's structural principles.

## The HPM Framework (Core Concepts)

HPM defines learning as the progressive discovery, stabilisation, and refinement of hierarchical patterns across multiple levels of abstraction. Any implementation should reflect these four structural roles:

1. **Pattern substrates** — where patterns are encoded (neural weights, symbolic representations, external memory, etc.)
2. **Pattern dynamics** — how patterns are created, updated, stabilised, and lost (prediction error, reinforcement, associative mechanisms, structural integration)
3. **Pattern evaluators/gatekeepers** — what determines which patterns are worth keeping (reward signals, coherence requirements, curiosity/boredom mechanisms)
4. **Pattern fields** — the broader environment/context that shapes which patterns are selected and maintained

Patterns have three required properties: internal coherence, functional utility (support prediction/action/simulation), and evaluator reinforcement.

### Hierarchy of pattern levels (example from motor learning)
1. Sensory regularities
2. Latent structural representations
3. Relational rules
4. Generative rules (mental simulation)
5. Meta-patterns (monitoring, error correction, strategy)

HPM is a **conceptual lens**, not a specific algorithm. It is compatible with reinforcement learning, predictive processing, Bayesian inference, and associative learning — these are interpreted as implementations of pattern dynamics under constraints of energy, time, risk, and social context.

## Language & Stack

- **Python** is the primary implementation language.
- No specific libraries have been chosen yet — decisions should be guided by which pattern dynamics/substrates are being implemented (e.g. PyTorch/JAX for neural substrates, Gymnasium for environment interaction, Hugging Face for pre-trained model integration).
- Use a virtual environment (`venv` or `uv`). Once a `requirements.txt` or `pyproject.toml` exists, document the setup command here.

## Design Principles for Agents

- Agents should operate across **multiple abstraction levels** simultaneously, not just pattern-match at a single level.
- Learning episodes should be modelled as changes in how the system represents, predicts, and evaluates structure.
- Distinguish clearly between the substrate (where patterns live), the dynamics (how they change), and the evaluators (what selects them).
- HPM does not assume learners start as blank slates — agents may have innate priors that individual learning refines.
