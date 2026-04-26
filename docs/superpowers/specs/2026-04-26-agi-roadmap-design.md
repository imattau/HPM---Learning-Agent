# HPM AI AGI Roadmap — Design Spec

**Date**: 2026-04-26  
**Branch**: hpm-ai-v4-dev  
**Status**: Draft

---

## 0. Purpose

This spec defines the next three milestones for moving the current HPM AI stack from a narrow text system toward a general predictive-control agent.

The current stack already supports:
- stacked pattern learning (`l1-l5`)
- episodic polygraph retrieval and consolidation
- learned decoder-policy selection
- reloadable bundles and transfer-style validation

The next step is to broaden the substrate without breaking HPM principles.

---

## 1. HPM Alignment Constraints

These are the invariants the roadmap must preserve:

- **No hard central controller**. Control must emerge from pattern competition and meta-pattern selection.
- **No special memory engine**. Memory is relational structure over episodes, not a separate subsystem with its own ontology.
- **Small-K, deep hierarchy**. Depth should increase capability more reliably than widening latent state.
- **Selection must be outcome-driven**. Bootstrap priors may help cold start, but must decay and never monopolize new structure.
- **Learned control over heuristics**. Heuristics are acceptable as scaffolding only if they decay beneath learned meta-patterns.
- **Validation must stay explicit**. Every new capability needs a held-out benchmark or transfer test.

If a milestone violates any of these, it is not aligned.

---

## 2. Milestone 1: Ground the Stack in Environment State

### Goal

Move HPM from text-only prediction toward a world-model that learns from state, action, and consequence.

### What changes

- Add an interactive environment stream beyond plain text.
- Represent each step as:
  - observation
  - action or candidate action
  - outcome / reward
  - stack state
- Feed these episodes into the existing multi-polygraph structure.

### HPM role

- `l1-l3` continue to learn structure from observations.
- The polygraph records action/outcome episodes.
- `l4` begins to model action-conditioned generative sequences.
- `l5` begins to bias which action-family or decoder-family to use in a context.

### Success criteria

- The system can learn from an environment stream that is not text.
- Retrieval returns useful prior episodes for repeated contexts.
- Held-out action-selection improves after reload.
- No manual policy tuning is required per environment.

### What this proves

The stack is learning a general predictive substrate, not just language statistics.

---

## 3. Milestone 2: Add Tool and Action Use

### Goal

Make the reasoner capable of selecting and sequencing external actions.

### What changes

- Add a tool/action interface:
  - search
  - transform
  - query
  - inspect
  - generate
- Let the reasoner plan over candidate action sequences.
- Let the meta-policy choose among action families using polygraph evidence.

### HPM role

- `l4` becomes the generative-planning level for action trajectories.
- `l5` becomes the meta-policy over action family, horizon, and constraint mode.
- Episodic communities should bias repeated successful tool-use patterns.

### Success criteria

- The system can solve a simple multi-step task by calling tools in sequence.
- The chosen policy improves over repeated episodes instead of resetting each run.
- Held-out tasks show transfer across prompt forms and tool combinations.
- Retrieval of prior episodes changes the chosen plan when the context changes.

### What this proves

HPM control can operate over actions, not only text continuation.

---

## 4. Milestone 3: Continual Learning and Self-Curriculum

### Goal

Turn the stack into a system that can choose what to practice next and keep improving without collapsing old skills.

### What changes

- Add curriculum selection as a learned meta-pattern.
- Use the polygraph to rank:
  - which tasks currently produce useful structure
  - which tasks are overfit
  - which tasks improve transfer
- Add continual learning validation across corpora, tools, and environments.

### HPM role

- `l5` becomes a curriculum/control policy over task families.
- The polygraph stores and consolidates task episodes across domains.
- Bootstrap priors should be near-zero once the meta-policy matures.

### Success criteria

- The system can train on one task family and improve on a held-out one.
- No catastrophic collapse when new tasks are introduced.
- Learned policy selection persists across reloads and phase changes.
- The system can identify which tasks are worth practicing next.

### What this proves

HPM can support continual self-improvement rather than only single-task learning.

---

## 5. Code Surfaces

Likely areas to extend:

- `hpm_ai_v4/agents/reasoning.py`
  - multi-polygraph retrieval
  - action/tool planning
  - outcome-aware control context
- `hpm_ai_v4/simulations/layered_agent.py`
  - task routing
  - policy selection
  - bundle save/load for broader state
- `hpm_ai_v4/agents/meta_decoder_policy.py`
  - learned policy over decoder and action families
- `hpm_ai_v4/simulations/`
  - one simulation per milestone
- `hpm_ai_v4/tests/`
  - held-out transfer and reload regressions

---

## 6. Anti-Patterns

Do not:

- hard-code a single “AGI controller”
- make memory a separate vector-store subsystem
- scale by width before depth is justified
- let bootstrap priors become permanent selection bias
- validate only on the training domain
- use a new mechanism without a benchmark that distinguishes it from the old one

---

## 7. Progression Rule

Only move to the next milestone when the current one passes a held-out test and the learned policy survives reload.

That is the HPM-compatible gate against premature architectural inflation.
