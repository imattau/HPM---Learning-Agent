# HPM AI v5 Core

This is the minimal substrate-agnostic core for the v5 branch.

## Pipeline

```text
raw input
→ preprocessing adapter pipeline
→ polygraph views
→ HPM core
→ structured action
→ agent policy
→ postprocessing adapter pipeline
→ validated output
```

## Purpose

The core operates on structure, not raw content.
Capability grows through composable adapters around it.
An agent layer turns the generic core into goal-directed behaviour.

## Core principle

Adapters may depend on other adapters.
The HPM core must not depend on adapters.

The core receives only normalised structures:

- `State`
- `Delta`
- `Context`
- `Polygraph views`
- `Structured action`

## Shared packet model

Adapters communicate through a shared packet:

```python
packet = {
  "raw": raw_input,
  "clean": None,
  "tokens": None,
  "entities": None,
  "relations": None,
  "states": [],
  "deltas": [],
  "views": [],
  "core_action": None,
  "draft_output": None,
  "validated_output": None,
  "trace": []
}
```

Each adapter reads fields and writes new fields.

## Shallow hierarchy

- `State`
- `Delta`
- `Pattern`
- `PatternSequence`
- `Action`

Meta-patterns are deferred until this level is stable.

## Core rules

- Learn from deltas, not full replay.
- Reuse known patterns before creating new ones.
- Canonicalize equivalent patterns before comparing them.
- Compress repeated sequences into smaller reusable units.
- Let density, contextual recall, and goal utility shape selection.
- Evaluate polygraphs independently.
- Keep the API small and explicit.
- Keep domain logic out of the core.

## Folder split

- `hpm_ai_v5/adapter/`
  - shared packet model and dependency-aware registry
- `hpm_ai_v5/preprocessors/`
  - domain-specific adapters that populate `State`, `Delta`, and context
- `hpm_ai_v5/polygraphs/`
  - multi-view adapters that generate alternative structural views
- `hpm_ai_v5/core/`
  - pattern learning, retrieval, scoring, selection, simulation
- `hpm_ai_v5/postprocessors/`
  - domain-specific adapters that render and validate output

## Core model

- `State`
  - current observation plus goal/context
- `Delta`
  - transformation between two representations
  - carries a `level` field so the same abstraction can be used across layers
- `Pattern`
  - reusable delta template
  - density
  - context recall
  - utility
- `PatternSequence`
  - reusable sequence of patterns
  - sequence density
  - sequence context recall
  - sequence utility
- `PatternEngine`
  - observe
  - select
  - act

## Polygraph evaluation

Each polygraph is scored using a small reliability metric:

- concentration
- average density
- fragmentation

The goal is to prefer views that create stable reusable patterns and avoid noisy views that fragment the store.

## Selection model

Patterns and sequences are scored by the same small weighted sum:

```text
score = α * accuracy + β * density + γ * context_match + δ * goal_utility
```

Polygraph scores can be added as a small bias on top of pattern-level selection.

Longer-horizon planning should use agreement across polygraphs, not a single-view score.

## Reliability rules

- Low confidence can defer action.
- Every action keeps traceability.
- Postprocessing validates output before it escapes the pipeline.

## Minimal fixes in scope

- Generalized `Delta` with a `level` field.
- Canonical rotation-based comparison to reduce duplicate equivalents.
- Compact repeat-unit learning for repeated sequences.
- A shallow `PatternSequence` layer over pattern names.
- A polygraph layer with independent views and simple view scoring.
- Agreement-based selection for longer-horizon planning.
- A light pruning rule based on density and support.
- Derived context keys such as delta kind and delta shape.
- A dependency-aware adapter registry and shared packet model.
- A thin pipeline wrapper that connects preprocessing, polygraphs, core, and postprocessing.

## What this version does not include

- no deep recursive hierarchy
- no large competition graph
- no global planning tree
- no external persistence format
- no domain-specific logic in the core

## Agent layer

- `hpm_ai_v5/agents/`
  - owns goal handling
  - owns conversation/task state
  - orchestrates preprocessing, core decisions, and postprocessing
  - keeps specialised behaviour out of `PatternEngine`
  - can compose fixed or routed agent pipelines

The agent owns:

- adapter pipeline
- pattern library wiring
- conversation/task state
- goal handling
- postprocessing
- validation

Agent types can share the same interface while differing in adapter stacks and goals.

Agent pipelines are a higher-order behaviour layer above the core. They remain outside `PatternEngine`.

## Next steps

1. Add more adapters only where the core needs them.
2. Add longer-horizon planning only after the current loop is stable.
3. Add meta-patterns only after base reuse is proven.
