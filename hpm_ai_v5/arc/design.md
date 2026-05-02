# ARC HPM v5

## Purpose

ARC is handled as a small, substrate-specific extension around the shared HPM v5 stack.

```text
ARC task JSON
→ ARC adapter pipeline
→ ARC polygraphs
→ ARC agent pipeline
→ validated output grid
```

## Phase 1 scope

Only a minimal subset is implemented initially:

- connected components
- colour mapping
- translation
- simple exact-match validation

The current implementation also covers a small but useful extension of that subset:

- rotation
- horizontal reflection
- vertical reflection
- crop-to-object

This is enough to test the architecture without overbuilding the solver.

## Folder split

- `hpm_ai_v5/arc/adapters/`
  - task parsing
  - grid normalization
  - object extraction
  - colour mapping
  - geometry hints
  - example-pair deltas
- `hpm_ai_v5/arc/polygraphs/`
  - pixel view
  - object view
  - colour view
  - geometry view
- `hpm_ai_v5/arc/agents/`
  - router
  - hypothesis generation
  - simulation
  - critic
  - output
- `hpm_ai_v5/arc/`
  - shared grid/object/transformation helpers
  - pipeline and solver entry point

## Design rule

Adapters create structural candidates.
Polygraphs create competing interpretations.
Agents test and select transformations.

The shared HPM core remains reusable and domain-agnostic.

## Next phases

1. Add symmetry completion and line extension.
2. Add composition of multiple simple patterns.
3. Add strategy memory keyed by task signature.
