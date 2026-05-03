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

The current implementation also covers a useful extension of that subset:

- rotation
- horizontal reflection
- vertical reflection
- crop-to-object
- edge lists
- object and shape signatures
- pairwise spatial relations
- relation graphs over object candidates
- grid, object, colour, and structural deltas
- image/object/transformation polygraphs

This is enough to test the architecture without overbuilding the solver, while
still giving the agent a richer structured representation bank to compose.

The low-level region labelling and distance views are now backed by standard
libraries (`scipy.ndimage`) rather than custom flood-fill code, which keeps the
adapter layer smaller and makes the ARC image views more reliable.

Shape feature extraction now uses `skimage.measure.regionprops` for region
metrics such as perimeter, solidity, extent, and Euler number. That keeps the
shape adapter aligned with the library-first direction instead of maintaining a
custom geometry classifier.

The regression suite now also uses real ARC-AGI-2 training JSON tasks from
`data/ARC-AGI-2/data/training` for the routes the current solver can actually
close end-to-end. That keeps the adapter and pattern-management path honest
without pretending the solver is broader than it is.

The objective report now includes both:

- a small ARC smoke subset for quick regression
- an opt-in full labeled ARC-AGI-2 training split for dataset-wide evaluation

The full split is behind `HPM_RUN_FULL_ARC=1` so normal development runs stay
fast, while a dedicated full benchmark still exists for boundary testing.

## Folder split

- `hpm_ai_v5/arc/adapters/`
  - task parsing
  - grid normalization
  - object extraction
  - colour mapping
  - geometry hints
  - example-pair deltas
  - structural analysis views
- `hpm_ai_v5/arc/polygraphs/`
  - image view
  - object view
  - transformation view
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

ARC-specific structure now lives in the adapter layer:

- `EdgeList` exposes adjacency structure.
- `EdgeList` is now graph-backed through `networkx.grid_2d_graph`.
- `ObjectDetector` and `ShapeDefiner` expose discrete object and shape views.
- `SpatialRelation` exposes between-object geometry.
- `ArcHypothesisAdapter` turns the structural views into explicit candidate transformations.
- `SymmetryCompletionAdapter` surfaces rotational and reflection completions.
- `LineExtensionAdapter` surfaces border and stroke extension candidates.
- `relation_graph(...)` uses `networkx` to package object relations as graph views.
- `ColorMapper` exposes palette statistics.
- `PatternMiner` exposes repeating sub-blocks.
- `GridDelta`, `ObjectDelta`, `ColorDelta`, and `StructuralDelta` expose training-time transformation evidence.
- `ImagePolygraph`, `ObjectPolygraph`, and `TransformationPolygraph` provide competing views for agreement.

## Next phases

1. Add composition of multiple simple patterns.
2. Add strategy memory keyed by task signature.
3. Expand hypothesis ranking across ARC families.
