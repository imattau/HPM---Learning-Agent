# ARC World Model Design

**Date:** 2026-03-28
**Status:** Implemented and validated

## Summary

A four-layer world model for ARC grid observations, implemented entirely as HFN nodes within a single Forest. All layers are protected priors — invariant structural knowledge the Observer learns above, not within.

## Core Principle

The encoder is not a preprocessing step. It is a prior HFN inside the Forest, competing alongside all other nodes to explain raw pixel-space observations. When the Observer expands into it, it finds the world model priors as decomposition paths. No coordinate space translation occurs — everything lives in pixel space (D = rows × cols).

## Architecture

### Layer 1: Primitives (`arc_primitives.py`)

Atomic vocabulary — the smallest meaningful units in the ARC domain. All in pixel space, grid-size parameterised via `build_primitives(rows, cols)`.

| Node | D | mu encoding |
|---|---|---|
| `primitive_cell_rc` | rows×cols | one-hot at position (r,c) |
| `primitive_cell` | rows×cols | uniform 1/D — "a cell exists" |
| `primitive_row_r` | rows×cols | uniform across row r |
| `primitive_row` | rows×cols | centroid of all rows |
| `primitive_col_c` | rows×cols | uniform across col c |
| `primitive_col` | rows×cols | centroid of all cols |
| `primitive_region` | rows×cols | centre-weighted Gaussian blob |
| `primitive_relationship` | rows×cols | uniform 0.5 — "relationships exist" |
| `primitives_hfn` | rows×cols | centroid of all primitives, σ=3 |

### Layer 2: Relationships (`arc_relationships.py`)

Relational vocabulary — pure concepts describing how two primitives relate. Defined as centroids of the primitive pairs that exhibit each relationship type.

| Node | mu encoding |
|---|---|
| `relationship_adjacency` | centroid of all horizontally/vertically adjacent cell pairs |
| `relationship_mirror` | centroid of all H/V reflected cell pairs |
| `relationship_repeat` | centroid of cell pairs with consistent displacement |
| `relationships_hfn` | centroid of all relationship types, σ=3 |

**prim_* compressions** — typed relationship slots, prior compressions of primitive + relationship:

- `prim_adjacency` = `recombine(primitive_cell, relationship_adjacency)`
- `prim_mirror` = `recombine(primitive_cell, relationship_mirror)`
- `prim_repeat` = `recombine(primitive_region, relationship_repeat)`

When the Observer discovers specific adjacency/mirror/repeat instances from data, those learned compressions are grounded instances of these typed slots.

### Layer 3: World Model Priors (`arc_prior_forest.py`)

Conceptual-level priors encoding what kinds of properties and relationships exist in the ARC world. Grid-size parameterised via `build_prior_forest(rows, cols)`.

```
prior_grid
  ├── prior_extent
  ├── prior_density → prior_sparse / prior_medium / prior_dense
  └── prior_spatial_organisation
        ├── prior_row_band → prior_row_top / prior_row_mid / prior_row_bot
        ├── prior_col_band → prior_col_left / prior_col_mid / prior_col_right
        ├── prior_diagonal
        └── prior_corners

prior_structure → prior_connectivity / prior_symmetry / prior_boundary
prior_colour
prior_grid_transform → prior_size_preserving / prior_size_changing / prior_content_transform
prior_transformation → prior_placement / prior_substitution / prior_geometric
```

### Layer 4: Encoder HFN (`arc_encoder_hfn.py`)

A protected prior encoding the belief that "grids can be perceived through structural concepts." Its children are shared world model prior nodes — no copies.

```
encoder_hfn (σ=2)
  ├── prior_grid
  ├── prior_density
  ├── prior_spatial_organisation
  ├── prior_structure
  ├── prior_colour
  └── prior_transformation
```

The encoder does not transform observations. It is a structurally rich node in pixel space. When the Observer expands into it, it finds the world model priors as decomposition paths. It acts as an expansion path, not a direct explainer.

### Assembly (`arc_world_model.py`)

Single entry point: `build_world_model(rows, cols) → (Forest, registry)`.

The Forest's D = rows × cols (pixel space). All nodes in the registry are added to `protected_ids` — the Observer learns above this layer.

## Data Flow

```
raw grid (pixel vector, D=rows×cols)
  → Observer.observe(x)
  → Forest.retrieve(x, k=5)       ← finds candidate nodes by mu proximity
  → expansion loop
       ├── primitive_cell_rc       ← explains single-cell observations
       ├── prior_row_top           ← explains row-organised observations
       ├── prior_sparse            ← explains sparse observations
       ├── encoder_hfn             ← expands into priors when surprising
       └── ...
  → weight/score updates
  → new nodes created from residual surprise (reference priors as children)
```

## Validation Results (3×3, 250 observations, 5 passes)

| Layer | Observations explained |
|---|---|
| World model priors | 180 (72%) |
| Primitives | 33 (13%) |
| New learned nodes | 8 (3%) |
| Unexplained | 29 (12%) |

Total by priors: **85%**. Zero priors absorbed.

Primitives explain 33 observations the prior layer missed — single-cell patterns caught by `primitive_cell_01`, `primitive_cell_11`, `primitive_cell_00`.

Relationships and encoder act as expansion paths (σ=2/3), not direct explainers — by design.

## Design Decisions

**Shared nodes, not copies.** The encoder's children are the same HFN objects as the world model priors. An HFN can appear as a child of multiple parents — this is a DAG, not a tree. No duplication.

**Primitives as first-pass.** In future, specific `primitive_cell_rc` nodes could replace the cell priors in `experiment_arc_priors.py` — they encode the same concept but with a cleaner architectural home.

**Relationships not firing yet.** `prim_adjacency`, `prim_mirror`, `prim_repeat` have broad sigma — they're expansion targets. Specific relationship instances will emerge as compressed nodes from co-occurring primitives observed in data.

**Encoder is a prior.** In the ARC context, the belief that "grids can be perceived structurally" is not learned — it is given. The encoder is therefore protected, like all other priors.

## Future Work

- Extend `prim_*` compressions with specific relationship instances as the Observer discovers them
- Add `primitive_region` instances for common ARC object shapes
- Test on 10×10 data with `build_world_model(rows=10, cols=10)`
- Consider a perception prior for colour when encoding is extended beyond binary
