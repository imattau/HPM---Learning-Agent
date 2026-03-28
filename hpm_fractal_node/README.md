# HPM Fractal Node (HFN)

A minimal implementation of the Hierarchical Pattern Modelling (HPM) framework applied to the ARC-AGI-2 benchmark. The core idea: learning is the progressive discovery and stabilisation of hierarchical patterns encoded as Gaussian distributions in a directed acyclic graph.

---

## Core Concepts

### The HFN

Every node in the system is an **HFN (HPM Fractal Node)** — a Gaussian identity fused with a DAG body:

```
HFN = N(μ, Σ)          ← compressed face: predictive identity
    + [child₁, child₂, …]  ← structural face: internal composition
```

**Fractal uniformity**: a Forest (the world model) is itself an HFN. Every level of the hierarchy has the same interface.

**Key properties:**
- No parent references — the graph is traversed top-down only
- Same node can be a child of multiple parents simultaneously (shared structure, not copies)
- No mutation from queries — observation and learning are the Observer's job, not the node's

### The Forest

A collection of active HFN nodes constituting the world model. The Observer adds, removes, and rewires nodes. Priors in the Forest are **protected** — the Observer cannot absorb or remove them.

### The Observer

Holds all dynamic state: weights, co-occurrence counts, absorption history. Takes observations (numpy vectors) and:

1. Finds the best-explaining node (lowest surprise below `tau`)
2. Updates weights (gain/loss)
3. Creates new nodes for high-residual observations
4. Compresses frequently co-occurring nodes via `recombine()`
5. Absorbs redundant nodes that are dominated by others

---

## The ARC World Model

The world model for ARC-AGI-2 is assembled in `arc_world_model.py` as 7 layers of protected priors plus one cross-layer bridge, all living in the same pixel space (D = rows × cols).

```
Layer 0  Perception     prior_signal → prior_pixel → prior_pixel_colour ──┐
                                     → prior_cell_concept                  │ (bridge)
                                     → prior_spatial_adjacency             │
                                           → prior_field                   │
                                                 → prior_grid ←────────────┘

Layer 1  Primitives     primitive_cell_rc (×D), primitive_row_r (×rows),
                        primitive_col_c (×cols), primitive_region,
                        primitive_relationship, primitives_hfn

Layer 2  Relationships  relationship_adjacency, relationship_mirror,
                        relationship_repeat, relationship_cell_colour
                        + prim_adjacency, prim_mirror, prim_repeat,
                          prim_cell_colour  (typed compression slots)

Layer 3  Structural     prior_grid, prior_extent, prior_density,
                        prior_spatial_organisation, prior_structure,
                        prior_grid_transform, prior_transformation,
                        + spatial band priors (row/col/diagonal/corners)

Layer 4  Colour         prior_colour_background (v=0), prior_colour_low (v=1-3),
                        prior_colour_mid (v=4-6), prior_colour_high (v=7-9),
                        prior_colour_uniform, prior_colour_group

Layer 5  Semantic       object_hfn (shape + colour + count)
                        scene_hfn (background + objects)
                        rule_hfn (transform + count-based + colour-map)

Layer 6  Encoder        encoder_hfn — children are structural priors;
                        encodes HOW to read a grid through its structure

Bridge   prior_pixel_colour — a pixel has both a position AND a colour identity
                        Children: prior_pixel (Layer 0) + prior_colour_group (Layer 4)
```

**Node counts (colour encoding, value/9.0):**

| Grid | D | Prior nodes | Notes |
|------|---|-------------|-------|
| 3×3  | 9  | 79  | Abstracts dominate |
| 5×5  | 25 | 101 | +25 cell primitives, +5 row/col |
| 10×10 | 100 | 184 | +100 cell primitives, +10 row/col |

Primitive nodes grow O(D); all abstract priors (colour, structural, semantic, encoder) are grid-size-independent.

---

## Value Encoding

ARC uses 10 colour values (0–9). Rather than binary {0, 1}:

```python
vec[i] = cell_value / 9.0   # each cell in [0.0, 1.0]
```

This preserves colour identity in the observation vector. Distances now reflect both spatial position AND colour similarity. D is unchanged (still rows × cols).

**Why it matters:**
Binary encoding achieved 42% prior coverage on 10×10 grids. Value encoding achieves 90%, with 0 absorbed priors. The colour priors do real work — `prior_colour_background` alone explains ~23% of 10×10 observations.

---

## Experiments

| Script | What it tests | Key result |
|--------|---------------|------------|
| `experiment_arc_colour.py` | 3×3, colour encoding, full world model | **100% coverage**, 0 unexplained, 0 absorbed |
| `experiment_arc_world_model.py` | 3×3, binary encoding, full world model | 85% coverage, 9% unexplained |
| `experiment_arc_10x10.py` | 10×10, colour encoding, full world model | **90% coverage**, 5% unexplained, 0 absorbed |
| `experiment_arc_prior_forest.py` | 3×3, structural priors only | Baseline prior coverage |
| `experiment_arc_observer.py` | 3×3, no priors, pure observation | What emerges without prior knowledge |

**Run any experiment from the project root:**
```bash
python3 -m hpm_fractal_node.experiment_arc_colour
python3 -m hpm_fractal_node.experiment_arc_10x10
```

---

## Test Coverage

Tests live in `tests/hpm_fractal_node/`. Run with:
```bash
python3 -m pytest tests/hpm_fractal_node/ -v
```

| Test file | What it covers |
|-----------|---------------|
| `test_hfn_structure.py` | HFN identity, DAG operations, recombine, fractal uniformity |
| `test_forest.py` | Forest registration, deregistration, node lookup |
| `test_observer.py` | Weight updates, absorption, compression, protected IDs |
| `test_arc_representation.py` | ARC-specific encoding, prior structure |
| `test_scaling.py` | Scaling invariants (see below) |

### What the scaling tests reveal

**`test_node_count_scales_with_grid_size`**
Confirms primitive nodes scale exactly as D (one per cell), while abstract priors are constant. This is the intended asymmetry: the vocabulary grows with the domain, but the concepts don't.

**`test_log_prob_diagonal_fast_path`**
Every prior node must have `_sigma_diag` cached at construction. If any node is created with a non-diagonal covariance, it will fall back to O(D³) Cholesky and this test will catch it.

**`test_log_prob_throughput_scales_linearly`**
Verifies the O(D) fast path holds across grid sizes. At D=100, each `log_prob` call takes ~0.0075ms (vs. the O(D³) Cholesky which was the bottleneck causing 10×10 experiments to time out).

**`test_world_model_build_time`**
Build time limits: 3×3 < 1s, 5×5 < 2s, 10×10 < 5s. Regression guard for accidentally expensive prior construction.

**`test_observer_processes_observations`**
Verifies that after 20 random observations, no protected prior was absorbed or removed — the protected_ids mechanism holds at every grid size.

**`test_abstract_priors_fire_at_all_scales`**
Colour and density priors explain uniform-colour observations at every grid size. Confirms structural knowledge transfers across scales.

---

## Key Insights

**1. Prior knowledge is a prerequisite, not an enhancement**
Without the colour layer, 33% of 10×10 observations were unexplained. Adding `prior_colour_*` nodes dropped unexplained to 5%. This parallels HPM's claim that learners don't start as blank slates — relevant domain priors are necessary for learning to proceed efficiently.

**2. The perceptual chain matters**
The perception layer (signal → pixel → cell → field) encodes *how* the system perceives before *what* it perceives. Without it, the world model knows about grids but not why grids are coherent. The field concept (adjacent cells with consistent structure form a bounded whole) is the foundation of spatial reasoning.

**3. Pixel vs cell vs colour is not a trivial distinction**
Three concepts that are easy to conflate:
- `prior_pixel_colour`: a *raw perceptual measurement* carries both position and colour (Layer 0 — before the grid frame exists)
- `relationship_cell_colour`: a *cell within a known frame* is bound to a colour value (Layer 2 — structural binding)
- `prim_cell_colour`: the typed slot the Observer fills with specific (position, colour) instances

Conflating these produced gaps in coverage. Keeping them distinct allows the Observer to form specific bindings via compression without pre-enumerating all (r, c, v) combinations.

**4. Shared DAG nodes are essential**
`object_shape` references `primitive_region` directly — not a copy. `object_colour` references `prior_pixel_colour`. When the Observer updates `primitive_region`, that change is immediately reflected in `object_hfn` because they share the same node. This is HPM's "no redundant encoding" principle in practice.

**5. The Forest is stable under observation**
In all experiments, 0 protected priors were absorbed. The world model is invariant — the Observer learns above it, not through it.

---

## Known Issues and Open Questions

**5% unexplained at 10×10**
These 18 observations have colour patterns specific enough that no broad prior fits below `tau`. They are candidates for `prim_cell_colour` compressions — the Observer needs to see the same (position, colour) co-occurrence multiple times before forming the compression. Increasing passes or reducing `compression_cooccurrence_threshold` should close this gap.

**Tau is fixed across all layers**
The same `tau = baseline + 5.0` applies to a perception prior (very broad, σ=4) and a specific primitive (σ=1). A prior calibrated for `prior_signal` is far too loose for `primitive_cell_23`. Fractal tau scaling — `tau_layer_n = tau_0 × r^n` — would give each layer an appropriate tolerance. This will matter more as the hierarchy deepens.

**`vec_to_grid` display uses binary threshold**
The experiment display function (`> 0.5`) was written for binary encoding. With value encoding, most colour values are < 0.5, so grids display as empty even when the node is correctly matching a sparse colour grid. This is a display-only issue — the `log_prob` computation is correct.

**2 registry entries not in forest**
`full_registry` contains 2 more entries than the Forest has active nodes. These are nodes that are referenced as children (shared DAG) but not registered as top-level Forest members. They are reachable, but not directly visible to the Observer. This is intentional but should be documented explicitly in `arc_world_model.py`.

**Observer timing at D=100**
Before the O(D³)→O(D) log_prob optimisation, 10×10 experiments timed out at 240s. With the diagonal sigma cache (`__post_init__` in `hfn.py`), `log_prob` runs at ~0.0075ms/call. Full two-pass experiment now completes in ~60–70s. Further speedup would require batching log_prob across all nodes simultaneously (vectorised over node population).

---

## Architecture Diagram

```
Observation vector x ∈ ℝᴰ  (D = rows × cols, value/9.0 encoding)
         │
         ▼
    ┌─────────────────────────────────┐
    │         Observer                │
    │  ┌──────────────────────────┐   │
    │  │  1. Score all nodes      │   │
    │  │  2. Update weights       │   │
    │  │  3. Create if residual   │   │
    │  │  4. Compress if cooccur  │   │
    │  │  5. Absorb if dominated  │   │
    │  └──────────────────────────┘   │
    │          ↕ reads/writes         │
    └─────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────┐
    │   Forest (world model)          │
    │                                 │
    │  Protected priors (invariant)   │
    │  ┌──────┐ ┌──────┐ ┌──────┐   │
    │  │ L0   │ │ L1   │ │ L2   │   │
    │  │Perc  │ │Prim  │ │Rel   │   │
    │  └──────┘ └──────┘ └──────┘   │
    │  ┌──────┐ ┌──────┐ ┌──────┐   │
    │  │ L3   │ │ L4   │ │ L5   │   │
    │  │Struc │ │Colour│ │Sem   │   │
    │  └──────┘ └──────┘ └──────┘   │
    │  ┌──────┐                      │
    │  │ L6   │  Encoder             │
    │  └──────┘                      │
    │                                 │
    │  Learned nodes (dynamic)        │
    │  ┌────┐ ┌────┐ ┌────┐          │
    │  │new │ │comp│ │ …  │          │
    │  └────┘ └────┘ └────┘          │
    └─────────────────────────────────┘
```

---

## File Reference

| File | Role |
|------|------|
| `hfn.py` | HFN dataclass: Gaussian identity + DAG body + O(D) log_prob |
| `forest.py` | Forest: registration, deregistration, node lookup |
| `observer.py` | Observer: all dynamic learning state and mechanics |
| `arc_world_model.py` | Assembly entry point: builds all 7 layers into one Forest |
| `arc_perception_priors.py` | Layer 0: perceptual chain |
| `arc_primitives.py` | Layer 1: atomic spatial vocabulary |
| `arc_relationships.py` | Layer 2: relational vocabulary + prim_* compressions |
| `arc_prior_forest.py` | Layer 3: structural priors (density, spatial, transform) |
| `arc_colour_priors.py` | Layer 4: value-identity priors |
| `arc_object_scene_priors.py` | Layer 5: object, scene, rule priors |
| `arc_encoder_hfn.py` | Layer 6: encoder as structural prior |

---

## Fractal Convergence Analysis

### IFS Hypothesis

The Observer's `recombine(A, B)` operation creates a node whose `μ = mean(A.μ, B.μ)` — a contracting affine map in pattern space. This is how fractal attractors are constructed via Iterated Function Systems (IFS). If the Observer applies this consistently, the set of learned nodes should converge toward a fractal attractor whose dimension signals structural coherence.

**Tool:** `hfn.fractal.population_dimension(nodes)` — box-counting dimension of the μ-space point cloud.

### Experiment: World Model vs No Priors (3×3, colour encoding, 8 passes)

| Pass | World model (learned dim) | No priors (all dim) | WM nodes | NP nodes |
|------|--------------------------|---------------------|----------|----------|
| 1 | 0.049 | 0.326 | 22 | 14 |
| 2 | 0.125 | 0.406 | 23 | 16 |
| 3 | 0.044 | 0.508 | 23 | 20 |
| 4 | 0.022 | 0.271 | 19 | 34 |
| 5 | 0.117 | 0.179 | 20 | 54 |
| 6 | 0.039 | 0.109 | 25 | 48 |
| 7 | 0.079 | 0.175 | 21 | 36 |
| 8 | 0.035 | 0.112 | 17 | 43 |

### Key Findings

**1. Priors pre-seed the IFS attractor**
Without priors, the Observer initially spreads nodes across observation space (dim 0.33–0.51) then gradually finds clusters (dim drops to ~0.11). Structure eventually emerges but takes many passes and the node population grows unbounded (54 nodes).

With the world model, learned nodes immediately cluster near prior attractors (dim 0.02–0.13) and the population stays bounded (~20 nodes). The priors act as pre-placed contracting maps — the IFS attractor is seeded before any observation.

**2. Dimension tracks world model adequacy**
Low, stable learned dimension = well-specified world model (Observer filling gaps near known attractors).
High, unstable dimension = under-specified world model (Observer building structure from scratch).

This gives a quantitative diagnostic: if learned dimension approaches or exceeds prior dimension, the world model is inadequate for the domain.

**3. Prior knowledge reduces learning dimensionality**
The fractal analysis makes visible what coverage statistics cannot: prior knowledge doesn't just explain more observations — it changes the *structure* of what gets learned. Without priors, learned patterns are high-dimensional and unstable. With priors, learning is low-dimensional and convergent.

This is quantitative evidence for the HPM claim that prior knowledge is a structural prerequisite, not merely an optimisation.

### Run the diagnostic
```bash
python3 -m hpm_fractal_node.experiments.experiment_fractal_diagnostic
```

---

## Diagnostic #2: Self-Similarity Score

**Tool:** `hfn.fractal.self_similarity_score(nodes)` — coefficient of variation of log-count differences across box-counting scales. Lower = more self-similar; 0.0 = perfect power-law scaling (IFS attractor).

### Experiment: World Model vs No Priors (3×3, colour encoding, 5 passes)

| Pass | WM learned SS | WM all SS | NP all SS | WM nodes | NP nodes |
|------|--------------|-----------|-----------|----------|----------|
| 1 | 2.47 | 0.77 | 1.47 | 92 | 17 |
| 2 | 0.88 | 0.65 | 1.69 | 97 | 16 |
| 3 | 1.07 | 0.78 | 1.61 | 93 | 19 |
| 4 | 0.88 | 0.56 | 0.88 | 92 | 18 |
| 5 | 1.62 | 0.54 | 1.16 | 91 | 15 |

### Key Findings

**1. The prior structure is intrinsically self-similar (WM all SS ~0.54–0.78)**
The full world-model node population scores consistently lower than the no-priors condition. The 7-layer hierarchy (perception → primitives → relationships → structure → colour → semantic → encoder) creates natural clustering at multiple scales — this is what IFS attractors look like geometrically.

**2. Learned nodes alone are not yet self-similar (WM learned SS ~0.88–2.47)**
Learned nodes are filling gaps near priors but haven't formed their own self-similar sub-clusters at 5 passes. More passes are needed for convergence.

**3. No-priors is more scattered overall (NP all SS ~0.88–1.69)**
Even with 15–19 nodes, the no-priors population is less self-similar than the world-model population, confirming that priors encode geometric regularity the Observer would otherwise have to discover from scratch.

```bash
python3 -m hpm_fractal_node.experiments.experiment_fractal_self_similarity
```

---

## Diagnostic #3: Hausdorff Distance

**Tool:** `hfn.fractal.hausdorff_distance(nodes_a, nodes_b)` — worst-case nearest-neighbour distance between two node populations in μ-space. Tracks how close learned nodes are to the prior attractor, and how stable the no-priors node population is across passes.

### Experiment: World Model vs No Priors (3×3, colour encoding, 8 passes)

| Pass | WM Hausdorff | WM learned N | NP shift | NP nodes |
|------|-------------|-------------|----------|----------|
| 1 | 1.476 | 11 | — | 16 |
| 2 | 1.452 | 16 | 1.032 | 18 |
| 3 | 1.419 | 13 | 1.113 | 15 |
| 4 | 1.595 | 15 | 1.087 | 19 |
| 5 | 1.469 | 9 | 0.863 | 10 |
| 6 | 1.417 | 17 | 0.944 | 10 |
| 7 | 1.550 | 12 | 0.818 | 15 |
| 8 | 1.522 | 17 | 0.777 | 10 |

### Key Findings

**1. WM Hausdorff is stable, not converging (~1.4–1.6)**
The Hausdorff is dominated by 3 outlier learned nodes (dist=0.34, 0.41, 0.46) near structural priors (`prior_connectivity`, `relationship_repeat`, `relationship_adjacency`) that are too broad to claim those observations. Most learned nodes (14 of 17) are within dist=0.14 of a prior.

**2. Hausdorff identifies gaps in the prior vocabulary**
The 3 high-distance outliers consistently cluster near the same priors — those priors have covariance too large for their domain. This is a direct diagnostic for which priors need tighter sigma or more specific sub-priors.

**3. No-priors population is volatile (NP shift ~0.78–1.11)**
The no-priors Observer rebuilds different clusters each pass, with the population shifting ~1.0 unit in μ-space. A slight decrease over passes (1.11 → 0.78) suggests slow stabilisation — but far more volatile than the world-model condition.

**4. Hausdorff as a world-model improvement tool**
Running the Hausdorff diagnostic after adding new priors to the world model and measuring whether the WM Hausdorff decreases gives a quantitative signal of whether the new priors are filling the right gaps.

```bash
python3 -m hpm_fractal_node.experiments.experiment_fractal_hausdorff
```
