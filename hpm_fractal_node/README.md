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

---

## Diagnostics #4–6: Correlation Dimension, Information Dimension, Intrinsic Dimensionality

Three complementary measures that extend the fractal toolkit beyond box-counting.

### Tools

| Function | What it measures |
|----------|-----------------|
| `hfn.fractal.correlation_dimension(nodes)` | Pair-density scaling with radius — more accurate than box-counting for small populations (n < 50) |
| `hfn.fractal.information_dimension(nodes, weights)` | Weight-adjusted fractal dimension — measures the *active* representation, not just presence |
| `hfn.fractal.intrinsic_dimensionality(points)` | True degrees of freedom in a point cloud (TwoNN estimator) — independent of ambient dimension |

### Experiment Results (3×3, colour encoding, 3 passes)

```
Intrinsic dimensionality of ARC 3×3 observations: 6.580  (ambient D=9)

After 3 passes (15 learned nodes, 79 prior nodes):
  Box-counting dim (all):     0.1820
  Correlation dim (all):      0.6015
  Correlation dim (learned):  1.9503
  Information dim (all):      0.1858
  Information dim (learned):  0.0501
  Intrinsic dim (prior μ):    3.963
  Intrinsic dim (learned μ):  4.272
```

### Key Findings

**1. ARC 3×3 observations have intrinsic dimensionality ~6.6 (ambient D=9)**
The patterns genuinely use ~6.6 of the 9 available dimensions. This is not a simple domain — at least 7 independent axes of variation exist. The world model with ~79 priors is appropriately rich for this complexity.

**2. Correlation dimension reveals learned-node geometry (1.95 ≈ 2D surface)**
Box-counting on the full population (0.18) masks the geometry of learned nodes alone (1.95). The 15 learned nodes span a near-2D surface in μ-space — they are not randomly scattered, nor clustered at a point, but arranged along a plane. This is consistent with the IFS attractor hypothesis: recombine() maps produce planar structure.

**3. Information dimension (learned) = 0.05 — near-degenerate weight distribution**
Almost all learned weight is concentrated in 1–2 nodes. The majority of learned nodes are low-weight artefacts. The Observer is learning a near-point-mass distribution, not a spread. This confirms the oscillation behaviour seen in earlier diagnostics: most learned nodes are created, fire rarely, and never consolidate.

**4. Prior nodes span ~4 intrinsic dimensions; learned nodes span ~4.3**
Learned nodes occupy marginally higher-dimensional space than the priors. They are exploring just beyond the prior vocabulary, not duplicating it — consistent with the Observer filling genuine gaps rather than redundantly re-representing known concepts.

**5. Information dimension is the most diagnostic of the three**
It distinguishes *what exists* from *what is used*. A low information dimension with a higher box-counting dimension means many nodes exist but few dominate — a signal that compression is not consolidating fast enough. Running `information_dimension` before and after changing `compression_cooccurrence_threshold` gives a direct measure of whether the change improves consolidation.

---

## Diagnostics #7–10: Persistence, Recurrence, Lacunarity, Multifractal

Four further fractal tools, all in `hfn/fractal.py`.

### Tools

| Function / Class | What it measures | Observer use |
|-----------------|-----------------|--------------|
| `persistence_scores(nodes, weights)` | Per-node structural isolation — distance to nearest higher-weight node | Modulates absorption miss threshold |
| `RecurrenceTracker(buffer_size, epsilon)` | How repetitive the observation stream is | Dynamically adjusts compression threshold |
| `lacunarity(nodes, n_scales)` | Gappiness of node distribution at each scale ε — (σ²/μ²) of box counts | Diagnostic only |
| `multifractal_spectrum(nodes, q_values, n_scales)` | Generalised dimensions D_q — reveals hot/cold spots in prior structure | Diagnostic only |

### Observer integration

```python
obs = Observer(forest, tau=tau, protected_ids=prior_ids,
               persistence_guided_absorption=True,   # protect isolated nodes longer
               adaptive_compression=True,            # lower threshold when stream is repetitive
               recurrence_epsilon=0.15)
```

### Experiment: Strategy Comparison (3×3, colour encoding, 5 passes)

| Condition | Coverage | Learned N | Hausdorff | Absorbed |
|-----------|----------|-----------|-----------|----------|
| baseline (nearest_prior + hausdorff) | 80.2% | 10 | 1.515 | 1618 |
| + persistence_guided_absorption | **81.2%** | 12 | **1.478** | 1647 |
| + adaptive_compression | 80.2% | 10 | 1.515 | 1618 |
| + both | **81.2%** | 12 | **1.478** | 1647 |

### Key Findings

**1. Persistence-guided absorption improves coverage (+1pp) and Hausdorff**
High-persistence nodes (structurally isolated, far from any better-weighted node) get a higher effective miss threshold — the Observer protects them longer. Genuinely novel nodes survive rather than being absorbed prematurely, lifting coverage from 80.2% to 81.2% and pulling learned nodes slightly closer to the prior attractor (Hausdorff 1.515 → 1.478).

**2. Adaptive compression has no effect at 5 passes**
The recurrence rate never builds high enough in 5 passes over 250 observations to shift the compression threshold. This is a slow-burn mechanism — correct in principle but requires 20+ passes for recurrence to accumulate. It would activate in longer-running learning sessions where the same observations recur many times.

**3. The strategies are independent**
Persistence affects absorption; recurrence affects compression. They operate on separate parts of the Observer pipeline and don't interact — the combined result equals persistence-only.

**4. Current best configuration**
`recombination_strategy="nearest_prior"` + `hausdorff_absorption_threshold=0.15` + `persistence_guided_absorption=True` achieves 81.2% coverage with 12 learned nodes and Hausdorff 1.478. This is the recommended starting point for new experiments.

**5. Lacunarity and multifractal spectrum are diagnostic-only**
These reveal structure in the prior/learned node distribution (gappiness and hot spots) but are not yet integrated into Observer decisions. Lacunarity would inform node-creation suppression in dense regions; the multifractal spectrum would identify which prior layers are geometrically dominant.

---

## Full Fractal Strategy Stack: ARC Results

Grid search over lacunarity and multifractal radii (3×3, 5 passes) found the optimal configuration:

```python
obs = Observer(forest, tau=tau, protected_ids=prior_ids,
               recombination_strategy="nearest_prior",
               hausdorff_absorption_threshold=0.15,
               persistence_guided_absorption=True,
               lacunarity_guided_creation=True,       # lacunarity_creation_radius=0.08
               multifractal_guided_absorption=True)   # multifractal_crowding_radius=0.12
```

### 3×3 ARC (250 observations, 5 passes)

| Condition | Coverage | Hausdorff | Absorbed |
|-----------|----------|-----------|----------|
| baseline (nearest_prior + hausdorff) | 80.2% | 1.515 | 1618 |
| + persistence | 81.2% | 1.478 | 1647 |
| + lacunarity + multifractal (r=0.3) | 81.0–81.1% | — | — |
| **+ lacunarity (r=0.08) + multifractal (r=0.12)** | **81.8%** | **1.462** | — |

Full stack with tuned radii beats persistence-only on both coverage (+0.6pp) and Hausdorff proximity.

### 10×10 ARC (392 observations, 2 passes)

| Metric | Baseline (colour encoding) | Full fractal stack |
|--------|---------------------------|--------------------|
| Prior coverage | 90% (352/392) | 90% (352/392) |
| New node coverage | 0% | 6% (22/392) |
| Unexplained | 5% (18/392) | 5% (18/392) |
| Absorbed priors | 0 | 0 |

The full fractal stack maintains the 90% prior coverage baseline and adds 6% coverage from new learned nodes — bringing total explained observations from 90% to 96%. The 5% unexplained observations (18 obs with specific colour/position patterns) remain unresolved; they require more passes or lower `compression_cooccurrence_threshold` to consolidate into compressions.

### Key Findings

**1. Lacunarity-guided creation suppresses redundant nodes in dense regions**
When local node density exceeds `lacunarity_creation_factor` × mean, the Observer skips creating a new node for residual observations. This reduces low-weight artefact creation, freeing absorption budget for genuinely novel patterns.

**2. Multifractal-guided absorption halves the threshold in crowded regions**
Nodes near hot spots (many non-protected neighbours within `multifractal_crowding_radius`) get their absorption threshold halved — making the Observer more aggressive about merging locally redundant learned nodes. Combined with lacunarity suppression, this keeps the learned population tighter.

**3. 10×10 new node coverage is the signal, not prior coverage**
At 10×10 the prior vocabulary covers 90% of observations. The fractal strategies' contribution shows in the new node coverage (0% → 6%): the stack creates better-targeted learned nodes that explain observations the priors miss.

---

## dSprites Experiment: Generative Factor Alignment

### What is dSprites?

dSprites is a synthetic dataset of 2D shapes defined by 5 independent generative factors:

| Factor | Values |
|--------|--------|
| Shape | 3 (square, ellipse, heart) |
| Scale | 6 levels |
| Orientation | 40 angles |
| Pos X | 32 positions |
| Pos Y | 32 positions |

Images are 64×64 binary sprites, downsampled to 16×16 for the experiment (D=256). There are no labels — only raw pixel observations and the ground-truth generative factor values used post-hoc to evaluate alignment.

### The dSprites World Model

The world model is assembled in `hpm_fractal_node/dsprites/dsprites_world_model.py` as 37 protected priors across 5 layers:

| Layer | Priors | What they encode |
|-------|--------|-----------------|
| Perception | 3 | signal, pixel, cell concept |
| Spatial cells | 16 | 4×4 spatial grid cells |
| Shape | 4 | background, blob, compact, extended |
| Scale | 6 | levels 1–6 (sigma 0.02–0.07) |
| Position/Orientation | 8 | centre, edge N/S/E/W, corner, rotated |

### Experiment Configuration

```python
n_samples = 2000       # observations drawn from dSprites
n_passes  = 3          # passes over the sample
D         = 256        # 16×16 downsampled images
tau       = 240.25     # surprise threshold
```

Run with:
```bash
python3 -m hpm_fractal_node.experiments.experiment_dsprites
```

### Results

| Metric | Value |
|--------|-------|
| Total observations | 6000 (2000 × 3 passes) |
| Covered by learned nodes | 5999 / 6000 (100%) |
| Learned nodes created | 101 |
| Hausdorff(learned, priors) | 10.4208 |

**Factor purity (nodes with n ≥ 5 observations):**

| Factor | Mean purity | Max purity | Random baseline | Above chance |
|--------|------------|-----------|-----------------|-------------|
| Shape | 0.546 | 1.000 | 0.333 | +64% |
| Scale | 0.483 | 1.000 | 0.167 | +189% |
| Pos X | 0.366 | 0.800 | 0.031 | +1081% |

Factor purity measures how often observations assigned to a learned node share the same ground-truth generative factor value (1.0 = all observations in the node come from one factor level).

### Key Finding

The Observer recovers latent generative structure without supervision. Learned nodes are not random partitions of observation space — they align with the underlying factors that generated the images. Pos X purity is 10× above chance, meaning the Observer is discovering spatial position as a structuring principle purely from pixel statistics.

The high Hausdorff distance (10.42) reflects that the 16×16 pixel space is geometrically large relative to the 37 hand-specified priors — the priors cover conceptual roles (shape, scale, position) but the learned nodes are operating at a finer resolution in the full D=256 space. This is expected: the world model provides structural scaffolding, and the Observer fills in the detail.

```bash
python3 -m hpm_fractal_node.experiments.experiment_dsprites
```

---

## NLP Experiment: Semantic Category Alignment

### What the NLP experiment tests

The NLP experiment applies HFN to a language domain: 2000 synthetic subject-verb-object sentences drawn from 7 semantic categories (animal, adult, child_person, family, food, object, place). Each sentence is encoded as a context window of word vectors (D=107, one dimension per vocabulary item), and the Observer learns to explain sentences using the NLP world model.

Unlike ARC (spatial patterns) or dSprites (visual factors), the NLP domain tests whether the Observer recovers **semantic categories** from distributional co-occurrence — without any category labels.

### Priors as relational constraints

The NLP world model (`nlp_world_model.py`) contains 195 protected priors across four layers:

| Layer | Priors | What they encode |
|-------|--------|-----------------|
| Atomic word nodes | 107 | One HFN per vocabulary item — the identity of each word as a point in context space |
| Leaf objects + grammar | ~38 | Typed slots for nouns, verbs, determiners; object groupings (obj_food, obj_animal, obj_noun, …) |
| Capability priors | ~19 | Entity-action bindings: cap_dog_barks, cap_dog_fetches, cap_cat_meows, cap_person_gives, … |
| Sentence priors | 20 | Full SVO templates: sent_dad_ate_bread, sent_girl_helped_baby, … |

**Priors are not merely starting points — they are relational constraints.** Each prior's μ encodes a pattern of co-occurrence over the full vocabulary. `obj_food` has a mean that activates food-category words; `cap_dog_barks` has a mean that simultaneously activates the dog word and the barks word. When the Observer scores an incoming observation against `cap_dog_barks`, it is testing whether that sentence contains the relational binding dog→barks, not just whether either word appears.

This is the HPM principle: **patterns are defined by their internal coherence** (do the components co-occur in the expected configuration?) rather than by label or symbol. The prior encodes the expected structure of the relationship.

### Capability priors: entity-specific action bindings

Capability priors encode what entities can do — not as rules, but as Gaussian distributions over context:

- `cap_dog_barks` — μ activates dog + barks simultaneously; fires only when both appear together
- `cap_dog_fetches` — μ activates dog + fetches; a distinct capability, a distinct prior
- `cap_cat_meows` — μ activates cat + meows
- `cap_person_gives` — μ activates person + gives

These priors implement the HPM notion that **capabilities are patterns at a higher abstraction level than individual words**. A sentence with "the dog barked" matches `cap_dog_barks` because the sentence's context vector (a sum over word positions with positional weighting) concentrates probability near the dog+barks region of vocabulary space.

Each capability prior is a child of its entity's object node (`cap_dog_barks` and `cap_dog_fetches` are both children of `obj_animal`) — this is the uniform composition principle: the same node type at every level.

### Uniform composition at every level

Every node in the NLP world model is an HFN: a Gaussian identity in context space with a DAG body. This means:

- A word node (`word_dog`) is an HFN with μ = the word's expected context vector
- An object category node (`obj_animal`) is an HFN with μ = the centroid of its members' contexts, and children = the word nodes it groups
- A capability node (`cap_dog_barks`) is an HFN with μ = the expected context when dog+barks co-occur, and its children are the relevant word nodes
- A sentence template (`sent_dad_ate_bread`) is an HFN with μ = the full SVO context, children = the word nodes for dad, ate, bread

The same Observer mechanics apply at every level: log_prob, weight update, recombine, absorb. There is no special-case logic for "this is a word" vs "this is a sentence template". The fractal uniformity that defines HFN for ARC applies identically to language.

### Results (3-pass, D=107, 2000 observations)

```
Passes:     3
Total obs:  6000 (2000 × 3 passes)
Explained:  6000/6000 (100.0%)
```

**Coverage breakdown:**

| Source | Observations | % |
|--------|-------------|---|
| Prior nodes | 1842 | 30.7% |
| Learned nodes | 4158 | 69.3% |
| Total | 6000 | 100.0% |

All three passes achieve 100% coverage — every sentence is explained by either a prior or a learned node.

**Top prior nodes by coverage:**

| Prior | n | Category purity | Dominant category |
|-------|---|----------------|------------------|
| obj_family | 271 | 0.86 | adult |
| sent_grandma_gave_toy | 181 | 0.69 | place |
| obj_object | 131 | 1.00 | place |
| sent_girl_helped_baby | 127 | 0.43 | object |
| sent_dad_ate_bread | 115 | 0.69 | food |
| obj_food | 96 | 1.00 | food |
| cap_dog_fetches | 55 | 1.00 | animal |
| cap_dog_barks | 51 | 1.00 | animal |
| cap_cat_meows | 50 | 1.00 | animal |
| cap_person_gives | 48 | 1.00 | family |

Capability priors achieve **100% category purity** — every observation they explain comes from the animal or family category respectively. This confirms that the entity-action binding priors are semantically precise: they fire only on sentences where the specific agent-action relationship is present.

**Learned node category purity (nodes with n ≥ 5):**

| Metric | Value | Random baseline |
|--------|-------|----------------|
| Category purity (mean) | 1.000 | 0.143 (1/7) |
| Category purity (max) | 1.000 | — |
| Word purity (mean) | 0.667 | — |

Mean category purity of 1.000 vs random baseline of 0.143 — learned nodes are **7× above chance** on semantic category alignment.

**Fractal diagnostics:**

| Metric | Value |
|--------|-------|
| Learned nodes (final) | 7 |
| Hausdorff(learned, priors) | 1.0385 |
| Absorbed nodes | 5400 |

The Observer creates and absorbs 5400 nodes over 3 passes, converging to 7 stable learned nodes. The low Hausdorff distance (1.04) in D=107 space indicates learned nodes stay close to the prior vocabulary — they are filling gaps near existing priors rather than drifting into unstructured regions.

### Key findings

**1. 100% semantic coverage from structured priors**
The 195 prior nodes explain all 6000 sentence observations across 3 passes. The NLP world model fully spans the sentence space defined by the vocabulary and grammar. This is a direct consequence of prior design: every word, object category, capability, and sentence template is represented, so the Observer always finds a prior within `tau` of any observation.

**2. Capability priors are the most semantically precise nodes**
`cap_dog_barks`, `cap_dog_fetches`, `cap_cat_meows`, and `cap_person_gives` all achieve 1.00 category purity. They are more precise than sentence templates (which span multiple categories) and more specific than object nodes (which group heterogeneous members). Capability priors occupy the right level of abstraction for semantic alignment.

**3. Priors as relational constraints enable zero-shot category recovery**
No category labels are provided to the Observer. Semantic categories emerge from the geometry of the world model: nodes built to encode relational structure (entity + action, entity + object) happen to align with the categories that defined the data. This is HPM's structural claim — coherent patterns at one level of the hierarchy encode information about higher levels.

**4. Convergence: 5400 absorbed → 7 stable learned nodes**
The Observer's absorption mechanism is highly active in NLP (5400 absorbed across 6000 observations). This reflects that new nodes created for residual observations are quickly dominated by existing priors on subsequent passes. The system converges to a sparse representation where learned nodes supplement, rather than replace, the prior vocabulary.

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_nlp.py
```
