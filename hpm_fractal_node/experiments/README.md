# HPM Hierarchical Abstraction Experiments (SP54-SP56+)

## Experiment 44: Unified Perception-Action Schema Learning (SP54)
Demonstrates that a single agent can synthesise executable Python programs from
input-output examples using goal-directed BFS planning over structured concept scaffolds.

### Tasks Solved
| Task | Description | Method | Depth |
|------|-------------|--------|-------|
| A | Add one (scalar) | Stochastic search | ~11–40 iters |
| B | Map add one | BFS (MAP scaffold) | 6 |
| C | Map double | BFS (MAP scaffold + OP_MUL2) | 6 |
| D | Filter positive | BFS (FILTER scaffold) | 6 |

### Key Design
- **EmpiricalOracle**: 20D state vector computed by actually executing synthesised code
- **ASTRenderer**: Context-aware code generation (`val` vs `x` depending on loop depth)
- **Goal-type detection**: MAP vs FILTER goals use separate operator sets, keeping BFS tractable
- **Schema transfer**: MAP structure from Task B reused for Task C (double) and beyond

See [README_unified_perception_action.md](README_unified_perception_action.md) for full analysis.

### Running
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_unified_perception_action.py
```

---

## Experiment 45: Induced Schema Library (SP61)
Demonstrates the full HPM loop: Perceive → Execute → Verify → Compress → Transfer → Meta-Abstract.
Schemas are NOT hardcoded — they emerge from solved tasks, stored as Polygraph macro nodes, and
reused on harder tasks via macro decomposition search.

### Tasks Solved
| Phase | Task | Method | Depth |
|-------|------|--------|-------|
| 1 | add_1, mul_2, sub_1 (scalar) | Direct enumeration over percept ops | 1 |
| 2 | MAP+1 (list) | Scaffold-restricted BFS | 6 |
| 3 | MAP*2 (list) | Macro decomposition (percept+1 → OP_MUL2) | **2** |
| 4 | FILTER_pos (list) | Macro decomposition (percept+1 → COND_IS_POSITIVE) | 2 |
| 5 | MAP+2 (list) | Macro decomposition (percept+1 → percept+2) | 2 |
| 6 | Meta-schema | Common prefix across MAP/FILTER macros | L3 node |

### Success Conditions
- Phase 3 depth ≤ 2: MAP macro reuse demonstrated
- Phase 5 depth ≤ 3: Compound macro composition demonstrated
- Phase 6 L3 node exists: Meta-schema induction demonstrated

See [README_induced_schema_library.md](README_induced_schema_library.md) for full analysis.

### Running
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_induced_schema_library.py
```

---

## Experiment 46: Generative Forward Model (SP62)
Demonstrates **HPM Level 4: Generative Rules / Mental Simulation**. A `StateTransitionModel`
learns per-node state deltas from solved paths; imaginative BFS navigates over predicted
state vectors with ZERO oracle calls during search. Oracle called only at the end to verify.

### Tasks Solved
| Phase | Task | Method | Oracle calls |
|-------|------|--------|--------------|
| 1-4 | add_1, MAP+1, MAP*2, FILTER_pos (+ transition recording) | Same as SP61 | Normal |
| 5 | MAP*2 (held-out unseen inputs) | Imaginative BFS | 0 during search, 2 at verification |
| 6 | Forward model accuracy | Structural dim MAE on STRUCT_DIMS | MAE = 0.000 |

### Success Conditions
- Phase 5: oracle_calls_during_search == 0 AND solution correct → L4 Mental Simulation demonstrated
- Phase 6: mean_prediction_error < 0.15 on STRUCT_DIMS → Forward model accurate

See [README_generative_forward_model.md](README_generative_forward_model.md) for full analysis.

### Running
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_generative_forward_model.py
```

---

## Experiment 47: Meta-Strategy Controller (SP63)
Demonstrates **HPM Level 5: Meta-patterns / Metacognition**. The agent observes its own
solve behaviour across tasks, learns which strategies succeed in which contexts
(goal_type × n_macros_bucket), and adapts strategy selection accordingly. Oracle call
overhead is reduced to ≤80% of the fixed-order baseline.

### Tasks / Phases
| Phase | Task | Method |
|-------|------|--------|
| 1 | add_1, mul_2, sub_1 (scalar) | Enumeration + strategy recording |
| 2 | MAP+1 (list) | BFS + strategy recording |
| 3 | MAP*2 (list) | Macro decomposition + recording |
| 4 | FILTER_pos (list) | Macro decomposition + recording |
| 5 | 6 novel tasks | Meta-directed strategy selection |
| 6 | Same 6 tasks | Fixed-order baseline comparison |
| 7 | — | Meta-pattern report |

### Success Conditions
- Phase 5: strategy_match ≥ 4/6 tasks → Meta-controller selects learned strategy
- Phase 6: meta_oracle_calls ≤ 0.80 × baseline → Oracle efficiency demonstrated
- Phase 7: ≥ 3 distinct meta-patterns encoded → L5 meta-patterns emerge

See [README_meta_strategy_controller.md](README_meta_strategy_controller.md) for full analysis.

### Running
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_meta_strategy_controller.py
```

---

## Experiment 48: Cross-Domain Structural Analogy Transfer — AGI Stretch (SP64)
Demonstrates **HPM L5 expertise: structural analogy across surface-different domains**.
MAP and FILTER schemas learned on integer lists transfer to string lists with **zero
domain-B training examples**. The scaffold nodes (VAR_INP, LIST_INIT, FOR_LOOP, etc.)
are domain-invariant; only the op slot differs across domains.

### Tasks / Phases
| Phase | Task | Method | Domain-B examples |
|-------|------|--------|-------------------|
| 1–4 | add_1, MAP+1, MAP*2, FILTER_pos | Same as SP63 | — |
| 5 | Learnability probe | LearnabilityProbe.assess() | 0 |
| 6 | MAP_upper: ["hello"] → ["HELLO"] | DomainTransferBridge substitution | 0 |
| 7 | FILTER_starts_a: keep words starting with 'a' | Bridge substitution | 0 |
| 8 | Classification robustness | 3 probe tasks | 0 |

### Success Conditions
- Phase 5: classification == "analogous"
- Phase 6: MAP_upper depth ≤ 2 AND 0 domain-B training examples
- Phase 7: FILTER_starts_a depth ≤ 2 AND 0 domain-B training examples
- Phase 8: 3/3 correct learnability classifications

See [README_cross_domain_analogy.md](README_cross_domain_analogy.md) for full analysis.

### Running
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_cross_domain_analogy.py
```

---

## Experiment 49: Autonomous Op Discovery (SP65)
Removes the last hand-seeded assumption — the L1 op vocabulary. The agent discovers
its own primitive ops from raw I/O pairs: element type detection activates a candidate
library; empirical consistency testing filters candidates; intersection mode resolves
ambiguity. The schema BFS + oracle pipeline runs unchanged on discovered ops.

### Tasks / Phases
| Phase | Task | Key result |
|-------|------|------------|
| 1 | Integer bootstrap from 3 seed examples | ≥ 3 ops discovered |
| 2 | Schema acquisition on discovered ops | MAP + FILTER macros |
| 3 | String bootstrap | str_upper + str_cond_a discovered |
| 4 | Cross-domain transfer with discovered string ops | depth ≤ 2, 0 domain-B examples |
| 5 | Float bootstrap | Duck-typed from int library |
| 6 | Ambiguity resolution | 2nd example narrows to val *= 2 |

### Success Conditions
- All 6 phases pass → General HPM AI L1 bootstrapping demonstrated

See [README_autonomous_op_discovery.md](README_autonomous_op_discovery.md) for full analysis.

### Running
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_autonomous_op_discovery.py
```

---

# HPM Hierarchical Abstraction Experiments (SP54-SP56)

This directory contains experiments for **HPM-Native Synthesis, Library Discovery, and Compositional Abstraction**.

## Experiment 45: Library Discovery & Probing (SP55)
Demonstrates an HFN agent autonomously discovering an external Python library, probing its functions to determine their behavioral "portraits," and using those portraits to solve data-transformation tasks.

### Key Components
- `mock_tool_lib.py`: An opaque external library with functions like `find_pairs`, `uniquify`, and `invert_nested`.
- `library_query.py`: Implements `LibraryScannerQuery` (discovery) and `LibraryProbingConverter` (active sensing).
- `experiment_library_discovery.py`: The main experiment driver.

### HPM Principles Demonstrated
1.  **Gap-Driven Exploration**: Discovery is triggered by coverage gaps ($Accuracy - Complexity$) when existing priors fail.
2.  **Behavioral Probing**: Functions are "recognized" by their input-output deltas, not their names or source code.
3.  **No Oracle Leakage**: Uses a generic, hash-based dense projection oracle (`BehavioralOracle`) instead of hand-crafted features.
4.  **Probabilistic Retrieval**: Tools are selected based on HFN log-likelihood (`node.log_prob(goal_vec)`).

## Experiment 46: Compositional Abstraction (SP56)
Demonstrates the **Hierarchical Pattern Stack** by abstracting second-order patterns (Meta-Relations) from first-order relations.

### Key Components
- `sp56_oracle.py`: A stateful oracle that populates a 90D manifold (30D Content | 30D Relation | 30D Meta-Relation).
- `experiment_compositional_abstraction.py`: The main experiment driver.

### HPM Principles Demonstrated
1.  **Compositional Abstraction**: Building Level 3 (Meta-Relation) nodes from sequences of Level 2 (Relation) nodes.
2.  **Zero-Shot Cross-Domain Transfer**: Using abstract structural principles (e.g., "Oscillation") discovered in Math/Spatial domains to predict patterns in a strictly novel Boolean domain.
3.  **Manifold Factorization**: Demonstrates how hierarchical constraints emerge naturally from the geometric factorization of a shared latent space.

### Running the Experiments
```bash
# Run SP55: Library Discovery
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_library_discovery.py

# Run SP56: Compositional Abstraction
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_compositional_abstraction.py

# Run SP66: Enhanced Cross-Domain Analogy
PYTHONPATH=. .venv/bin/python hpm_fractal_node/experiments/experiment_cross_domain_analogy_enhanced.py
```

---

## Experiment 66: Enhanced Cross-Domain Analogy (SP66)
Extends cross-domain analogy transfer with **Pattern Density**, **Affective Evaluators**, and **AST-level substitution**.

### HPM Principles Demonstrated
1. **Pattern Density (App A.8)**: Tracks connectivity, reinforcement, and usage frequency as HFN nodes. Implements density-guided pruning.
2. **Affective Evaluator (§9.3, §9.4)**: Manages emotional state (arousal, valence) as HFN state. Anxiety-driven persistence and curiosity-driven exploration.
3. **AST Substitution**: Robust structural transformation of code scaffolds using Python's `ast` module.
4. **Fractal Uniformity**: All persistent internal state (density, affect) is stored as standard HFN nodes.

See [README_SP66.md](README_SP66.md) for full analysis.
