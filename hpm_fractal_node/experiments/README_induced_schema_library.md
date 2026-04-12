# SP61: Experiment 45 — Induced Schema Library

## Overview

Demonstrates the **full HPM loop**: Perceive → Execute → Verify → Compress → Transfer → Meta-Abstract.

The key advance over Experiment 44 (SP54): schemas are **not hardcoded**. They emerge from solved tasks, get stored as Polygraph macro nodes (from SP60), and are reused on harder tasks via macro decomposition search (substituting constituent ops).

## Success Conditions (all three pass)

| Condition | Criterion | Result |
|-----------|-----------|--------|
| Phase 3: MAP macro reuse | depth ≤ 2 | PASS (depth 2, 0.0s) |
| Phase 5: Compound macro composition | depth ≤ 3 | PASS (depth 2) |
| Phase 6: Meta-schema induction | L3 node exists | PASS (prefix length 4) |

## Curriculum

### Phase 1: Primitive Acquisition
Three scalar tasks (add_1, mul_2, sub_1) are solved by direct enumeration over perceptual ops. Each solution is registered as a Polygraph macro node with its constituent HFN path preserved.

- `add_1`: `x += 1` (VAR_INP + percept_op_0)
- `mul_2`: `x *= 2` (VAR_INP + prior_rule_OP_MUL2)
- `sub_1`: `x += -1` (VAR_INP + percept_op_2)

### Phase 2: MAP Schema Discovery (MAP+1)
BFS over scaffold-restricted candidates discovers the 6-step MAP+1 program:
```
VAR_INP → LIST_INIT → FOR_LOOP → ITEM_ACCESS → percept_op_0(+1) → LIST_APPEND
```
Registered as `macro(MAP_plus1)` with 6 constituent nodes.

### Phase 3: MAP Transfer via Decomposition (MAP*2) — DEPTH REDUCTION TEST
`_macro_decompose_search()` iterates over `macro(MAP_plus1)` constituents. Substituting `percept_op_0` (+1) with `prior_rule_OP_MUL2` (*2) yields MAP*2 at **depth 2** (vs depth 6 for fresh BFS). This is the core depth-reduction demonstration.

### Phase 4: FILTER Schema Discovery
Same decomposition: substituting `percept_op_0` with `prior_rule_COND_IS_POSITIVE` produces a FILTER program. Registered as `macro(FILTER_pos)`.

### Phase 5: Repeated MAP (MAP+2)
MAP+2 (add 2 to each element) is solved at depth 2 by decomposition: substituting `percept_op_0` (+1) with `percept_op_2` (+2 — a perceptual op seeded by the agent). This demonstrates compound macro composition.

### Phase 6: Meta-Schema Discovery
`discover_meta_schema()` finds the longest common prefix across all list-processing macros (MAP_plus1, MAP_mul2, FILTER_pos). The shared prefix of length 4 is:
```
VAR_INP → LIST_INIT → FOR_LOOP → ITEM_ACCESS
```
This is registered as an L3 `meta_list_iteration` node with `relation_type="meta_schema"` and `protected=True`, representing the abstract pattern of list iteration independent of the specific operation applied.

## Architecture

### InducedSchemaAgent
Extends `SchemaTransferAgent` (Experiment 44) with:

1. **`register_macro(name, path, inputs, outputs)`** — Compresses a solution path into a Polygraph HFN node. The macro's `inputs` list preserves constituent op nodes, enabling structural decomposition.

2. **`_macro_decompose_search(inputs, expected_outputs)`** — For each registered macro, tries substituting each "substitutable" constituent (perceptual ops, OP_MUL2, COND_*) with every alternative op. Returns depth=2 on success.

3. **`_macro_exact_search(inputs, expected_outputs)`** — Tries each macro as-is. Returns depth=1 on success.

4. **`_induced_bfs(inputs, expected_outputs, max_depth)`** — Scaffold-restricted BFS that:
   - First tries exact macro match (depth 1)
   - Then tries macro decomposition (depth 2)
   - Falls back to BFS with MAP or FILTER scaffold restriction based on goal type detection

5. **`discover_meta_schema()`** — Finds longest common prefix across macros with ≥4 constituents. Registers L3 node if prefix ≥ 3.

### Key Design Decisions

**Substitutable IDs**: The decomposition search targets specific node IDs (`percept_op_0..3`, `prior_rule_OP_MUL2`, `prior_rule_COND_*`). This keeps the search space tractable while covering the relevant substitutions.

**MAP scaffold restriction**: `_induced_bfs()` detects goal type (list vs scalar, filter vs map) and restricts candidates to the appropriate scaffold nodes. This mirrors `_deterministic_bfs()` in Experiment 44 and keeps BFS tractable.

**Macro as single step**: In BFS, a macro node counts as depth 1 even though it expands to 6 ops for execution. This is the mechanism that achieves depth ≤ 2 for decomposed solutions.

**Percept ops seeded at init**: `InducedSchemaAgent` inherits `_seed_perceptual_ops()` from `SchemaTransferAgent`, which seeds percept_op_0 (+1), percept_op_1 (-1), percept_op_2 (accumulate +2), percept_op_3 (identity). This ensures mul_2 and sub_1 are reachable in Phase 1.

## HPM Principles Demonstrated

| HPM Component | Implementation |
|---------------|----------------|
| Pattern substrate | HFN macro nodes in Forest (Polygraph) |
| Pattern dynamics | Macro registration after solution; decomposition search |
| Pattern evaluator/gatekeeper | BFS correctness check (output match); mastery gate |
| Pattern fields | Curriculum phases; scaffold restriction by goal type |
| Hierarchy | L1 macros (scalars) → L2 macros (MAP/FILTER) → L3 meta-schema |
| Transfer | Depth reduction from 6 → 2 via macro reuse |
| Meta-abstraction | Common prefix extraction across MAP and FILTER macros |

## Running

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_induced_schema_library.py
```

## Dependencies

- `hpm_fractal_node/experiments/experiment_unified_perception_action.py` — provides `ASTRenderer`, `EmpiricalOracle`, `PythonExecutor`, `SchemaTransferAgent`, constants
- `hfn/hfn.py`, `hfn/forest.py`, `hfn/observer.py`, `hfn/retriever.py`, `hfn/evaluator.py`
