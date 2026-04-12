# SP54: Experiment 44 — Unified Perception-Action Schema Learning

This experiment demonstrates that the HPM framework can learn structured, executable
programs from input-output examples by coupling **grounded perceptual operations** with
**structural program schemas** through a goal-directed BFS planner.

## 1. Objective

Show that a single agent can:
1. Learn a grounded perceptual operation (+1) from scalar examples (Task A)
2. Discover the MAP schema from list examples (Task B)
3. Transfer MAP to a new operation (×2) without re-learning structure (Task C)
4. Discover the FILTER schema from conditional examples (Task D)

All tasks are solved by synthesising executable Python programs verified by an
empirical oracle, not by pattern-matching over pre-labelled features.

## 2. Architecture

### Components

| Component | Role |
|-----------|------|
| `ASTRenderer` | Translates HFN node sequences into executable Python AST |
| `EmpiricalOracle` | Runs synthesised code against examples; returns a 20-dim state vector |
| `PythonExecutor` | Executes candidate programs in a sandboxed scope |
| `SchemaTransferAgent` | Orchestrates planning, retrieval, observation, and BFS |
| `GoalConditionedRetriever` | Retrieves prior-rule HFN nodes whose geometry matches the goal delta |

### State Vector (20D)

| Dims | Meaning |
|------|---------|
| 0 | valid (no execution errors) |
| 1 | is_list (output is a list) |
| 2 | avg output length |
| 3–7 | content statistics (mean, min, max, first, last) |
| 8 | has_mutation (output ≠ input) |
| 9 | is_const (constant output) |
| 10–16 | code structure flags (for_loop, list_append, list_init, item_access, var_inp, list_type_match, op_mul2) |

### Prior Rule Concepts (14)

`RETURN`, `CONST_1`, `VAR_INP`, `OP_ADD`, `OP_MUL2`, `OP_SUB`,
`LIST_INIT`, `FOR_LOOP`, `ITEM_ACCESS`, `LIST_APPEND`,
`COND_IS_EVEN`, `COND_IS_POSITIVE`, `BLOCK_ELSE`, `BLOCK_END`

Plus 4 **grounded perceptual ops** with learned deltas: +1, −1, +2, +3.

## 3. Planning Strategy

### Phase 0: Goal-Type Directed BFS

Before running stochastic search, the planner runs a deterministic BFS with a
**goal-type-specific restricted operator set**:

| Goal type | Detection | Operator set (~ops) | Typical depth |
|-----------|-----------|---------------------|---------------|
| Scalar | `is_list = 0` | — (skip BFS) | — |
| MAP | `is_list = 1`, same output length | VAR_INP, LIST_INIT, FOR_LOOP, ITEM_ACCESS, OP_MUL2, LIST_APPEND + perceptual_ops | 6 |
| FILTER | `is_list = 1`, shorter output | VAR_INP, LIST_INIT, FOR_LOOP, ITEM_ACCESS, COND_IS_POSITIVE, COND_IS_EVEN, LIST_APPEND, BLOCK_END | 6 |

The BFS runs a time-limited search (30s). If no solution is found it falls back to the
stochastic tree search. This structure means list programs are solved deterministically in
seconds rather than requiring thousands of random iterations.

### Phase 1: Stochastic Tree Search (fallback / scalar)

For scalar goals and unsolvable list goals, a stochastic tree search selects parent nodes
proportionally to `exp(advantage / 5) × curiosity_bonus`, where
`advantage = accuracy − ema_baseline`. The baseline tracks EMA of accuracy (not
total_score) to avoid inflation from one-time novelty bonuses.

## 4. Results

All four tasks solve reliably in a single run:

| Task | Description | Inputs → Outputs | Planner | Depth | Iterations |
|------|-------------|-----------------|---------|-------|------------|
| A | Add one (scalar) | `[1,5,10] → [2,6,11]` | Stochastic | — | ~11–40 |
| B | Map add one | `[[1,2],[10,20]] → [[2,3],[11,21]]` | BFS (MAP) | 6 | 1 (BFS) |
| C | Map double | `[[3,5],[-1,0]] → [[6,10],[-2,0]]` | BFS (MAP) | 6 | 1 (BFS) |
| D | Filter positive | `[[-1,2,-3,4],[0,5,-2]] → [[2,4],[5]]` | BFS (FILTER) | 6 | 1 (BFS) |

### Synthesised Programs

**Task A** (stochastic search):
```python
x = inp
x += 1
return x
```

**Task B** (BFS — MAP with grounded +1 percept):
```python
x = inp
res = []
for item in list(x):
    val = item
    val += 1
    res.append(val)
return res
```

**Task C** (BFS — MAP with OP_MUL2):
```python
x = inp
res = []
for item in list(x):
    val = item
    val *= 2
    res.append(val)
return res
```

**Task D** (BFS — FILTER with COND_IS_POSITIVE):
```python
x = inp
res = []
for item in list(x):
    val = item
    if val > 0:
        res.append(val)
return res
```

## 5. Key Bugs Fixed During Development

| Bug | Symptom | Fix |
|-----|---------|-----|
| IDDFS shared `visited_codes` | Depth-6 solutions never explored — paths visited at depth N blocked at N+1 | Replaced IDDFS with plain BFS; `visited_codes` accumulates monotonically |
| BFS over all CONCEPTS | 14 ops × depth 7 = millions of paths; timeout after 30s | Restricted to goal-type-specific scaffold (6–8 ops) |
| OP_MUL2 renderer skip | `val *= 2` never emitted — OP_MUL2 was in exclusion list | Added context-aware renderer case: `val *= 2` inside loop, `x *= 2` outside |
| ema_baseline inflation | One-time pair novelty bonuses (+40×5) pushed baseline to ~+50; deep nodes had advantage ≈ −83 → weight ≈ 0 | Store accuracy (not total_score) in state field 1 for EMA tracking |
| `_goal_scaffold_candidates` missing VAR_INP | FOR_LOOP never tried because `x=inp` step wasn't seeded | Added VAR_INP as first scaffold candidate for list goals |
| `planning_nodes` O(n×m) per iteration | `active_nodes()` called inside candidates loop; ~20k ops/iteration at scale | Hoisted outside loop; capped to top-50 by weight |

## 6. HPM Principles Demonstrated

1. **Pattern substrates**: HFN nodes encode program fragments; the forest is the substrate
   for structural program patterns.

2. **Pattern dynamics**: The observer updates node weights based on empirical accuracy;
   compression (macro-creation) stabilises frequently co-activated sequences.

3. **Pattern evaluators**: The EmpiricalOracle is a pure pattern evaluator — it gates
   which synthesised programs are worth retaining.

4. **Hierarchy**: Task A learns a grounded (Level 1) percept. Tasks B/C/D learn
   structural schemas (Level 2) that compose percepts into list-processing programs.
   The MAP schema discovered in Task B transfers to Task C without re-learning structure.

## 7. Running the Experiment

```bash
cd /path/to/HPM---Learning-Agent
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_unified_perception_action.py
```

Expected runtime: < 2 minutes for all 4 tasks.
