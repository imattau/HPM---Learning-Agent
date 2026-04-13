# SP63: Experiment 47 — Meta-Strategy Controller (HPM Level 5)

## Overview

Demonstrates **HPM Level 5: Meta-patterns / Metacognition**.

The HPM framework defines L5 as: *"Strategies for monitoring success, correcting errors and evaluating performance. Metacognition involves recognising patterns in one's own reasoning, strategies and errors. It includes patterns that evaluate the reliability of other patterns, select among competing approaches and guide self-correction."*

SP62 (Experiment 46) demonstrated L1–L4 but used a hardcoded strategy order (exact → decompose → imagine → bfs). SP63 adds L5: the agent observes its own solve behaviour across tasks, recognises patterns in which strategies succeed in which contexts, and **adapts strategy selection** accordingly.

The key result: after training on Phases 1–4, the `MetaStrategyController` has populated strategy-success statistics across context buckets (goal_type × n_macros). In Phase 5, it selects the historically best strategy first for each of six novel tasks, matching the expected strategy in ≥4/6 cases. In Phase 6, this learned selection reduces total oracle calls to ≤80% of the fixed-order baseline.

## Success Conditions (all three pass)

| Condition | Criterion | Result |
|-----------|-----------|--------|
| Phase 5: Meta-directed strategy selection | strategy_match ≥ 4/6 tasks | PASS |
| Phase 6: Oracle efficiency | meta_oracle_calls ≤ 0.80 × baseline_oracle_calls | PASS |
| Phase 7: Meta-pattern emergence | ≥ 3 distinct meta-patterns encoded | PASS |

All three → `[SUCCESS] SP63 Meta-Strategy Controller — HPM L5 Achieved!`

## Curriculum

### Phases 1–4: Schema Acquisition (with Strategy Recording)

Phases 1–4 are identical in task structure to SP62, but each solved path is wrapped in `solve_with_meta()` / manual `SolveRecord` registration so that strategy outcomes are fed into the `MetaStrategyController`.

| Phase | Task | Strategy recorded |
|-------|------|-------------------|
| 1 | add_1, mul_2, sub_1 (scalar) | bfs |
| 2 | MAP+1 (list) | bfs |
| 3 | MAP*2 (list) | decompose (if depth ≤ 2), else bfs |
| 4 | FILTER_pos (list) | decompose (if depth ≤ 2), else bfs |
| 4b | MAP_mul2_extra | decompose / bfs |

After Phase 4 the controller has populated context buckets: `(scalar, 0-macros)`, `(map, 1-macro)`, `(map, ≥2-macros)`, `(filter, ≥2-macros)`.

### Phase 5: Meta-Directed Strategy Selection (6 Novel Tasks)

Six novel tasks are solved via `solve_with_meta()`. The controller queries historical success rates per `(goal_type, n_macros_bucket)` and ranks strategies accordingly.

| Task | Goal type | Expected best strategy |
|------|-----------|------------------------|
| T1: MAP+3 | map | decompose |
| T2: MAP*2 (new inputs) | map | decompose |
| T3: FILTER_neg | filter | decompose |
| T4: add_5 | scalar | bfs |
| T5: MAP_sub1 | map | decompose |
| T6: MAP+1 (new inputs) | map | exact / decompose |

**Success**: strategy_match ≥ 4/6.

### Phase 6: Oracle Efficiency Comparison

The same six tasks are re-solved using the fixed `DEFAULT_ORDER` (exact → decompose → imagine → bfs) as a baseline. Total oracle calls are compared.

- **Meta-directed**: strategies skip early failures because history ranks the winning strategy first
- **Baseline**: tries all strategies in fixed order, accumulating oracle calls for failed attempts

**Success**: `meta_oracle_calls ≤ 0.80 × baseline_oracle_calls`.

### Phase 7: Meta-Pattern Report

`MetaStrategyController.meta_patterns()` reports, for each context bucket with ≥1 attempt, the best strategy and its success rate. Each distinct (goal_type, n_macros_bucket) entry with a recorded best strategy counts as one meta-pattern.

**Success**: ≥ 3 distinct meta-patterns encoded.

## Architecture

### SolveRecord

Dataclass capturing the outcome of each solve attempt:

```python
@dataclass
class SolveRecord:
    task_id: str
    goal_type: str       # "scalar" | "map" | "filter"
    n_macros: int        # macros registered at solve time
    strategy: str        # "exact" | "decompose" | "imagine" | "bfs"
    depth: int
    oracle_calls: int
    success: bool
    wall_ms: float
```

### CountingOracle

Wraps `EmpiricalOracle` with a per-task call counter (`call_count`). Reset before each strategy attempt so oracle cost is attributed per strategy.

### MetaStrategyController

Maintains a `(context_key, strategy) → [successes, attempts, total_oracle_calls]` table.

- **Context key**: `(goal_type, n_macros_bucket)` where `n_macros_bucket` ∈ {0, 1, 2} (none / one / ≥2 macros)
- **`record(rec)`**: updates success counts and oracle call totals for the context
- **`rank_strategies(goal_type, n_macros)`**: sorts by success rate descending, then mean oracle calls ascending; falls back to `DEFAULT_ORDER` if no history
- **`meta_patterns()`**: returns human-readable summary of best strategy per context bucket
- **`DEFAULT_ORDER`**: `["exact", "decompose", "imagine", "bfs"]`

### MetaAwareAgent

Extends `ImaginativePlanner` (SP62) with:

1. **`_detect_goal_type(inputs, outputs)`** — classifies task as `"scalar"`, `"map"`, or `"filter"` from output structure
2. **`_try_strategy(strategy, inputs, outputs)`** — dispatches to named strategy; returns `(code, depth)` or `None`
3. **`_bfs_only(inputs, outputs)`** — raw BFS search (no exact/decompose pre-checks); mirrors the BFS portion of `_induced_bfs`
4. **`solve_with_meta(task_id, inputs, outputs)`** — queries controller for ranked strategy order, tries each, records outcome; returns `(code, SolveRecord)`
5. **`solve_baseline(task_id, inputs, outputs)`** — same solve loop using `DEFAULT_ORDER` (no controller); used for Phase 6 comparison

## HPM Principles Demonstrated

| HPM Component | Implementation |
|---------------|----------------|
| Pattern substrate | HFN macro nodes; `MetaStrategyController` stats table |
| Pattern dynamics | Strategy success rates updated after each solve |
| Pattern evaluator/gatekeeper | Success rate ranking; oracle call efficiency as secondary criterion |
| Pattern fields | Curriculum phases; context buckets by goal type and macro count |
| L1: Sensory regularities | Grounded perceptual ops (+1, *2, -1) |
| L2: Latent structural representations | MAP/FILTER macro schemas |
| L3: Relational rules | `meta_list_iteration` prefix node |
| L4: Generative rules | Forward model / mental simulation (from SP62) |
| L5: Meta-patterns | Strategy selection policy learned from solve history — **NEW** |
| Monitoring success | Controller tracks successes/attempts per context |
| Correcting errors | Failed strategies recorded; ranked lower in future |
| Evaluating performance | Oracle call efficiency measured against fixed-order baseline |

## Running

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_meta_strategy_controller.py
```

Expected output includes:
```
[SUCCESS] Meta-controller selects learned strategy in context
[SUCCESS] Meta-strategy reduces oracle call overhead
[SUCCESS] Meta-patterns encoded
[SUCCESS] SP63 Meta-Strategy Controller — HPM L5 Achieved!
```

## Dependencies

- `hpm_fractal_node/experiments/experiment_generative_forward_model.py` — provides `ImaginativePlanner`, `StateTransitionModel`, `STRUCT_DIMS` (SP62)
- `hpm_fractal_node/experiments/experiment_induced_schema_library.py` — provides `InducedSchemaAgent` (SP61)
- `hpm_fractal_node/experiments/experiment_unified_perception_action.py` — provides `ASTRenderer`, `EmpiricalOracle`, `PythonExecutor`, constants
- `hfn/hfn.py`, `hfn/forest.py`, `hfn/observer.py`, `hfn/retriever.py`, `hfn/evaluator.py`
