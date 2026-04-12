# SP62: Experiment 46 — Generative Forward Model

## Overview

Demonstrates **HPM Level 4: Generative Rules / Mental Simulation**.

The HPM framework defines L4 as the ability to manipulate higher-level patterns in loosely decoupled ways — running simulated scenarios, evaluating options, and updating beliefs without immediate dependence on external feedback.

SP61 (Experiment 45) demonstrated L1-L3, but every planning step called `PythonExecutor` — the agent was fully externally-grounded. SP62 adds L4 by building a `StateTransitionModel`: a learned forward model that predicts 20D oracle state vectors from node sequences **without executing any code**. Planning (Phase 5) navigates over predicted states; the oracle is called only at the end to verify.

The key result: **0 oracle calls during BFS navigation** for the held-out MAP*2 task. The agent imagines the effect of each candidate plan and selects the one predicted to match the goal state, then verifies once.

## Success Conditions (all three pass)

| Condition | Criterion | Result |
|-----------|-----------|--------|
| Phase 5: L4 Mental Simulation | oracle_calls_during_search == 0 AND solution correct | PASS |
| Phase 6: Forward model accuracy | MAE < 0.15 on STRUCT_DIMS | PASS (MAE = 0.000) |
| Overall | Both Phase 5 and Phase 6 pass | PASS: "[SUCCESS] SP62 Generative Forward Model — HPM L4 Achieved!" |

## Curriculum

### Phases 1-4: Primitive and Schema Acquisition (with Transition Recording)

Phases 1-4 are identical in task structure to SP61, but each solved path additionally records per-step state transitions that populate the `StateTransitionModel`.

#### Phase 1: Primitive Acquisition
Three scalar tasks solved by direct enumeration over perceptual ops. Each solution is registered as a macro AND its step-by-step oracle states are recorded as per-node deltas.

- `add_1`: `x += 1` (VAR_INP + percept_op_0)
- `mul_2`: `x *= 2` (VAR_INP + prior_rule_OP_MUL2)
- `sub_1`: `x += -1` (VAR_INP + percept_op_2)

#### Phase 2: MAP Schema Discovery (MAP+1)
BFS over scaffold-restricted candidates discovers the 6-step MAP+1 program:
```
VAR_INP → LIST_INIT → FOR_LOOP → ITEM_ACCESS → percept_op_0(+1) → LIST_APPEND
```
Registered as `macro(MAP_plus1)` with 6 constituent nodes. Transitions recorded.

#### Phase 3: MAP Transfer via Decomposition (MAP*2)
`_macro_decompose_search()` substitutes `percept_op_0` (+1) with `prior_rule_OP_MUL2` (*2) to yield MAP*2 at depth 2. Registered as `macro(MAP_mul2)`. Transitions recorded.

#### Phase 4: FILTER Schema Discovery
Substituting `percept_op_0` with `prior_rule_COND_IS_POSITIVE` produces a FILTER program. Registered as `macro(FILTER_pos)`. Transitions recorded.

After Phase 4, the forward model knows per-node deltas for all distinct nodes across the training paths.

### Phase 5: Imaginative Planning (L4 Demonstration)

**Task**: MAP*2 on held-out unseen inputs `[[2,4,6],[1,3,5]]` → `[[4,8,12],[2,6,10]]`.

The agent already has `macro(MAP_mul2)` from Phase 3. The goal state is computed from that macro's rendered code on the held-out inputs (one pre-BFS oracle call for goal setup — not a search call).

**Imaginative BFS** then navigates:
1. Starts from the baseline state (`pass`)
2. At each step, calls `forward_model.predict(state, op)` — no oracle, no code execution
3. Distances are computed over `STRUCT_DIMS` only (structural flags, not data-dependent content)
4. All paths predicted within the distance threshold are collected as candidates
5. After BFS completes, candidates are verified in order by calling the oracle once per candidate

**Phase 5 result**:
- BFS finds two candidates within threshold: `macro(MAP_plus1)` and `macro(MAP_mul2)`
- Verification tries MAP_plus1 first → fails oracle check
- Verification tries MAP_mul2 → succeeds
- Oracle calls during BFS navigation: **0**
- Oracle calls for verification: **2** (one failed, one success)

### Phase 6: Forward Model Accuracy Report

Evaluates mean absolute error (MAE) between forward model predictions and true oracle states, restricted to `STRUCT_DIMS`.

**Phase 6 result**: MAE = **0.000** across all training paths on structural dimensions.

## Architecture

### StateTransitionModel

A delta-based forward model that learns from observed execution paths.

**Training** (`record_path`): For each solved path of length k, the oracle is called at every prefix (k+1 calls total). The per-step delta `state[i+1] - state[i]` is stored per `node_id`. Multiple deltas for the same node are averaged at prediction time.

**Prediction** (`predict`): Returns `current_state + mean_delta` for the given node. For macro nodes, prediction is **recursive** — the model steps through each constituent node in sequence, composing their deltas.

**Accuracy** (`prediction_error`): Mean absolute error of `predict_path(start, path)` vs the true final oracle state.

### ImaginativePlanner

Extends `InducedSchemaAgent` (SP61) with:

1. **`_record_transitions(path, inputs)`** — Step-executes a solved path to collect per-node state transitions for the forward model. Called automatically after each macro registration during Phases 1-4.

2. **`_imaginative_bfs(inputs, outputs, goal_state)`** — BFS over forward-model-predicted states. No oracle calls inside the search loop. Returns all candidate paths predicted within the structural distance threshold, sorted by predicted distance.

3. **`imagine_and_verify(inputs, outputs, goal_state)`** — Calls `_imaginative_bfs`, then verifies each candidate in order with oracle calls only at verification. Returns the first candidate whose execution output matches the expected outputs.

### Structural Dimensions (STRUCT_DIMS)

```python
STRUCT_DIMS = [0] + list(range(10, 17))
```

The 20D oracle state vector mixes code-structure flags with data-dependent content statistics. Only structural dims are used for BFS navigation and accuracy evaluation:

| Dims | Description | Predictable? |
|------|-------------|-------------|
| 0 | valid (execution succeeded) | Yes — structural |
| 1-9 | is_list, avg_length, content stats (mean/min/max/first/last), has_mutation, is_const | No — data-dependent |
| 10-16 | for_loop, list_append, list_init, item_access, var_inp, list_type_match, op_mul2 | Yes — code-structure flags |
| 17-19 | (additional dims) | Excluded |

Content statistics (dims 1-9) vary with input values and produce noise when trained across both scalar and list paths (0.5 averaging). Code-structure flags (dims 10-16) transition exactly once when the corresponding op is applied, making them reliably predictable regardless of input type.

## Key Design Decisions

**STRUCT_DIMS excludes data-dependent dims**: Content statistics depend on the actual values in the input lists, not just the program structure. A forward model trained on multiple tasks cannot predict them reliably. Restricting distance computation to structural flags ensures that BFS navigation is grounded in what the model actually knows.

**Dedup by path identity, not predicted state**: The BFS de-duplicates by the tuple of node IDs in the path, not by the predicted state vector. This ensures that macros with structurally similar predicted states (e.g., MAP_plus1 and MAP_mul2 both predict `for_loop=1, list_append=1`) are all evaluated as candidates rather than the first one shadowing the rest.

**imagine_and_verify returns all threshold candidates**: The method collects every path predicted within the distance threshold, sorts by distance, and verifies each in turn. This tolerates forward model imprecision: if the closest predicted match fails verification, the next candidate is tried.

**Macro prediction is recursive through constituents**: When the forward model encounters a macro node, it does not use a stored delta for the macro as a whole. Instead it recurses through the macro's constituent nodes, composing their individual deltas. This means macro-level prediction accuracy comes for free from the per-primitive delta learning done in Phases 1-4.

**Goal state from macro's own rendered code**: The goal state for Phase 5 is computed by running the actual ASTRenderer-generated MAP*2 code (from `macro(MAP_mul2).inputs`) on the held-out inputs. This ensures structural flags (dims 10-16) match exactly what the forward model was trained on, so that the predicted state and goal state are on the same representational footing.

## HPM Principles Demonstrated

| HPM Component | Implementation |
|---------------|----------------|
| Pattern substrate | HFN macro nodes in Forest (Polygraph); StateTransitionModel delta table |
| Pattern dynamics | Delta accumulation per node from training paths; recursive macro prediction |
| Pattern evaluator/gatekeeper | Structural distance threshold in imaginative BFS; oracle verification |
| Pattern fields | Curriculum phases; STRUCT_DIMS restriction; held-out inputs for generalization |
| L1: Sensory regularities | Grounded perceptual ops (+1, *2, -1) learned from scalar execution |
| L2: Latent structural representations | MAP/FILTER macro schemas from Phases 2-4 |
| L3: Relational rules | meta_list_iteration prefix (common structure across MAP and FILTER) |
| L4: Generative rules | Forward model predicts state from node sequences; BFS plans without oracle |
| Transfer | Macro reuse from training tasks to held-out inputs; 0 oracle calls during navigation |

## Running

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_generative_forward_model.py
```

Expected output includes:
```
[SUCCESS P5] L4 Mental Simulation demonstrated — 0 oracle calls during search
[SUCCESS P6] Forward model quantitatively accurate
[SUCCESS] SP62 Generative Forward Model — HPM L4 Achieved!
```

## Dependencies

- `hpm_fractal_node/experiments/experiment_induced_schema_library.py` — provides `InducedSchemaAgent` (SP61)
- `hpm_fractal_node/experiments/experiment_unified_perception_action.py` — provides `ASTRenderer`, `EmpiricalOracle`, `PythonExecutor`, `SchemaTransferAgent`, constants
- `hfn/hfn.py`, `hfn/forest.py`, `hfn/observer.py`, `hfn/retriever.py`, `hfn/evaluator.py`
