# SP65: Experiment 49 — Autonomous Op Discovery (General HPM AI L1)

## Overview

Removes the last hand-seeded assumption: the L1 op vocabulary.

SP54–SP64 all call `_seed_perceptual_ops()`, which hard-codes `+1`, `-1`, `*2`, etc. SP65 replaces this with **autonomous discovery**: given only raw I/O example pairs, the agent detects element type, generates a candidate op library, tests each candidate for empirical consistency, and registers consistent ops as L1 HFN nodes. The schema pipeline (BFS + oracle) runs unchanged on top of the discovered ops.

The key insight: `discover()` returns all candidates consistent with the provided examples (union mode by default). The schema BFS + oracle acts as the evaluator/gatekeeper — it filters out spurious L1 candidates that survive single examples but fail on task verification.

## Success Conditions (all six pass)

| Condition | Criterion |
|-----------|-----------|
| Phase 1: Integer bootstrap | ≥ 3 ops discovered from 3 seed examples |
| Phase 2: Schema acquisition | MAP and FILTER schemas built on discovered ops |
| Phase 3: String bootstrap | str_upper and str_cond_a discovered from string examples |
| Phase 4: Cross-domain transfer | MAP_upper and FILTER_starts_a transferred (depth ≤ 2, 0 domain-B training examples) |
| Phase 5: Float bootstrap | Float ops discovered (duck-typed from int library) |
| Phase 6: Ambiguity resolution | 2 candidates from sparse example; second example resolves to `val *= 2` |

All six → `[SUCCESS] SP65 Autonomous Op Discovery — General HPM AI L1 Achieved!`

## Curriculum

### Phase 1: Integer Domain Bootstrap

Three seed examples are provided (one per primitive task: add_1, mul_2, filter_pos). `OpDiscoverer.discover()` runs in union mode: each example is processed independently and ops consistent with any example are collected.

From three seed examples the agent discovers ≥3 distinct ops (`val += 1`, `val *= 2`, `val > 0`, and others), all registered as `discovered_A_N` L1 HFN nodes.

**Success**: ≥ 3 ops discovered.

### Phase 2: Schema Acquisition Using Discovered Ops

The full BFS + oracle pipeline runs on the discovered ops exactly as in SP63/SP64. MAP+1, MAP*2, and FILTER_pos schemas are acquired and registered as macro nodes. No hand-seeded ops are used.

**Success**: MAP and FILTER macros built on discovered ops.

### Phase 3: String Domain Bootstrap

String seed examples are provided. `OpDiscoverer.detect_element_type()` identifies `str` elements; the `CandidateOpLibrary` returns the string op templates. `discover()` finds ops consistent with the examples, including `val = item.upper()` and `val.startswith('a')`.

**Success**: str_upper and str_cond_a discovered.

### Phase 4: Cross-Domain Transfer with Discovered String Ops

The discovered string ops are registered as domain-B ops via `bootstrap_ops(domain="B")`. `DomainTransferBridge` substitutes them into the integer MAP/FILTER scaffolds as in SP64.

**Success**: MAP_upper and FILTER_starts_a transferred at depth ≤ 2, 0 domain-B training examples.

### Phase 5: Float Domain Bootstrap

Float seed examples are provided (e.g. `[1.0, 2.0] → [2.0, 4.0]`). `detect_element_type()` returns `"float"`, which duck-types to the int op library. Float-compatible ops (`val *= 2`, `val += 1`, etc.) are discovered and registered.

**Success**: float ops discovered from the int library via duck typing.

### Phase 6: Ambiguity Resolution

A sparse single example admits multiple consistent candidates (e.g. both `val *= 2` and `val += N` are consistent with `[3] → [6]`). A second example is provided; `discover()` is called in `intersect=True` mode, which requires consistency across all examples. The intersection narrows to `val *= 2`.

**Success**: ambiguous candidates reduced to the correct unique op by the second example.

## Architecture

### CandidateOpLibrary

Provides finite candidate op templates per element type and op kind:

| Element type | `map` ops | `filter` (cond) ops |
|-------------|-----------|---------------------|
| `int` / `float` | `val += 1`, `val += 2`, `val *= 2`, `val *= 3`, `val //= 2`, `val = val**2`, `val = abs(val)`, `val = -val`, `val = val`, … | `val > 0`, `val < 0`, `val >= 0`, `val % 2 == 0`, `val % 2 != 0`, `val > 1`, `val > 5` |
| `str` | `val = val.upper()`, `val = val.lower()`, `val = val[0]`, `val = val[-1]`, `val = val + val`, `val = val[::-1]`, `val = str(len(val))` | `val.startswith('a')`, `val.startswith('b')`, `val[0].isupper()`, `len(val) > 3`, `len(val) > 5` |

Float elements duck-type to int ops.

### OpDiscoverer

Tests candidate ops against I/O examples and returns those that are empirically consistent.

- **`detect_element_type(examples)`** — infers `"int"`, `"float"`, or `"str"` from the first non-empty input
- **`detect_task_kind(examples)`** — infers `"map"` (length-preserving) or `"filter"` (length-reducing)
- **`discover(examples, max_ops, intersect)`** — main entry point:
  - **Union mode** (`intersect=False`, default): an op is returned if consistent with ANY example. Suitable for mixed-task seed sets where different examples demonstrate different ops.
  - **Intersection mode** (`intersect=True`): an op must be consistent with ALL examples. Suitable for ambiguity resolution when multiple examples narrow a single unknown op.
  - Deduplicates by `render_hint`.

The key design: `discover()` returns **all** consistent ops without selecting among them. The downstream schema BFS + oracle acts as the evaluator/gatekeeper, rejecting ops that produce wrong outputs on actual tasks.

### BootstrappingAgent

Extends `AnalogicalAgent` (SP64) by replacing `_seed_perceptual_ops()` with autonomous discovery.

1. **`_seed_perceptual_ops()`** — overridden as a no-op; the hard-coded op vocabulary is entirely removed
2. **`bootstrap_ops(seed_examples, domain)`** — calls `OpDiscoverer.discover()` and registers each returned op as a `discovered_{domain}_{i}` L1 HFN node; domain-B ops are also added to `domain_ops_b` for `DomainTransferBridge`
3. **`_register_op(node_id, op_dict)`** — registers op as an HFN node with `grounded_op` relation type; stores `render_hint` and `callable` for bridge substitution
4. **`get_discovered_ops(domain)`** — returns op dicts for a given domain prefix
5. **`_try_analogy(inputs, outputs)`** — extended version that uses `op_registry` kind info to correctly distinguish filter ops from map ops when selecting scaffold substitution strategy

## HPM Alignment

| HPM Component | Implementation |
|---------------|----------------|
| Pattern substrate | L1 HFN nodes registered from discovered ops; same forest as schema pipeline |
| Pattern dynamics | Op candidates tested empirically; consistent ones registered as nodes; schema BFS builds L2+ on top |
| Pattern evaluator/gatekeeper | Schema BFS + oracle filters spurious L1 candidates; intersection mode resolves ambiguity |
| Pattern fields | Element type detection activates the correct candidate library; domain prefix tracks op provenance |
| L1: Sensory regularities | Discovered from empirical I/O testing — no hard-coding |
| L2: Schemas | Built on discovered L1 ops; pipeline unchanged from SP63/SP64 |
| L3–L5 | Inherited unchanged from SP64 |
| Innate priors | `CandidateOpLibrary` provides structured inductive bias (the "candidate space") without specifying which ops will be useful |
| Blank slate refusal | HPM explicitly rejects blank-slate learning; the candidate library is the innate prior that individual learning (discovery) refines |

## Running

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_autonomous_op_discovery.py
```

Expected output includes:
```
[SUCCESS P1] Integer domain: N ops discovered
[SUCCESS P2] Schema acquisition on discovered ops
[SUCCESS P3] String domain: str_upper and str_cond_a discovered
[SUCCESS P4] Cross-domain transfer with discovered string ops
[SUCCESS P5] Float domain: ops duck-typed from int library
[SUCCESS P6] Ambiguity resolved to val *= 2
[SUCCESS] SP65 Autonomous Op Discovery — General HPM AI L1 Achieved!
```

## Dependencies

- `hpm_fractal_node/experiments/experiment_cross_domain_analogy.py` — provides `AnalogicalAgent`, `DomainTransferBridge`, `LearnabilityProbe`, `_replace_map_body`, `_replace_filter_condition` (SP64)
- `hpm_fractal_node/experiments/experiment_meta_strategy_controller.py` — provides `MetaAwareAgent`, `MetaStrategyController`, `CountingOracle`, `SolveRecord` (SP63)
- `hpm_fractal_node/experiments/experiment_unified_perception_action.py` — provides `ASTRenderer`, `EmpiricalOracle`, `PythonExecutor`, constants
- `hfn/hfn.py`, `hfn/forest.py`, `hfn/observer.py`, `hfn/retriever.py`, `hfn/evaluator.py`
