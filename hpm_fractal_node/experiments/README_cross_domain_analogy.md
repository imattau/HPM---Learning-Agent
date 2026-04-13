# SP64: Experiment 48 — Cross-Domain Structural Analogy Transfer (AGI Stretch)

## Overview

Demonstrates **HPM Level 5 expertise: structural analogy across surface-different domains**.

The HPM paper (p.22, Section 9.1) predicts:
> "Learners should be more sensitive to changes in deep structure than surface details."
> "Experts can: design new models, construct novel explanations, recognise deep similarity
> across superficially unrelated problems, adapt patterns rapidly to new situations."

SP63 completed the HPM five-level hierarchy within a single domain (integer lists). SP64 adds the defining AGI capability: given a structurally analogous but surface-different domain (string lists), transfer schemas with **zero domain-B training examples**.

The key insight: MAP is not a fact about integers. It is a structural pattern — iterate over sequence, apply operation, collect results — that applies to any sequence regardless of element type. The scaffold nodes (`VAR_INP`, `LIST_INIT`, `FOR_LOOP`, `ITEM_ACCESS`, `LIST_APPEND`) are **domain-invariant**.

## Success Conditions (all four pass)

| Condition | Criterion |
|-----------|-----------|
| Phase 5: Learnability probe | classification == "analogous" |
| Phase 6: MAP_upper transfer | depth ≤ 2 AND domain_b_training_examples == 0 |
| Phase 7: FILTER_starts_a transfer | depth ≤ 2 AND domain_b_training_examples == 0 |
| Phase 8: Classification robustness | 3/3 correct learnability classifications |

All four → `[SUCCESS] SP64 Cross-Domain Structural Analogy — AGI Stretch Achieved!`

## Curriculum

### Phases 1–4: Domain A Training (Integer Lists)

Identical to SP63 Phases 1–4. The agent acquires MAP and FILTER macro schemas over integer lists using `solve_with_meta()`.

| Phase | Task | Method |
|-------|------|--------|
| 1 | add_1, mul_2, sub_1 (scalar) | Enumeration |
| 2 | MAP+1 (list) | BFS |
| 3 | MAP*2 (list) | Macro decomposition |
| 4 | FILTER_pos (list) | Macro decomposition |

### Phase 5: Domain B Seeding + Learnability Probe

String ops are seeded as L1 HFN nodes: `str_op_upper` (`item.upper()`), `str_op_lower` (`item.lower()`), `str_cond_starts_a` (`item.startswith('a')`).

`LearnabilityProbe.assess()` examines probe task outputs structurally:
- Are outputs lists of the same length as inputs? → MAP-like structure detected
- Are outputs lists shorter than inputs? → FILTER-like structure detected
- Do length patterns match known scaffold templates AND are domain ops available? → **"analogous"**

**Success**: classification == "analogous"

### Phase 6: MAP_upper Transfer (0 Domain-B Training Examples)

Task: `[["hello", "world"]] → [["HELLO", "WORLD"]]`

`DomainTransferBridge.find_transfer()` iterates over known macros × domain ops. For `MAP_plus1` (the closest scaffold), it:
1. Renders the base MAP scaffold code
2. Identifies the inner loop mutation line (`val += 1`)
3. Substitutes with `str_op_upper`'s render hint (`val = item.upper()`)
4. Executes the modified code and verifies with the oracle

Solution found at **depth 2** (scaffold reuse + op substitution). No domain-B training examples needed.

**Success**: depth ≤ 2 AND domain_b_training_examples == 0

### Phase 7: FILTER_starts_a Transfer (0 Domain-B Training Examples)

Task: `[["apple", "banana", "avocado", "cherry"]] → [["apple", "avocado"]]`

The same bridge substitutes the FILTER scaffold's condition line with `str_cond_starts_a` (`val.startswith('a')`).

**Success**: depth ≤ 2 AND domain_b_training_examples == 0

### Phase 8: Learnability Classification Robustness

`LearnabilityProbe` is tested on three contrasting probe tasks:
- A task with consistent MAP-like structure and available domain ops → "analogous"
- A task with outputs longer than inputs (no known scaffold produces this) → "random"
- A task where an existing macro already solves it directly → "trivial"

**Success**: 3/3 correct classifications

## Architecture

### LearnabilityReport

Dataclass capturing domain assessment:

```python
@dataclass
class LearnabilityReport:
    classification: str          # "random" | "trivial" | "analogous" | "novel"
    recommended_strategy: str    # "bfs" | "exact" | "analogy"
    scaffold_match: List[str]    # macro names with matching scaffold structure
```

### LearnabilityProbe

Evaluates domain learnability before committing to transfer. Implements HPM's curiosity-as-evaluator: prefer environments where pattern improvement is possible.

Assessment order:
1. **Trivial**: any known macro executes correctly as-is → "trivial"
2. **Structural analysis**: outputs are length-preserving (MAP) or length-reducing (FILTER) lists, AND domain ops are available → "analogous"
3. **Novel**: list outputs with recognisable scaffold pattern but no domain ops → "novel"
4. **Random**: no list structure, inconsistent patterns, or output longer than input → "random"

### DomainTransferBridge

Identifies structural scaffold → domain-op substitution mappings.

**Scaffold IDs** (domain-invariant): `prior_rule_VAR_INP`, `prior_rule_LIST_INIT`, `prior_rule_FOR_LOOP`, `prior_rule_ITEM_ACCESS`, `prior_rule_LIST_APPEND`, `prior_rule_BLOCK_END`

**Variable slots**: constituent nodes whose IDs are not in `SCAFFOLD_IDS` — these are the op positions eligible for substitution.

- **`variable_slots(macro)`** — returns indices of non-scaffold constituent nodes
- **`_build_code_with_op(macro, slot_idx, domain_op, renderer)`** — renders base macro code and substitutes the inner mutation line (MAP) or condition line (FILTER) with the domain op's render hint
- **`find_transfer(inputs, outputs, domain_ops, known_macros, executor, renderer)`** — iterates over macros × ops, verifies each substitution; returns `(macro_name, code, depth=2)` on success

### AnalogicalAgent

Extends `MetaAwareAgent` (SP63) with cross-domain transfer capability.

1. **`seed_domain_b()`** — seeds string ops as L1 HFN nodes in the forest
2. **`_try_analogy(inputs, outputs)`** — uses `DomainTransferBridge` to find scaffold substitutions; MAP ops replace the inner mutation line, FILTER ops replace the condition line
3. **`solve_with_meta()`** — inherited from SP63; `MetaStrategyController` learns to rank "analogy" first for domain-B list tasks after observing successful transfers

### Code Substitution Helpers

- **`_replace_map_body(code, render_hint)`** — replaces the inner loop mutation line (`val += N`, `val *= N`) with the domain op's transformed render hint
- **`_replace_filter_condition(code, cond_hint)`** — replaces the `if val ...` line with the domain op's condition, transforming `item` references to `val`

## HPM Grounding

The HPM paper provides two direct theoretical anchors for this experiment:

**Deep structure transfer (p.22)**: HPM predicts experts recognise structural similarity beneath surface differences. The scaffold nodes encode deep structure (sequence iteration); the op slot encodes surface detail (what to do to each element). SP64 operationalises this distinction concretely.

**Curiosity evaluator (p.28–29)**: `LearnabilityProbe` implements HPM's curiosity-as-evaluator mechanism — assessing whether a domain offers "intermediate difficulty" (analogous structure available but ops differ) before committing to transfer. "Random" and "trivial" domains are filtered out; "analogous" domains are prioritised.

## HPM Principles Demonstrated

| HPM Component | Implementation |
|---------------|----------------|
| Pattern substrate | HFN macro nodes; domain-invariant scaffold node IDs |
| Pattern dynamics | Bridge substitution: scaffold preserved, op slot replaced |
| Pattern evaluator/gatekeeper | LearnabilityProbe classifies domain before transfer; oracle verifies substituted code |
| Pattern fields | Domain A (integer lists) → Domain B (string lists); 8-phase curriculum |
| L1: Sensory regularities | String ops seeded as L1 nodes (upper, lower, startswith) |
| L2: Latent structural representations | MAP/FILTER macro schemas transfer domain-invariantly |
| L3: Relational rules | meta_list_iteration prefix applies to string domain without relearning |
| L4: Generative rules | Forward model continues to operate on structural dims (domain-invariant) |
| L5: Meta-patterns | Strategy controller learns to prefer "analogy" for structurally familiar domains |
| Deep structure transfer | Scaffold nodes shared; op slots substituted — 0 domain-B training examples |

## Running

```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_cross_domain_analogy.py
```

Expected output includes:
```
[SUCCESS P5] Learnability: classification=analogous
[SUCCESS P6] MAP transfer: depth=2, domain_b_training_examples=0
[SUCCESS P7] FILTER transfer: depth=2, domain_b_training_examples=0
[SUCCESS P8] 3/3 learnability classifications correct
[SUCCESS] SP64 Cross-Domain Structural Analogy — AGI Stretch Achieved!
```

## Dependencies

- `hpm_fractal_node/experiments/experiment_meta_strategy_controller.py` — provides `MetaAwareAgent`, `MetaStrategyController`, `CountingOracle`, `SolveRecord` (SP63)
- `hpm_fractal_node/experiments/experiment_generative_forward_model.py` — provides `ImaginativePlanner` (SP62)
- `hpm_fractal_node/experiments/experiment_unified_perception_action.py` — provides `ASTRenderer`, `EmpiricalOracle`, `PythonExecutor`, constants
- `hfn/hfn.py`, `hfn/forest.py`, `hfn/observer.py`, `hfn/retriever.py`, `hfn/evaluator.py`
