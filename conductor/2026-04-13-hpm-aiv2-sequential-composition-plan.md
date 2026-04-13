# Robust Sequential Composition Using AST Transformation

## Objective
Address the failure of Hypothesis H4 in the Multi-Specialist Social Learning (MS-SL) experiment (`hpm_ai_v2/experiments/experiment_ms_sl.py`). The current `RecombinationMixin` attempts to concatenate inputs of two macros, resulting in invalid code. This plan implements a robust alternative: an AST-based `SequentialCompositionMixin` that composes two macros by rendering them as standalone functions and generating a wrapper function that calls them sequentially.

## Background & Motivation
The previous structural recombination method concatenates the constituent nodes of two macros. However, without semantic guidance, the resulting AST often fails to execute properly for compound tasks (e.g., `add1` then `mul2`). By using AST transformation to wrap valid macros into a sequential pipeline, we guarantee syntactic correctness and enable true functional composition.

## Implementation Steps

### 1. Extend `ASTRenderer`
**File:** `hpm_ai_v2/utils/renderer.py`
- Modify the existing `render` method to accept an optional `func_name` parameter (defaulting to `"test_func"`), replacing the hardcoded function name in the generated wrapper.
- Add a new method `render_function(self, node: HFN, func_name: str = "macro_func") -> str` that calls `render` with the specified `func_name`.

### 2. Create `SequentialCompositionMixin`
**File:** `hpm_ai_v2/agents/mixins/sequential_composition.py` (New File)
- Implement `SequentialCompositionMixin` with a `compose_sequential` method.
- `compose_sequential` will:
  - Render both constituent macros as standalone functions with unique names using the updated `ASTRenderer`.
  - Generate a new function body that passes the input through the first macro and its result through the second.
  - Parse the combined code using the `ast` module to ensure validity.
  - Register a new `HFN` macro node and store the pre-generated code in `self.pattern_metadata`.
- Implement `_try_sequential_compose` to iterate over pairs of existing macros, attempt composition, and verify the resulting composite function against task inputs/outputs.

### 3. Update `SocialAnalogicalAgent`
**File:** `hpm_ai_v2/agents/agents.py` (or where the class is defined)
- Import `SequentialCompositionMixin`.
- Add `SequentialCompositionMixin` to the inheritance list of `SocialAnalogicalAgent` alongside existing mixins.

### 4. Update the MS-SL Experiment (Phase 4)
**File:** `hpm_ai_v2/experiments/experiment_ms_sl.py`
- Replace the failing recombination logic in Phase 4.
- Update `phase4_recombination` (or equivalent) to use `alice._try_sequential_compose` on the compound task inputs/outputs.
- Update the success reporting to reflect the new sequential composition logic.

## Verification
- Run the MS-SL experiment: `PYTHONPATH=. python3 hpm_ai_v2/experiments/experiment_ms_sl.py`
- Verify that Phase 4 (Hypothesis H4) now reports `[SUCCESS]` and correctly generates a composite function that applies `add1` followed by `mul2`.