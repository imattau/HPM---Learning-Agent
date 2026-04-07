# Redesign Plan: Structural Decoding & Execution

## Diagnosis
The user correctly identified that our current `TaskRunner` treats HFN like an autoregressive LLM, searching for raw syntax tokens one by one (e.g. `[=]`, `[1]`). The HFN should instead search over *structural priors* (e.g. `return_constant`, `for_loop`), and a `Decoder` should translate that structure into valid Python syntax.

## 1. Quick Fix Experiment (Hardcode Test)
- Create a simple hardcoded test in `TaskRunner` that completely bypasses the HFN planner.
- Hardcode the plan to be the structural concept: `["return_constant_1"]`.
- The `Decoder` (newly implemented) will map `"return_constant_1"` -> `"return 1"`.
- The `PythonExecutor` will run it and the `Evaluator` will score it.
- This verifies the downstream pipeline before fixing the upstream HFN search.

## 2. Refactor Decoder
- Create a `CodeDecoder` class that maps structural semantic strings (or HFN node IDs) to Python syntax.
- Example: `node.id == "prior_return_constant"` -> `return 1`.
- The Decoder's job is to guarantee syntactic validity so the HFN doesn't have to learn Python syntax from scratch, only program structure.

## 3. Refactor Agent Planning (Structural Search)
- Modify `DevelopmentalAgent.plan` so that instead of returning raw `VOCAB` indices, it returns the *IDs of the prior rules* it selected (e.g. `["prior_return_constant"]`).
- The action space is no longer 80D `VOCAB_INDEX`. We don't need token-by-token generation.
- The state space can represent the "conceptual state" of the program, or we can use a simplified semantic vector.

## 4. Refactor Priors
- Define high-level structural priors in `_load_python_priors` rather than low-level token priors.
- Prior 1: `id="prior_return_constant"`, `Delta=[target_value]`.
- Prior 2: `id="prior_assign"`, etc.

## Execution Steps
1. Add the Hardcode Test to `run_task` to prove the `Executor` and `Evaluator` work.
2. Implement `CodeDecoder` to map high-level actions to Python code.
3. Update `DevelopmentalAgent` and `TaskRunner` to use this new structural pipeline.