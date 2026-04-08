# Implementation Plan: Structural Refinement (Self-Debugging)

## Objective
Implement Experiment 23 to validate the HFN's ability to "debug" and "patch" existing program graphs by localizing causal faults and splicing in corrective nodes.

## Implementation Steps

### Step 1: Primitives & Tasks Expansion
- **Task**: Add `OP_SUB` (-1) and `OP_MUL2` (*2) to `CONCEPTS`.
- **Task**: Update `_inject_priors` in `DevelopmentalAgent` with these new rules.
- **Task**: Create `hpm_fractal_node/experiments/tasks/perturbation_curriculum.json` with `map_add_two` and `map_add_one_and_double`.
- **Conformity Check**: Verify new priors have correct semantic deltas.

### Step 2: Code Renderer & Executor Upgrades
- **Task**: Update `CodeRenderer` to handle `OP_SUB` and `OP_MUL2`.
- **Task**: Ensure `PythonExecutor` remains robust to structural mutations.
- **Conformity Check**: Test rendering of a patched tree.

### Step 3: Implement Causal Fault Localization & Patching
- **Task**: Implement `patch_graph(root_node, target_delta)` in `DreamingAgent` (or a subclass).
- **Task**: Logic: Locate the "workhorse" node (e.g., `OP_ADD`) and wrap it in a new composition with the corrective patch.
- **Conformity Check**: Verify that `patch_graph` returns a new tree with the correct logical modification.

### Step 4: Update TaskRunner for Iterative Debugging
- **Task**: Modify `run_task` to compute the `residual_delta` on failure.
- **Task**: Trigger `patch_graph` instead of just random Walk/scaffolding.
- **Conformity Check**: Verify the agent attempts to "repair" the current best chunk.

### Step 5: Final Evaluation & Report
- **Task**: Run comparison between "Self-Debugging" and "Restart-from-Scratch" baseline.
- **Task**: Generate `README_structural_refinement.md`.
- **Conformity Check**: Verify efficiency gains (nodes explored) and success on perturbed tasks.

## Verification & Testing
- Automated test: solve `map_add_two` by patching `map_add_one`.
- Verify no regression on base `return_constant` tasks.
