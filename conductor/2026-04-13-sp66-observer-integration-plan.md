# SP66 Observer Integration Plan

## Objective
Finalize the integration of `PatternDensityTracker` and `AffectiveEvaluator` directly into the `Observer` learning loop (`hfn/observer.py`), ensuring that density and affective state are updated dynamically as patterns are discovered, reinforced, and used.

## Implementation Steps
- [ ] Add imports for `time`, `PatternDensityTracker`, and `AffectiveEvaluator` to `hfn/observer.py`.
- [ ] Update `Observer.__init__` to accept configuration parameters for density and affective trackers (`use_density_tracker`, `use_affective_evaluator`, `density_cold_dir`, `affective_cold_dir`).
- [ ] Initialize `self.density_tracker` and set up the `evaluator` with `AffectiveEvaluator` if requested in `__init__`. Set a flag `self._affective_enabled`.
- [ ] Modify `_create_node` (or the exact equivalent where new nodes are registered in `Observer`) to call `self.density_tracker.update_structural_connectivity(new_node)` when a new node is created.
- [ ] Modify `_update_weights` to:
    - Retrieve `aff_bonus` from the affective evaluator if enabled.
    - Apply the `aff_bonus` to the accuracy update for effective explaining nodes.
    - Call `self.density_tracker.update_evaluator_reinforcement` for both explaining (success=True) and non-explaining (success=False) nodes.
    - Call `self.density_tracker.update_field_amplification` for all nodes in the effective explanation tree.

## Verification
- Run the full unit test suite `pytest tests/` to confirm that standard observer behavior remains robust.
- Ensure the newly added `tests/test_density.py` and `tests/test_affective.py` continue to pass.
