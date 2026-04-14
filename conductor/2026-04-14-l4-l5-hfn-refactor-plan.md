# L4 & L5 HFN Refactoring Plan

## Objective
Refactor the L4 Forward Model and L5 Meta-Controller to store their state as native HFN nodes within `TieredForest` instances. This replaces the current Python dictionary-based storage, bringing full fractal uniformity to the HPM architecture where all state (patterns, forward models, meta-strategies) is represented consistently.

## Scope & Impact
- **L4 (Forward Model):** Create `HFNStateTransitionModel` in `hpm_ai_v2/utils/hfn_forward_model.py` to store state deltas as HFN nodes (`delta:{node_id}`).
- **L5 (Meta-Controller):** Create `HFNMetaStrategyController` in `hpm_ai_v2/utils/hfn_meta_controller.py` to store strategy performance metrics as HFN nodes (`meta:{goal_type}:{bucket}:{strategy}`).
- **Integration:** Update `BaseHFNAgent.__init__` to conditionally inject these new components via optional boolean flags (`use_hfn_forward_model`, `use_hfn_meta_controller`).
- **Backward Compatibility:** All existing benchmarks and experiments will continue to work seamlessly if these flags default to `False`.

## Proposed Solution

### 1. `hpm_ai_v2/utils/hfn_forward_model.py`
Create `HFNStateTransitionModel`.
- Uses a `TieredForest` configured with `D=config.m_dim` to store delta vectors in the node's `mu`.
- Implements `record_path` to compute deltas and update the corresponding delta node's `mu` via an Exponential Moving Average (EMA).
- Implements `predict` and `predict_path` to apply these deltas to incoming states.

### 2. `hpm_ai_v2/utils/hfn_meta_controller.py`
Create `HFNMetaStrategyController`.
- Uses a `TieredForest` configured with `D=4` to store meta-stats: `[successes, attempts, total_oracle_calls, last_timestamp]`.
- Implements `record` to update the HFN node corresponding to the `(goal_type, macros_bucket, strategy)` tuple.
- Implements `rank_strategies` to compute success rates and average calls directly from the nodes' `mu` vectors, ranking them dynamically.

### 3. Update `BaseHFNAgent`
Modify `__init__` to accept:
- `use_hfn_forward_model: bool = False`
- `use_hfn_meta_controller: bool = False`
- `forward_model_cold_dir: Optional[str] = None`
- `meta_cold_dir: Optional[str] = None`

Conditionally instantiate the new HFN-based models or fallback to the dictionary-based ones. Update the `self.meta` and `self.forward_model` properties.

## Implementation Steps
- [ ] Create `hpm_ai_v2/utils/hfn_forward_model.py`.
- [ ] Create `hpm_ai_v2/utils/hfn_meta_controller.py`.
- [ ] Update `hpm_ai_v2/utils/__init__.py` to export these new classes.
- [ ] Modify `BaseHFNAgent.__init__` to accept the new configuration flags and initialize the corresponding models.
- [ ] Run a subset of benchmarks (e.g., `experiment_sp71_image_fewshot.py` and `experiment_sp74_graph_fewshot.py`) with the flags explicitly enabled to verify correctness.

## Verification
- The default behavior (flags = `False`) must cause no regressions.
- When enabled, the HFN variants must correctly manage forest instances, create nodes, update `mu` vectors, and provide identical functional outcomes.
