# SP56: Compositional Abstraction Implementation Plan

## Phase 1: Substrate Preparation
- [ ] Create `hpm_fractal_node/code/sp56_oracle.py`:
    - Implement `StatefulOracleSP56` to maintain temporal state and calculate L2/L3 deltas.
    - Use fast zlib-based hashing for Level 1 encoding.
- [ ] Create `hpm_fractal_node/experiments/experiment_compositional_abstraction.py`:
    - Setup the 90D HFN Forest and Observer.
    - Implement curriculum generators for Math, Spatial, and Boolean domains.

## Phase 2: Hierarchical Training
- [ ] Implement Phase 1 Training: Pairwise presentations to stabilize L2 relations in Math and Spatial domains.
- [ ] Implement Phase 2 Training: Full sequence presentations to trigger L3 meta-node creation via utility-driven compression.
- [ ] Verify L3 node formation: Inspect forest for nodes with high energy in the `[60:90]` Meta-Relational slice.

## Phase 3: Zero-Shot Transfer & Validation
- [ ] Implement Phase 3 Test: Present Boolean sequence prefix.
- [ ] Verify L2 discovery: Ensure new Boolean-specific L2 nodes are created.
- [ ] Verify L3 activation: Confirm the L3 `Oscillator` node is retrieved and constrains the Boolean prediction.
- [ ] Calculate success metrics: Accuracy, Compression Ratio, and Manifold Separation.

## Phase 4: Optimization & Finalization
- [ ] Integrate `AsyncHFNController` for parallel sequence processing if performance is an issue.
- [ ] Document results in `hpm_fractal_node/experiments/README_SP56.md`.
