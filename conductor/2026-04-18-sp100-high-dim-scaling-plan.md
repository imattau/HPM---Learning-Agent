# SP100: High-Dimensional Structure vs Correlation (Scaling Test)

## Objective
Test whether HPM can extract and reuse simple hidden structure (quadratic + sine) from high‑dimensional input spaces filled with spurious, correlated, and irrelevant features. This directly targets HPM’s claim that it prioritises deep structure over surface correlations.

## Phased Implementation Plan

### Phase 1: High-Dimensional Domain Components
Implement the infrastructure needed to handle D-dimensional inputs.

- **`HighDimDomainConfig`**:
    - Defines base concepts: `OP_ADD`, `OP_SUB`, `OP_MUL`, `OP_DIV`, `OP_SIN`, `OP_SQUARE`, `OP_CONST`.
    - Dynamic concept generation: `SELECT(i)` for $i \in [0, D-1]$.
- **`HighDimRenderer`**:
    - Renders `SELECT(i)` as `push(inp[i])`.
    - Supports standard stack-based ops.
- **`HighDimOracle`**:
    - Computes correlations between output and individual input dimensions to guide BFS.
    - Encodes structure flags for used dimensions in the state vector.

### Phase 2: Experiment Script Implementation
Create `hpm_ai_v2/experiments/experiment_sp100_high_dim_scaling.py`.

- **Data Generation**: Implement $y = a t^2 + b \sin(\theta) + \text{noise}$ with D-dimensional inputs including:
    - $t, \theta$ (True)
    - $y + \text{noise}$ (Spurious shortcut)
    - $t + \text{noise}$, $\sin(t) + \text{noise}$, $t^2 + \text{noise}$ (Distractors)
    - Random noise for remaining dimensions.
- **`run_scaling_test(D)`**:
    1. Initialize agent with `HighDim` components.
    2. Phase 1: Discovery under D dimensions.
    3. Phase 2: Audit macro for true variable usage.
    4. Phase 3: Transfer to different coefficients ($a, b$).
    5. Phase 4: Feature permutation robustness.

### Phase 3: Scaling Curve Execution
- Run for $D \in [10, 50, 100, 200, 500]$.
- Measure Structure Recovery Rate, BFS time, and Transfer success.

## Key Files
- `hpm_ai_v2/domains/highdim_domain.py`
- `hpm_ai_v2/domains/highdim_renderer.py`
- `hpm_ai_v2/utils/oracle/highdim_oracle.py`
- `hpm_ai_v2/experiments/experiment_sp100_high_dim_scaling.py`

## Verification & Metrics
- **Metric 1: Structure Recovery Rate**: % of runs where macro uses only dimensions 0 and 1.
- **Metric 2: Transfer Efficiency**: Depth of solution in Phase 3 (should be 1).
- **Metric 3: Permutation Robustness**: Success rate stability after column shuffling.
- **Metric 4: Scaling Limit**: Point where BFS time becomes prohibitive or recovery rate drops below 60%.
