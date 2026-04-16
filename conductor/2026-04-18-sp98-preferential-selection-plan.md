# SP98: Preferential Structure Selection

## Objective
Prove that HPM prefers true invariant structure over spurious correlations *even when both work*, demonstrating that its "Physical Intelligence" transcends simple statistical fitting.

## Design: Competitive Selection
- **True Structure**: $y = 0.5 g t^2$.
- **Spurious Shortcut**: $z_1 = 0.5 g t^2 + 0.2 \sin(t) + \text{noise}$.
- **Tension**: 
  - $z_1$ is **simpler** (1 node vs 6 nodes).
  - True structure is **more accurate** and has **higher scientific correlation** with $t^2$.
- **Constraint**: Set agent tolerance high enough (e.g., 0.3) so both are technically "valid."

## Implementation Steps

### 1. Data Generation
- Implement `generate_preference_data` where $z_1$ is distorted by $\sin(t)$.
- Ensure $z_1$ remains highly correlated but has higher empirical error than the true physics.

### 2. Experimental Phases
- **Phase 1: Task 1 Discovery**: Solve Vertical Drop with $z_1$ available. Observe the choice.
- **Phase 2: Pattern Audit**: Verify if the agent selected the 6-node structure or the 1-node $z_1$.
- **Phase 3: Task 2 Transfer**: Solve a scaled task ($y = 2.0 g t^2$) and measure if the Phase 1 solution is reused.
- **Phase 4: Statistical Stability**: Run across 5 different noise seeds to confirm the preference is not accidental.

## Verification & Testing
- **Selection Ratio**: Measure how many runs out of 5 choose the invariant.
- **Transfer Utility**: Compare the number of oracle calls for Task 2 when $t^2$ was chosen vs when $z_1$ was chosen (using a control run if needed).
- **Verdict Logic**: `SP98 VALIDATED` if the agent consistently chooses the invariant and demonstrates its reuse potential.
