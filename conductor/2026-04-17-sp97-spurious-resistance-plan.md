# SP97: Spurious Correlation Resistance

## Objective
Prove that HPM prefers true underlying physical structure over spurious statistical correlations when both are present, and can recover from "hallucinated" correlations under distribution shift.

## Key Files & Context
- `hpm_ai_v2/domains/projectile_domain.py`: Add spurious variable concepts ($z_1, z_2, z_3$).
- `hpm_ai_v2/domains/projectile_renderer.py`: Update rendering for spurious variables.
- `hpm_ai_v2/experiments/experiment_sp97_spurious_resistance.py`: New multi-phase experiment script.

## Implementation Steps

### 1. Domain & Renderer Upgrades
- Extend `PROJECTILE_CONCEPTS` with `VAR_Z1`, `VAR_Z2`, `VAR_Z3`.
- Update `ProjectileRenderer` to push values from input indices 4, 5, 6 respectively.

### 2. Data Generation (SP97 Specific)
- **Confounded Data**: $z_1 \approx y$ (Target), $z_2 \approx t$ (Variable), $z_3 = \text{random}$.
- **Shifted Data**: $z_1, z_2, z_3$ all disconnected from physics (pure noise).

### 3. Experimental Phases
- **Phase 1: Confounded Discovery**: Solve the Vertical Drop task ($y = 0.5 g t^2$) with all primitives (including $z_1, z_2, z_3$). 
- **Phase 2: Distribution Shift**: Take the solution from Phase 1 and evaluate it on Shifted Data.
- **Phase 3: Structural Recovery**: Re-run discovery on Shifted Data. 
- **Phase 4: Comparative Verification**: Verify that the final converged pattern uses $t^2$ and ignores $z_1$.

## Verification & Testing
- **Pattern Audit**: Check if the code for the Phase 1 solution uses `VAR_Z1` (likely) vs `OP_SQUARE`.
- **Performance Delta**: Measure the accuracy drop between Confounded and Shifted data for the $z_1$-based model.
- **Recovery Success**: Confirm that the agent successfully discovers the correct $t^2$ structure after the spurious correlation fails.
- **Verdict Logic**: `SP97 VALIDATED` if the agent eventually converges on the invariant structure despite the presence of "easy" spurious alternatives.
