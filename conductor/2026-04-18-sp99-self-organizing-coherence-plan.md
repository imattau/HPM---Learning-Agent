# SP99: Self-Organizing Coherence

## Objective
Prove that HPM's preference for true structural invariants over spurious shortcuts can emerge naturally through recurring success and native Observer dynamics, without manual intervention or "engineered" weights.

## Phased Implementation Plan

### Phase 1: Native Reinforcement Utility
Instead of manual weight injection (`_set_state_field`), we implement a mechanism to present successful solutions back to the Observer.

- **`reinforce_solve(agent, path, inputs, outputs)`**:
    1. Render the `path` to code.
    2. Compute the empirical delta state using the `Oracle`.
    3. Construct a full fractal vector `x = [0 | concept_vec | delta_vec]`.
    4. Call `agent.observer.observe(x)`.
    5. This triggers `DecisionPolicy.weight_update`, naturally increasing the node weight of the explaining macro.

### Phase 2: Multi-Contextual Stabilization (The Training)
Show that a physical invariant naturally gains "Coherence" (weight) when it consistently explains diverse phenomena.

- **Tasks**:
    1. **T1: Standard Drop** ($y = 0.5 g t^2$).
    2. **T2: Moon Drop** ($g = 1.62$).
    3. **T3: Jupiter Drop** ($g = 24.79$).
- **Process**: 
    1. Agent discovers the $t^2$ macro in T1.
    2. After each task, call `reinforce_solve`.
    3. Audit: Confirm `motif_vertical_drop` weight increases (e.g., 0.1 -> 0.4 -> 0.7).

### Phase 3: The Blind Ambiguity Test (The Proof)
The ultimate test of SP98 Strong Form, but with natural weights.

- **Task**: **Confounded Drop** (Vertical drop with Distorted $Z_1$ shortcut).
- **Condition**: Accuracy Parity (Z1 error $\approx$ Physics error).
- **Selection**: The agent uses **Unified Utility Selection** ($Utility = Accuracy + Weight$).
- **Success Criteria**: The agent must prefer the physical invariant because its *naturally learned* weight (e.g. 0.7) is higher than the shortcut's default prior weight (0.5).

### Phase 4: Final Validation
- Run 5 seeds.
- Compare "Shortcut chosen" vs "Invariant chosen".
- Confirm 100% preference for invariant without any manual weighting.

## Key Files
- `hpm_ai_v2/experiments/experiment_sp99_self_organizing_coherence.py` (New experiment script).
- `hpm_ai_v2/agents/base_agent.py` (Verify Unified Utility Selection is active).

## Verification & Metrics
- **Metric 1: Stabilization Curve**: Plot Node Weight vs. Number of Tasks.
- **Metric 2: Preference Accuracy**: % of seeds choosing invariant under parity.
- **Metric 3: Zero-Shot Transfer**: Successful reuse in a final unseen task.
