# Revised SP67: Multi‑Agent Structural Recombination with Full HPM Abstraction Stack

## Objective
Implement SP67 to demonstrate a multi-agent extension of the full HPM abstraction stack (L1-L5). It ensures each phase corresponds to a distinct HPM level, from L1 sensory primitives through L5 meta-patterns, with social pattern fields as a horizontal extension that strictly respects fractal uniformity (storing social state as HFN nodes).

## Updated Abstraction Levels in SP67

| Phase | Level | What is demonstrated | HPM reference |
|-------|-------|----------------------|---------------|
| 1 | L1 + L2 | Individual schema acquisition (nested domains) | §7.4.1–7.4.2 |
| 2 | L3 | Agents discover **relational invariant** (list iteration) from exchanged macros | §7.4.3 |
| 3 | L4 | Forward model predicts partner’s macro behaviour before sharing | §7.4.4 |
| 4 | Social field | Shared forest accelerates convergence on novel tasks | §9.5 |
| 5 | Appendix E | Structural recombination with insight reward (L4‑generative) | Appendix E |
| 6 | Social field | Blackboard reduces spurious pattern persistence | §9.7 |
| 7 | L5 | Meta‑controller prioritises social strategies | §7.4.5 |
| 8‑10 | – | AST substitution, learnability robustness (from SP66) | – |

## Architectural Changes

### 1. L3 Relational Pattern Discovery
After exchanging macros, agents run `discover_meta_schema()` on the union of both agents' macros to produce an L3 `meta_schema` node (e.g., `meta_list_iteration` representing the common `[VAR_INP, LIST_INIT, FOR_LOOP, ITEM_ACCESS]` prefix).

### 2. L4 Forward Model for Social Exchange
Before sharing a macro, an agent uses its L4 state transition model to predict the macro's effect on a probe input. Sharing only occurs if the prediction error is low (< 0.1), demonstrating mental simulation of another agent's pattern without execution.

### 3. Fractal Uniformity for Social State
`SocialForest` stores exchange logs and blackboard failures as native HFN nodes in dedicated `TieredForest` instances (`_exchange_forest`, `_blackboard_forest`), rather than Python lists/dicts.

## Implementation Steps

### Phase 1: Update `sp67_social.py`
- [ ] Refactor `SocialForest` to use `TieredForest` for `_exchange_forest` (D=4) and `_blackboard_forest` (D=4).
- [ ] Implement `log_exchange` and `post_failure` using HFN nodes.
- [ ] Update `SocialAnalogicalAgent` to integrate the `StateTransitionModel` (L4).
- [ ] Add `should_share(macro, probe_input)` to `SocialAnalogicalAgent` utilizing the forward model.

### Phase 2: Update `experiment_sp67_social_recombination.py`
- [ ] Implement the revised 10-phase curriculum.
- [ ] **Phase 1**: Acquire nested domain macros for Alice and Bob.
- [ ] **Phase 2**: Social exchange and L3 relational invariant discovery (`meta_list_iteration`).
- [ ] **Phase 3**: Validate L4 forward model prediction error < 0.1 prior to sharing.
- [ ] **Phase 4**: Social Convergence Test (compare oracle calls with an isolated control).
- [ ] **Phase 5**: Recombination Insight (structural recombination of macros to form new atomic ops).
- [ ] **Phase 6**: Blackboard scaffolding (spurious rule avoidance).
- [ ] **Phase 7**: L5 Meta-Controller strategy prioritization (`social_exchange` selection).
- [ ] **Phases 8-10**: AST substitution, learnability, density (SP66 regressions).

### Phase 3: Validation
- [ ] Run the experiment script.
- [ ] Ensure all success conditions are met and output explicitly reports the L1-L5 HPM alignment.
