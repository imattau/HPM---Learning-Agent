# Structured Agents HPM Completion Spec v2

## Purpose

Close the remaining HPM-framework gaps identified in the current structured-agent implementation, without changing implementation behavior yet.

## Scope

In scope:
- deterministic recombination and initialization seeding
- cross-scale identity continuity across structured/hierarchical levels
- outcome-driven meta-evaluation adaptation
- replayable decision traces with causal inputs
- benchmark gates for the above

Out of scope:
- HPM AI v2 expansion work
- domain-specific shortcuts
- free-form inter-agent language in the critical path
- implementation changes in this phase

## Gap-Driven Targets

1. Deterministic control.
- All stochastic decision paths must respect explicit config seeds.
- Recombination, lifecycle updates, and benchmark runs must be reproducible under fixed seed.

2. Identity continuity.
- Pattern identity must persist across recombination, pruning, sharing, and level transitions.
- Lineage must represent parentage, layer origin, and step provenance.

3. Meta-evaluation.
- Evaluator adaptation must be driven by rolling outcome quality, not only instantaneous proxy signals.
- The meta-evaluator must expose current weights and update events.

4. Traceability.
- Decisions must carry the selected patterns, constraint references, evaluator vector, lineage references, and the outcome signal that changed the control state.
- Trace records must be serializable and capturable through a trace sink for later replay.

5. HPM alignment.
- The stack must preserve pattern substrates, pattern dynamics, evaluators, and field interactions without turning the completion layer into a parallel architecture.

## Required Data Contracts

1. `PatternIdentity`
- `id`
- `parent_ids`
- `layer_origin`
- `created_at`
- `last_seen_at`
- `source_step`
- `lineage_kind`

2. `PatternState`
- `identity_id`
- `lifecycle_state`
- `stability`
- `decay_rate`
- `reinforcement_count`
- `absence_count`
- `reactivation_count`

3. `MetaEvaluatorState`
- `mode`
- `weights`
- `last_signal`
- `update_count`
- `signal_source`

4. `DecisionTrace`
- `trace_id`
- `selected_pattern_ids`
- `selected_parent_ids`
- `evaluator_vector`
- `constraint_ids`
- `meta_evaluator_state`
- `signal_source`
- `action`

5. `FieldConstraint`
- `constraint_type`
- `scope`
- `strength`
- `source`
- `timestamp`

## Functional Requirements

1. Seed propagation.
- `AgentConfig.seed` must govern all stochastic subpaths unless a component explicitly declares its own seed.
- Recombination must not silently use a hardcoded RNG seed.

2. Lifecycle and lineage.
- Existing lifecycle transitions remain supported.
- Parentage from recombination or promoted patterns must be retained in the lineage data model.
- Cross-level identity projections must be explicit, not inferred from shared UUIDs alone.
- Absence and reactivation counters must be tracked as separate lifecycle signals.

3. Meta-evaluation control.
- The meta-evaluator must ingest a rolling outcome signal derived from task success, conflict, and recovery quality.
- Adaptive weight updates must be visible in the trace stream.
- Fixed mode must remain a no-op baseline.

4. Decision traces.
- Every `step()` result must include enough information to reconstruct the control choice without replaying the whole agent.
- Traces must identify which patterns and constraints influenced the final outcome.

5. Benchmark comparability.
- New gates must compare baseline structured agents against the completion-enhanced path using fixed seeds.
- Benchmark output must include delta reporting for the new control signals.

## Non-Functional Requirements

- Backward compatible defaults for existing callers.
- Deterministic behavior under fixed seed.
- Minimal surface area changes before implementation.
- Typed schemas for all new completion contracts.

## Benchmark and Validation Requirements

1. Seed reproducibility test.
- Same config seed must produce identical initialization and recombination decisions in the same scenario.

2. Lineage preservation test.
- Recombined patterns must carry parent IDs through observation, store updates, and trace emission.

3. Meta-evaluation drift test.
- Successive positive and negative outcome signals must produce measurable weight drift.

4. Trace replay test.
- A decision trace must be sufficient to identify selected inputs, constraints, and control state.

5. Regression suite.
- Existing structured-agent and benchmark tests must remain green.

## Acceptance Criteria

- AC1: All stochastic paths respect the configured seed or declare an explicit exception.
- AC2: Cross-level lineage data is represented consistently in the trace and identity contracts.
- AC3: Meta-evaluator state changes are attributable to outcome signals, not just proxy metrics.
- AC4: Decision traces are replayable at the level of control inputs and selected patterns.
- AC5: No regression on existing structured-agent benchmarks.

## Delivery Artifacts

- Updated spec and plan artifacts.
- Test matrix for seed, lineage, meta-evaluation, and replay.
- Benchmark comparison template for baseline vs completion-enhanced agents.

## Status

Draft specification recorded. No implementation changes included.
