# Structured Agents HPM Completion Final Spec

## Purpose

Define the final HPM-completion target for the baseline `hpm` structured-agent stack before any HPM AI v2 branching work.

## Canonical Baseline

- The canonical implementation target is `StructuredOrchestrator` and its supporting agent, field, lifecycle, and completion contracts.
- `HierarchicalOrchestrator` is legacy support only. It is out of scope unless a regression test explicitly requires parity for backward compatibility.
- The completion target must not introduce a parallel architecture. It must close HPM gaps inside the existing structured-agent line.

## Scope

In scope:
- deterministic seeding across all stochastic control paths, including recombination
- developmental dynamics: consolidation, decay, forgetting, and reactivation tracking
- evaluator plurality with explicit arbitration and adaptive meta-evaluation
- field coupling through explicit constraints that can influence routing and selection
- cross-scale identity continuity across structured-level transitions
- replay-grade traces with causal inputs and control-state provenance
- benchmark gates for the above

Out of scope:
- HPM AI v2 expansion work
- free-form inter-agent language in the critical path
- domain-specific shortcuts embedded into core contracts
- changes to `HierarchicalOrchestrator` beyond legacy compatibility needs

## HPM Completion Targets

1. Determinism.
- Every stochastic path must respect the configured seed unless a component explicitly declares its own private RNG.
- Recombination, initialization, and benchmark execution must be reproducible under fixed seed.

2. Developmental dynamics.
- Pattern states must model lifecycle progression, consolidation, decay, and reactivation.
- Absence and recovery must be visible in metrics and traces.

3. Evaluator plurality.
- The agent must expose distinct evaluator channels for predictive, coherence, cost, and horizon pressure.
- Arbitration must support fixed, adaptive, and bandit-style control modes.
- Adaptive meta-evaluation must be driven by rolling outcome quality rather than instantaneous proxy score alone.

4. Field coupling.
- `PatternField` must accept explicit constraints and use them in scoring or routing decisions.
- Constraints must be traceable and causal, not merely recorded as metadata.

5. Identity continuity.
- Pattern identity must persist across recombination, pruning, sharing, and structured-level transitions.
- Lineage must represent parentage, layer origin, and step provenance.
- Cross-level identity projection must be explicit rather than inferred from shared UUIDs.

6. Traceability.
- Decisions must carry selected patterns, parent IDs, constraints, evaluator state, and the outcome signal that changed control state.
- Trace records must be serializable and suitable for replay or offline inspection.
- The system must support both inline summary traces and a sink-capable trace store.

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
- `AgentConfig.seed` must govern initial seeding and recombination unless explicitly overridden.
- No hardcoded RNG seeds may remain in the default control path.

2. Lifecycle and lineage.
- Existing lifecycle transitions must remain supported.
- Recombined or promoted patterns must retain parentage in lineage data.
- Identity maps must traverse structured-level handoffs explicitly.

3. Meta-evaluation control.
- The meta-evaluator must ingest a rolling outcome signal derived from task success, conflict, and recovery quality.
- Adaptive weight updates must be visible in the trace stream.
- Fixed mode must remain a no-op baseline.

4. Decision traces.
- Each `step()` result must include enough information to reconstruct the control choice without rerunning the full episode.
- Traces must identify which patterns and constraints influenced the result.

5. Benchmark comparability.
- New gates must compare baseline structured agents against the completion-enhanced path using fixed seeds.
- Benchmark outputs must report deltas for seed stability, lineage integrity, evaluator drift, and trace completeness.

## Non-Functional Requirements

- Backward compatible defaults for existing callers.
- Typed schemas for all new completion contracts.
- Deterministic behavior under fixed seed.
- Minimal surface-area expansion beyond the structured-agent baseline.

## Benchmark and Validation Requirements

1. Seed reproducibility.
- Same config seed must produce identical initialization and recombination decisions in the same scenario.

2. Lineage preservation.
- Recombined patterns must carry parent IDs through observation, store updates, and trace emission.

3. Meta-evaluation drift.
- Successive positive and negative outcome signals must produce measurable weight drift.

4. Trace replay.
- A decision trace must be sufficient to identify selected inputs, constraints, and control state.

5. Regression suite.
- Existing structured-agent and benchmark tests must remain green.

## Acceptance Criteria

- AC1: All stochastic paths respect the configured seed or declare an explicit exception.
- AC2: Cross-level lineage data is represented consistently in the trace and identity contracts.
- AC3: Meta-evaluator state changes are attributable to rolling outcome signals.
- AC4: Decision traces are replayable at the level of control inputs and selected patterns.
- AC5: No regression on the existing structured benchmark suite.

## Delivery Artifacts

- Final spec and final plan documents.
- Test matrix for seed, lineage, meta-evaluation, and replay.
- Benchmark comparison template for baseline vs completion-enhanced agents.

## Status

Final specification. Implementation is intentionally excluded from this document.
