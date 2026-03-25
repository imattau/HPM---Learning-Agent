# Structured Agents HPM Completion Spec

## Purpose

Formal specification for completing missing HPM-framework capabilities in baseline `hpm` structured agents.

## Scope

In scope:
- developmental dynamics over time
- evaluator plurality and arbitration
- external field coupling
- cross-scale identity continuity
- meta-evaluator adaptation
- traceable, replayable decision pipeline

Out of scope:
- domain-specific shortcuts
- free-form inter-agent language in critical path
- HPM AI v2 feature expansion before completion gates pass

## HPM Alignment Targets

1. Pattern substrates.
- Typed representations for atoms, relations, hypotheses, and identities must persist across levels.

2. Pattern dynamics.
- Patterns must support emergence, stabilization, decay, retirement, and recovery.

3. Pattern evaluators.
- Multiple evaluator families must jointly influence selection and action.

4. Pattern field.
- Field must include both internal pattern competition and explicit external constraints.

## Required Data Contracts

1. `PatternIdentity`
- `id`
- `parent_ids`
- `layer_origin`
- `created_at`
- `last_seen_at`

2. `PatternState`
- `identity_id`
- `lifecycle_state`
- `stability`
- `decay_rate`
- `reinforcement_count`

3. `EvaluatorVector`
- `predictive`
- `coherence`
- `cost`
- `horizon`
- `aggregate`
- `arbitration_mode`

4. `FieldConstraint`
- `constraint_type`
- `scope`
- `strength`
- `source`
- `timestamp`

5. `DecisionTrace`
- `trace_id`
- `selected_pattern_ids`
- `evaluator_vector`
- `constraint_ids`
- `action`

## Functional Requirements

1. Lifecycle engine.
- Support lifecycle transitions: `emergent` -> `stable` -> `decaying` -> `retired`.
- Support reactivation via retrieval reinforcement.

2. Forgetting and consolidation.
- Decay inactive patterns according to configurable half-life.
- Promote stable patterns based on repeated low-surprise support.

3. Evaluator plurality.
- Compute predictive/coherence/cost/horizon channels each step.
- Aggregate via configurable arbitration mode.

4. External field coupling.
- Ingest environment/task constraints as first-class signals.
- Include constraints in candidate scoring and selection.

5. Cross-level identity continuity.
- Preserve pattern identity across layer transforms.
- Resolve splits/merges with lineage tracking.

6. Meta-evaluation dynamics.
- Update evaluator weights from rolling outcome quality.
- Emit adaptation events in trace logs.

7. Observability.
- Every action decision must be reconstructable from trace artifacts.

## Non-Functional Requirements

- Backward-compatible default behavior when completion features are disabled.
- Deterministic behavior under fixed seed for benchmark comparability.
- Strict typed schemas for all inter-layer and inter-agent messages.

## Benchmark and Validation Requirements

1. Long-horizon drift test.
- Validate consolidation, decay, and recovery dynamics.

2. Constraint injection test.
- Validate decision changes due to explicit field constraints.

3. Identity continuity test.
- Validate cross-layer same-pattern tracking against known mappings.

4. Evaluator ablation matrix.
- Quantify causal value of each evaluator channel.

5. Contradiction shock recovery test.
- Validate replan/rollback stability under structural conflicts.

## Acceptance Criteria

- AC1: Lifecycle metrics are produced and pass thresholds.
- AC2: Evaluator plurality contributes measurable causal lift vs single evaluator.
- AC3: Field constraints cause measurable decision/routing changes.
- AC4: Identity continuity accuracy passes agreed threshold.
- AC5: Meta-evaluator outperforms fixed arbitration baseline.
- AC6: No regression on existing structured benchmark suite.

## Delivery Artifacts

- Spec-approved contracts and invariants.
- Plan-approved milestone and gate checklist.
- Benchmark matrix and reporting template for completion sign-off.

## Status

Draft specification recorded. Implementation intentionally deferred.
