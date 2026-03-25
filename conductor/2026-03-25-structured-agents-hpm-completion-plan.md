# Structured Agents HPM Completion Plan

## Status

Planning-only document. No implementation is included in this phase.

## Goal

Close the remaining HPM-framework gaps in the baseline `hpm` structured-agent stack before starting HPM AI v2 expansion work.

## Plan First: Sequenced Roadmap

1. Define completion gates and invariants.
- Freeze the HPM alignment checklist for: developmental dynamics, evaluator plurality, external field coupling, cross-scale identity continuity, and meta-evaluation dynamics.
- Record pass/fail criteria for each gate.

2. Land data contracts and config surface.
- Introduce schemas for `PatternIdentity`, `PatternState`, `EvaluatorVector`, `FieldConstraint`, and `DecisionTrace`.
- Add config knobs for lifecycle decay/consolidation, evaluator arbitration, and meta-update rates.

3. Add developmental dynamics.
- Implement lifecycle states (`emergent`, `stable`, `decaying`, `retired`).
- Add controlled forgetting + retrieval reinforcement and consolidation windows.

4. Add evaluator plurality and arbitration.
- Implement predictive/coherence/cost/horizon evaluators.
- Add arbitration modes (`fixed`, `adaptive`, `bandit`) and aggregate scoring.

5. Add external field coupling.
- Extend `PatternField` with explicit constraint ingestion and constraint-indexed scoring.
- Ensure field constraints causally affect routing/selection decisions.

6. Add cross-scale identity continuity.
- Propagate stable pattern IDs across L1->L5 transformations.
- Add identity reconciliation/matching across steps and levels.

7. Add meta-evaluation dynamics.
- Add evaluator-weight adaptation based on rolling success/regret.
- Track evaluator drift and intervention events.

8. Instrumentation and traceability.
- Emit trace IDs, evaluator vectors, identity lineage, and constraint references on decisions.
- Add replayable failure traces.

9. Benchmark integration.
- Add dedicated tests for drift, constraints, identity continuity, evaluator ablation, and contradiction recovery.
- Run A/B against the current structured baseline.

10. Gate review and freeze.
- Require all acceptance criteria to pass before any v2 feature expansion.

## Milestones

- M1: Contracts + config + checklist approved.
- M2: Lifecycle + evaluator plurality operational.
- M3: Field constraints + identity continuity operational.
- M4: Meta-evaluator + trace replay operational.
- M5: Benchmark gates pass; baseline frozen for v2 deltas.

## Acceptance Gates

- Gate A: Lifecycle metrics reported and stable under long-horizon drift tests.
- Gate B: Evaluator channels show non-trivial causal contribution.
- Gate C: External constraints measurably alter decisions on stress tasks.
- Gate D: Cross-level identity continuity reaches threshold target.
- Gate E: Meta-evaluator outperforms fixed arbitration baseline.
- Gate F: No regression on existing structured benchmark suite.

## Risks and Controls

- Risk: Added complexity without measurable gain.
Control: enforce causal metrics and gate-based progression.

- Risk: Overfitting to one benchmark family.
Control: require cross-domain gate results.

- Risk: Message/protocol noise.
Control: strict schemas + provenance + confidence requirements.

## Exit Criteria

The completion phase ends when all gates pass and the structured-agent baseline is formally marked HPM-complete for v2 branching.
