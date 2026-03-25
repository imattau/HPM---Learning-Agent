# Structured Agents HPM Completion Final Plan

## Status

Planning-only document. No implementation is included in this phase.

## Goal

Close the remaining HPM-framework gaps in the baseline `hpm` structured-agent stack and freeze that baseline before any HPM AI v2 branching work.

## Execution Principles

- Use `StructuredOrchestrator` as the forward path.
- Treat `HierarchicalOrchestrator` as legacy support only unless a regression test forces compatibility work.
- Keep changes local to the existing `hpm` line.
- Require gate passing before advancing to the next milestone.

## Sequenced Roadmap

1. Freeze the completion checklist.
- Confirm the remaining gaps are deterministic seeding, developmental dynamics, evaluator plurality, field coupling, cross-scale identity continuity, and replay-grade traces.
- Record pass/fail criteria for each gap.

2. Finalize contracts and config surface.
- Keep the already-introduced contracts stable: `PatternIdentity`, `PatternState`, `EvaluatorVector`, `FieldConstraint`, `MetaEvaluatorState`, and `DecisionTrace`.
- Add or retain config knobs for lifecycle decay/consolidation, evaluator arbitration, and meta-update rate.

3. Verify deterministic control.
- Ensure recombination and any other stochastic subpath consume the configured seed path.
- Define explicit exceptions only for components that own a private RNG by design.

4. Complete developmental dynamics.
- Verify lifecycle states (`emergent`, `stable`, `decaying`, `retired`) are surfaced consistently.
- Confirm controlled forgetting, reinforcement, and reactivation are measurable.

5. Complete evaluator plurality and arbitration.
- Verify predictive/coherence/cost/horizon evaluator channels are exposed.
- Validate arbitration modes (`fixed`, `adaptive`, `bandit`) and their update behavior.

6. Complete field coupling.
- Ensure field constraints are accepted, stored, and applied to routing or selection decisions.
- Reject any design that records constraints without making them causal.

7. Complete identity continuity.
- Validate parent IDs and identity lineage across recombination and structured-level transitions.
- Specify how cross-level identity snapshots are projected through structured handoffs.

8. Complete meta-evaluation.
- Define the rolling outcome signal and update cadence.
- Ensure trace-visible weight drift is caused by outcome quality rather than a proxy-only score.

9. Complete replay-grade traceability.
- Emit trace records with selected pattern IDs, parent IDs, constraint IDs, evaluator state, and signal provenance.
- Keep both inline trace summaries and a sink-capable trace path available.

10. Benchmark integration.
- Add comparison cases for baseline structured agents vs completion-enhanced agents.
- Require deltas for seed stability, lineage integrity, evaluator drift, and trace completeness.

11. Gate review and freeze.
- Do not begin HPM AI v2 branching until all completion gates pass.

## Milestones

- M1: Contract and config surface finalized.
- M2: Deterministic control and lifecycle behavior verified.
- M3: Field coupling and identity continuity verified.
- M4: Meta-evaluator and trace replay verified.
- M5: Benchmark gates pass and the structured baseline is frozen for v2 branching.

## Acceptance Gates

- Gate A: Fixed-seed runs are reproducible across all targeted stochastic paths.
- Gate B: Identity lineage survives recombination and level transitions.
- Gate C: Evaluator channels show non-trivial causal contribution.
- Gate D: External constraints measurably alter decisions on stress tasks.
- Gate E: Meta-evaluator responds to rolling outcome signals.
- Gate F: Decision traces are sufficient for control-path replay.
- Gate G: No regression on the existing structured benchmark suite.

## Risks and Controls

- Risk: Overstating reproducibility where a component still owns its own RNG.
Control: declare and test each exception explicitly.

- Risk: Lineage fields remain local-only.
Control: require explicit cross-level trace checks.

- Risk: Meta-evaluator updates remain proxy-driven.
Control: separate outcome signal design from observable metrics.

- Risk: Legacy hierarchy path becomes a distraction.
Control: keep `HierarchicalOrchestrator` out of the completion work unless a failing regression needs it.

## Exit Criteria

The completion phase ends when all gates pass and the structured-agent baseline is formally marked HPM-complete for v2 branching.
