# Structured Agents HPM Completion Plan v2

## Status

Planning-only document. No implementation is included in this phase.

## Goal

Close the remaining HPM-framework gaps in the baseline `hpm` structured-agent stack that are still unresolved after the previous completion slices, before any HPM AI v2 branching work.

## Plan First: Sequenced Roadmap

1. Freeze the gap checklist.
- Confirm the remaining gaps are: deterministic recombination seeding, cross-level identity continuity, outcome-driven meta-evaluation, and replay-grade traces.
- Record pass/fail criteria for each gap.

2. Extend the contract layer.
- Add `source_step` and `lineage_kind` to identity tracking.
- Add `absence_count` to lifecycle state tracking.
- Add `MetaEvaluatorState` and trace fields for signal provenance.
- Keep existing contracts backward compatible.

3. Make stochastic control reproducible.
- Route recombination and any new stochastic subpath through explicit config seeding.
- Define exception cases where a component legitimately owns a private RNG.

4. Define lineage propagation.
- Preserve parent IDs for recombined or promoted patterns.
- Emit parent IDs in decision traces and identity snapshots.
- Specify how identity maps traverse structured and hierarchical levels.
- Define what constitutes a split, merge, or continuation.

5. Define meta-evaluation semantics.
- Specify the rolling outcome signal inputs and the update cadence.
- Separate proxy metrics from the signal used to update evaluator weights.
- Keep fixed mode as an inert baseline.

6. Define trace replay requirements.
- Emit trace records with pattern IDs, parent IDs, constraint IDs, and meta-evaluator state.
- Specify the minimum information required to reconstruct the decision path.
- Choose whether replay data is returned inline, persisted to a sink, or both.

7. Add validation gates.
- Seed reproducibility gate.
- Lineage preservation gate.
- Meta-evaluation drift gate.
- Trace replay gate.
- Existing benchmark non-regression gate.

8. Benchmark integration.
- Add comparison cases for baseline structured agents vs completion-enhanced agents.
- Require deltas for seed stability, lineage integrity, and control drift.

9. Gate review and freeze.
- Prevent HPM AI v2 branching until all gates pass.

## Milestones

- M1: Gap checklist and contracts approved.
- M2: Seed and lineage semantics approved.
- M3: Meta-evaluation and trace semantics approved.
- M4: Benchmarks and gates approved.
- M5: Completion baseline frozen for v2 branching.

## Acceptance Gates

- Gate A: Fixed-seed runs are reproducible across the targeted stochastic paths.
- Gate B: Identity lineage survives recombination and level transitions.
- Gate C: Meta-evaluator weights move in the expected direction under positive/negative outcomes.
- Gate D: Decision traces are sufficient for control-path replay.
- Gate E: No regression on the existing structured-agent benchmark suite.

## Risks and Controls

- Risk: Overstating reproducibility where components still own independent RNGs.
Control: declare and test each exception explicitly.

- Risk: Lineage fields remain local-only and do not propagate across levels.
Control: require explicit cross-level trace checks.

- Risk: Meta-evaluator updates remain proxy-driven.
Control: separate outcome signal design from observable metrics.

## Exit Criteria

The completion phase ends when all gates pass and the structured-agent baseline is formally marked ready for v2 branching.
