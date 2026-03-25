# Structured Agents Composite Promotion Plan

## Status

Planning-only document. No implementation is included in this phase.

## Goal

Add a bounded composite-promotion layer above the existing task decomposition system so repeated composites can become reusable patterns. This is the HPM-style “patterns building on patterns” step: a composite formed from parts should be eligible for promotion, reuse, decay, and retirement across tasks.

## Execution Principles

- Keep the current decomposition layer as the source of truth for parts and candidate composites.
- Add promotion as a separate ledger/memory layer, not as a rewrite of the extractor.
- Preserve the existing structured benchmark path and compare against it.
- Require explicit lineage and traceability for every promoted pattern.
- Keep promotion thresholds configurable so the layer can be tuned without changing extractor semantics.

## Ownership Boundaries

- Extraction and part graph construction stay in `hpm/decomposition.py`.
- Promotion bookkeeping lives in a dedicated promotion module or ledger, but can be backed by the existing contextual/tiered store stack and archival data.
- Candidate scoring in benchmarks may consume promoted patterns, but benchmark code must not become the only place where promotion exists.
- Selection and lifecycle integration remain with the existing structured-agent stack.

## Sequenced Roadmap

1. Freeze the promotion vocabulary.
- Define `CompositeOccurrence`, `PromotedPattern`, `PromotionRule`, `PromotionTrace`, and `PromotionLedger`.
- Keep the schema minimal and compatible with the current decomposition result shape.
- Define explicit handling for ambiguous and conflicting composites.

2. Build occurrence tracking.
- Record every candidate composite with task and trace provenance.
- Add stable composite signatures so repeated patterns can be matched across tasks.
- Use the contextual/tiered store and archived contexts to accumulate recurrence evidence.
- Preserve rejected composites as first-class records so promotion can be audited.

3. Add promotion rules.
- Promote composites when support, stability, and reuse exceed thresholds.
- Require explicit lineage to source parts and prior composites.
- Add demotion/retirement when support falls or contradictions accumulate.

4. Enable reuse in scoring.
- Allow promoted patterns to contribute to candidate assembly and ranking.
- Compare baseline decomposition scoring with promotion-aware scoring.
- Keep the existing decomposition scorer intact so comparisons remain causal.

5. Wire trace and ledger reporting.
- Emit promotion traces alongside decomposition traces.
- Report promotion rate, reuse rate, retirement rate, and downstream lift.
- Persist promotion ledgers through the store-backed archival flow where practical.
- Distinguish raw composites from promoted composites in benchmark outputs.

6. Validate against current structured baselines.
- Run promotion-disabled vs promotion-enabled comparisons on ARC first.
- Add Math and Reber only after the ARC path demonstrates traceable reuse.
- Ensure the existing structured benchmark suite still passes.

## Milestones

- M1: Shared promotion contracts and ledger interfaces defined.
- M2: Composite occurrence tracking implemented for ARC.
- M3: Promotion and reuse wired into candidate scoring.
- M4: Trace and benchmark reporting added.
- M5: Cross-domain promotion support added for Math and Reber.

## Acceptance Gates

- Gate A: Repeated composites are recorded with stable signatures and lineage.
- Gate B: At least one promoted pattern is reused in a later candidate assembly on a controlled ARC slice, and recurrence is visible through a store-backed ledger.
- Gate C: Promotion decisions are thresholded and explainable.
- Gate D: Promotion-enabled scoring changes downstream ranking or retention on at least two distinct task slices or seeds.
- Gate E: Traces show promoted, retained, rejected, and retired composite states.
- Gate F: Existing structured benchmarks continue to run unchanged when promotion is disabled.

## Risks and Controls

- Risk: Promotion collapses into ad hoc caching.
Control: require explicit thresholds, decay, and retirement rules.

- Risk: Promotion becomes a benchmark-only trick.
Control: keep the ledger in core code, backed by the existing contextual/tiered store flow, and let benchmarks consume it, not define it.

- Risk: Overpromotion of noisy composites.
Control: require reuse across tasks and a minimum lift signal before promotion.

- Risk: Promotion duplicates decomposition instead of extending it.
Control: treat promoted patterns as higher-order inputs to future assembly, not as a parallel extractor.

## Exit Criteria

The plan ends when the system can promote repeated composites into reusable patterns, reuse those promoted patterns in later tasks, and report the effect on structured benchmarks.
