# Structured Agents Task Decomposition Plan

## Status

Planning-only document. No implementation is included in this phase.

## Goal

Add explicit task decomposition and part-to-pattern assembly to `structured_agents` so tasks are represented as reusable parts that can compose into higher-order patterns. The decomposition layer should be a bounded, domain-specific module that feeds the existing structured-agent stack rather than a replacement parser.

## Execution Principles

- Keep the existing `structured_agents` stack as the baseline.
- Add explicit part graphs before adding richer compositional search.
- Use bounded, domain-specific extractors instead of a universal parser.
- Preserve compatibility with current benchmark and trace contracts.
- Keep ownership explicit: extraction and graph-building live in a decomposition module, while selection and lifecycle remain in the existing agent/orchestrator path.

## Sequenced Roadmap

1. Freeze the decomposition vocabulary.
- Define `TaskAtom`, `TaskPart`, `PartRelation`, `CompositePattern`, `HypothesisCandidate`, and `AssemblyTrace`.
- Keep the schema minimal and shared across domains.
- Define explicit handling for `unassigned_atoms` and `ambiguous_parts`.

2. Add domain extractors.
- ARC: extract object, spatial, symmetry, and transformation parts.
- Math: extract term, operator, and family parts.
- Reber: extract prefix, state, and transition parts.
- Each extractor must emit partial coverage explicitly when decomposition is uncertain.

3. Build part graphs.
- Connect parts with typed, weighted relations.
- Preserve provenance from atom to part to relation.

4. Assemble candidate composites.
- Generate candidate composites from related parts.
- Score candidates by coherence, support, novelty, and simplicity.
- Keep rejected composites and low-confidence assemblies in the trace for inspection.

5. Integrate with lifecycle and selection.
- Promote strong composites into normal pattern lifecycle handling.
- Retire weak or contradictory composites.

6. Add trace and benchmark support.
- Emit assembly traces with part lineage, scoring breakdowns, and ambiguity markers.
- Add benchmark gates for coverage, assembly success, rejection rate, and causal lift.

7. Validate against current structured baselines.
- Compare part-aware scoring to existing structured-agent scoring.
- Ensure no regression in current structured benchmark paths.

## Milestones

- M1: Shared contracts and extractor interfaces finalized.
- M2: ARC part extraction and graph construction implemented in a dedicated decomposition module.
- M3: Composite assembly and retention wired into lifecycle handling.
- M4: Traceability and benchmark coverage complete.
- M5: Cross-domain extractor support added for Math and Reber.

## Acceptance Gates

- Gate A: Every supported task produces explicit parts and relations.
- Gate B: Candidate composites can be traced to source parts.
- Gate C: Part-aware scoring changes at least one downstream benchmark decision on a controlled slice.
- Gate D: Part decomposition reports coverage, rejection, and ambiguity rates explicitly.
- Gate E: Existing structured benchmarks still pass.
- Gate F: Trace output can explain why a composite was retained, rejected, or marked ambiguous.

## Risks and Controls

- Risk: The part schema becomes too abstract to be useful.
Control: keep extractors domain-specific and bounded.

- Risk: The system becomes a symbolic parser project.
Control: require every part to feed the existing pattern lifecycle and scoring paths, and keep the extractor boundaries domain-specific.

- Risk: Parts remain descriptive only.
Control: require causal benchmark impact and retention/rejection evidence before expanding the vocabulary.

- Risk: The decomposition layer duplicates the encoder stack.
Control: keep the decomposition layer above encoders as explicit compositional structure, not another embedding model.

## Exit Criteria

The plan ends when explicit task parts and composite assembly are available for the structured-agent baseline and the benchmark suite can measure their effect.
