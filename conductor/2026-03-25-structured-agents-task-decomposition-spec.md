# Structured Agents Task Decomposition Spec

## Purpose

Make task decomposition explicit in `structured_agents` so a task is represented as reusable parts that compose into higher-order patterns, matching the HPM idea that patterns build from subpatterns. The decomposition layer is an implementation of HPM's subpattern/recombination idea, not a claim that every domain shares the same fixed ontology.

## Scope

In scope:
- explicit task-part extraction from domain observations
- typed part graphs and composition links
- candidate composite pattern generation from parts
- scoring and retention of composed patterns
- traceability from task parts to selected patterns
- benchmark hooks for decomposition quality and downstream impact

Out of scope:
- a universal symbolic parser for every domain
- a single hardcoded ontology that all tasks must share
- a universal symbolic parser for every domain
- replacing existing encoders or pattern stores
- full mathematical recomposition beyond bounded composition and scoring

## HPM Alignment

The paper treats learning as discovery, stabilisation, manipulation, and recombination of hierarchical patterns. This spec makes that concrete by forcing the stack to represent tasks as parts that can be assembled into composites.

A task should be decomposed in the same way a dog can be decomposed into:
- legs
- head
- ears
- tail
- body

The implementation target is not literal dog parsing. The target is domain-specific part extraction that produces reusable components, then composes them into candidate higher-order patterns. The dog example is only an analogy for nested part composition, not a universal ontology.

## Core Idea

Every task should flow through five stages:
1. `TaskAtom` extraction from the raw observation.
2. `TaskPart` grouping from atoms.
3. `PartRelation` construction between parts.
4. `CompositePattern` assembly from part graphs.
5. `HypothesisCandidate` ranking and retention.

The key requirement is that parts are explicit objects, not implicit embedding dimensions.
Unassigned atoms or low-confidence regions must be represented explicitly as such rather than silently discarded.

## Data Contracts

1. `TaskAtom`
- `atom_id`
- `kind`
- `features`
- `confidence`
- `source`
- `trace_id`

2. `TaskPart`
- `part_id`
- `label`
- `atoms`
- `features`
- `confidence`
- `parent_part_ids`
- `trace_id`

3. `PartRelation`
- `source_part_id`
- `target_part_id`
- `relation_type`
- `weight`
- `evidence`
- `trace_id`

4. `CompositePattern`
- `pattern_id`
- `part_ids`
- `relation_ids`
- `assembly_rule`
- `score`
- `stability`
- `parent_pattern_ids`
- `trace_id`

5. `HypothesisCandidate`
- `hypothesis_id`
- `hypothesis_type`
- `source_part_ids`
- `predicted_transform`
- `support`
- `novelty`
- `confidence`
- `trace_id`

6. `AssemblyTrace`
- `trace_id`
- `task_id`
- `extracted_parts`
- `selected_relations`
- `candidate_patterns`
- `selected_pattern_ids`
- `rejection_reason`
- `score_breakdown`

## Functional Requirements

1. Explicit part extraction.
- Each supported benchmark domain must expose a part extractor.
- The extractor must produce typed parts with confidence and provenance.
- If a domain cannot support fine-grained parts, it must still emit coarse parts rather than skipping decomposition.

2. Part graph construction.
- Parts must be connected with typed relations such as `contains`, `matches`, `transforms_to`, `symmetry_of`, `copies`, `moves`, and `changes_color`.
- Relations must be weighted and traceable.

3. Composite assembly.
- The system must assemble candidate composites from related parts.
- Candidate assembly must prefer structurally coherent assemblies that reduce uncertainty.
- Composite candidates must preserve parentage back to their source parts.

4. Hypothesis generation.
- The stack must derive candidate hypotheses from part graphs, not only from raw embeddings.
- Hypotheses must compete under a bounded scoring rule.
- Hypotheses can be rejected, retained, or promoted into stored patterns.

5. Retention and decay.
- Composites must enter the normal lifecycle machinery.
- Weak or inconsistent composites must decay or be retired.
- Strong composites must become reusable patterns.

6. Traceability.
- Every selection decision must identify the parts that contributed to it.
- The trace must show why one composite was preferred over alternatives.
- Benchmark output must report decomposition coverage and assembly success.

## Domain Strategy

Domain extractors are allowed to be specialized, but the contracts must remain shared.

Examples:
- ARC: object parts, spatial relations, symmetry, color changes, translations.
- Math: term parts, operator parts, transformation family parts.
- Reber: prefix/suffix/state-transition parts.

## Non-Goals

- One universal part grammar for all domains.
- Full semantic understanding equivalent to a human ontology.
- Free-form natural language decomposition in the critical path.
- Replacing pattern selection with a pure symbolic planner.

## Acceptance Criteria

- AC1: Tasks are represented as explicit part graphs, not just vector summaries.
- AC2: Candidate composites preserve source-part lineage.
- AC3: Decomposition quality is measurable and benchmarkable.
- AC4: Recombined composites can improve downstream candidate ranking or retention on at least one controlled slice.
- AC5: Candidate graphs must preserve lineage and ambiguity information.
- AC6: Existing structured-agent baselines remain runnable.

## Benchmark Requirements

1. Part coverage.
- Report how much of the task input was assigned to explicit parts.

2. Assembly quality.
- Report how often part assemblies become retained composites, and how often they are rejected or marked ambiguous.

3. Causal impact.
- Compare baseline vs part-aware scoring on the same tasks.

4. Trace completeness.
- Confirm that final outputs can be traced back to parts and relations.

## Status

Specification only. No implementation is included here.
