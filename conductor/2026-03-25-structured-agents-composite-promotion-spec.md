# Structured Agents Composite Promotion Spec

## Purpose

Make repeated composites explicit in `structured_agents` so patterns can build on patterns. The decomposition layer already yields task parts and candidate composites; this spec adds the next HPM step: promote stable composites into reusable higher-order patterns when they recur across tasks and improve downstream scoring.

This is a bounded promotion layer, not a universal symbolic parser and not a claim that every domain shares the same fixed ontology.

## Scope

In scope:
- cross-task composite tracking
- store-backed composite ledgering and warm-start support
- promotion of stable composites into reusable patterns
- lineage preservation from atoms -> parts -> composites -> promoted patterns
- composite reuse in candidate scoring and retention
- explicit rejection/retirement of weak or contradictory composites
- benchmark hooks for promotion rate, reuse rate, and downstream lift

Out of scope:
- full symbolic recomposition of arbitrary task semantics
- a universal part grammar across all domains
- replacing the current encoders or decomposition layer
- free-form natural language reasoning in the critical path
- long-horizon memory systems unrelated to pattern promotion

## HPM Alignment

HPM describes learning as the formation, stabilisation, recombination, and re-use of patterns at increasing levels of abstraction. This spec targets the missing middle layer: once task parts assemble into composites, some composites should themselves become reusable patterns.

The dog example remains an analogy only:
- legs
- head
- ears
- tail
- body
- dog

The implementation target is not literal animal parsing. The target is repeated structural reuse: a composite built from parts should later behave like a part for a larger composite.

## Core Idea

The system should flow through six stages:
1. `TaskAtom` extraction.
2. `TaskPart` grouping.
3. `PartRelation` construction.
4. `CompositePattern` assembly.
5. `CompositePromotion` into cross-task reusable patterns.
6. `PromotedPattern` reuse in later assembly and scoring.

Promotion is justified when a composite:
- appears repeatedly across tasks,
- maintains stable parentage,
- remains coherent under bounded re-assembly,
- and improves downstream ranking or retention.

Promotion state should be persisted through the existing store stack where possible, especially the contextual/tiered store path used for cross-task pattern retention. A dedicated ledger is still allowed, but it should integrate with store-backed archival, warm-start, and recurrence flows rather than living only inside benchmark memory.

## Data Contracts

1. `CompositeOccurrence`
- `occurrence_id`
- `task_id`
- `trace_id`
- `source_part_ids`
- `source_relation_ids`
- `candidate_pattern_id`
- `score`
- `coverage`
- `ambiguity_rate`
- `selected`

2. `PromotedPattern`
- `pattern_id`
- `label`
- `origin_composite_ids`
- `parent_part_ids`
- `promotion_count`
- `support`
- `stability`
- `retention_state`
- `last_seen_at`
- `trace_id`

3. `PromotionRule`
- `rule_id`
- `min_occurrences`
- `min_support`
- `max_ambiguity`
- `min_delta_lift`
- `retention_window`
- `trace_id`

4. `PromotionTrace`
- `trace_id`
- `task_id`
- `composite_id`
- `promotion_decision`
- `decision_reason`
- `support_breakdown`
- `lineage`
- `candidate_reuse`

5. `PromotionLedger`
- `ledger_id`
- `composite_ids`
- `promoted_pattern_ids`
- `retired_pattern_ids`
- `promotion_rules`
- `window_summary`

6. `StoreIntegration`
- `promotion_store_key`
- `archive_context_ids`
- `warm_start_pattern_ids`
- `recurrence_signature`
- `last_materialized_at`

## Functional Requirements

1. Composite occurrence tracking.
- Every candidate composite must be recorded with provenance.
- Recurrent composites across tasks must be linked by a stable signature.
- Ambiguous composites must remain explicit rather than being merged silently.

2. Promotion logic.
- A composite may be promoted when repeated evidence exceeds a configurable threshold.
- Promotion must preserve the original part lineage and trace references.
- Promotion must be reversible if later evidence shows contradiction or low utility.

3. Reuse in later tasks.
- Promoted patterns must be eligible as inputs to new composite assembly.
- Reused promoted patterns should influence candidate ranking and retention.
- Promotion should change the decision path, not just be recorded.

4. Retention and retirement.
- Patterns that stop recurring must decay.
- Contradictory or low-utility promoted patterns must be retired.
- Retirement must preserve history for auditability.

5. Traceability.
- Every promotion or rejection must be explainable by support, ambiguity, and reuse evidence.
- Benchmark output must show how many promoted patterns were reused.
- Traces must distinguish raw composites from promoted composites.

6. Store integration.
- Promotion records should be loadable from and writable to the existing contextual/tiered store ecosystem where possible.
- Warm-started contexts should be able to reuse promoted patterns as seed candidates.
- Archived task contexts should be able to contribute recurrence evidence to the promotion ledger.

## Domain Strategy

Domain extractors remain specialised. Promotion logic remains shared.

Examples:
- ARC: translation, symmetry, colour-change, object-structure composites.
- Math: operator families, transformation motifs, recurrence motifs.
- Reber: prefix/state-transition motifs and repeating transition clusters.

## Non-Goals

- A universal symbolic theory of all task semantics.
- A full latent-variable recomposition system.
- Natural language explanations as the primary representation.
- Replacing benchmark scoring with a pure memory lookup system.

## Acceptance Criteria

- AC1: Recurrent composites are recorded as promotable patterns with explicit lineage.
- AC2: Promoted patterns can be reused in later candidate assembly.
- AC3: Promotion decisions are tied to thresholds and support evidence.
- AC4: Promotion changes downstream ranking or retention on a controlled slice and remains positive across at least two distinct task slices or seeds.
- AC5: Promotion traces distinguish retained, rejected, and retired composites.
- AC6: Existing structured-agent benchmarks continue to run.
- AC7: Promotion state can be restored from store-backed context without losing lineage or recurrence evidence.

## Benchmark Requirements

1. Promotion rate.
- Report how many composites were promoted per task/window.

2. Reuse rate.
- Report how often promoted patterns were used in later assembly.

3. Downstream lift.
- Compare baseline decomposition vs promotion-enabled scoring.

4. Retirement rate.
- Report how often promoted patterns decayed or were retired.

5. Trace completeness.
- Confirm every promoted pattern can be traced back to source parts and occurrences.

6. Store recoverability.
- Confirm promotion state can be reconstructed from contextual/tiered store archives and the recurrence ledger.

## Status

Specification only. No implementation is included here.
