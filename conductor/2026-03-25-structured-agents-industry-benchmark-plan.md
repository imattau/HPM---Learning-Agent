# Structured Agents Industry Benchmark Plan

## Status

Planning-only document. No implementation is included in this phase.

## Goal

Test the latest `structured_agents` stack against two industry-relevant benchmarks that exercise the two core capabilities the HPM line now claims:

1. abstract relational reasoning under novelty
2. long-horizon software repair and execution

## Benchmark Targets

1. [ARC-AGI-2](https://arcprize.org/arc-agi/2/)
2. [SWE-bench Verified](https://www.swebench.com/verified.html)

## Why These Two

ARC-AGI-2 is the best fit for the structured stack's relational and hierarchical pattern machinery. It should tell us whether the added HPM completion machinery improves abstraction, consistency, and generalization on unfamiliar tasks.

SWE-bench Verified is the best fit for the structured stack's agentic execution machinery. It should tell us whether the same stack improves patch quality, repair consistency, and multi-step action under real repository constraints.

Together they cover the two main failure modes:

- weak abstraction over relational structure
- weak execution over long, stateful repair loops

## Scope

- Use only `structured_agents` benchmark paths.
- Treat `HierarchicalOrchestrator` as legacy-only and out of scope for new conclusions.
- Compare baseline structured agents against completion-enhanced structured agents.
- Prefer thin adapters over benchmark rewrites.

## Comparison Matrix

For each benchmark, run at least these conditions:

1. `baseline`
2. `completion`
3. `completion_no_lifecycle`
4. `completion_no_adaptive`
5. `completion_no_identity`

This matrix isolates which HPM completion feature actually changes performance.

## ARC-AGI-2 Plan

### Objective

Measure whether the updated structured agents improve abstract task solving on unseen ARC-AGI-2 problems.

### Primary Metrics

- task accuracy
- pass@1
- seed stability
- trace completeness
- relational/identity signal usage

### Secondary Metrics

- candidate rank of the correct answer
- error consistency across tasks from the same family
- runtime per task

### Protocol

1. Run the official ARC-AGI-2 task set with fixed seeds.
2. Use the existing structured benchmark style: task-level candidate ranking, not free-form generation.
3. Compare baseline structured agents to completion-enhanced structured agents.
4. Track which completion features are causally active in the final score.

### Acceptance Gates

- completion must not reduce accuracy below the baseline by more than a small tolerance
- completion should improve at least one of:
  - seed stability
  - relational consistency
  - trace completeness
- any improvement must be reproducible across multiple seeds

## SWE-bench Verified Plan

### Objective

Measure whether the updated structured agents improve real-world code repair under repository and test constraints.

### Primary Metrics

- resolved issue rate
- test pass rate
- valid patch rate
- time to first passing candidate
- tool-call efficiency

### Secondary Metrics

- edit churn
- patch compactness
- retry count
- failure mode classification

### Protocol

1. Run the verified SWE-bench subset with fixed budgets.
2. Keep the agent loop bounded so comparisons are fair and reproducible.
3. Record whether completion features improve repair selection or merely add bookkeeping.
4. Compare against the same structured-agent baseline, not a different architecture.

### Acceptance Gates

- completion must improve or preserve resolved issue rate under the same budget
- traces must remain sufficient to explain the chosen patch path
- any gain must not come from uncontrolled extra tool usage or budget expansion

## Benchmark Instrumentation

1. Record the active completion mode for every run.
2. Record the final decision trace for every scored task.
3. Record seed, task ID, benchmark version, and budget.
4. Emit per-condition deltas, not only aggregate scores.

## Rollout Order

1. Implement the ARC-AGI-2 adapter first.
2. Run a small smoke slice and check that the completion stack is neutral or beneficial.
3. Implement the SWE-bench Verified adapter next.
4. Run a small verified subset before any larger sweep.
5. Only then run the full comparison matrix.

## Decision Rule

If a completion feature does not change a benchmark decision, it is not yet causal enough to justify the added complexity. If it changes decisions but does not improve the target metric, it needs either tuning or removal.
