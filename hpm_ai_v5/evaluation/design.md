# HPM AI v5 Objective Evaluation

## Purpose

The v5 stack needs an objective evaluation layer that turns the existing
benchmarks into comparable numeric scores.

## What it measures

- core reasoning
- agent pipeline flow
- delayed-reward planning
- rule discovery and reuse
- learned utility from reward feedback
- triple sequence discovery and macro reuse
- polygraph agreement over multiple views
- scoring weight adaptation across environments
- online meta-pattern discovery across structurally similar tasks
- automatic adapter composition across preprocessing pipelines
- open adapter discovery with explicit defer on unsupported structure
- ARC transformation solving

Polygraph agreement is scored as a separate benchmark that measures whether the
stack prefers the clean views over the noisy one when multiple structural
representations are available.

Scoring weight adaptation is scored separately as agent-side meta-learning over
the core's fixed `α, β, γ, δ` formula.

Online meta-pattern discovery is scored separately as agent-side structural
abstraction and zero-shot transfer across similar tasks.

Automatic adapter composition is scored separately as agent-side pipeline
selection and reuse over held-out task variants.

Open adapter discovery is scored separately as agent-side adapter selection
and calibration over known numeric and grid families, plus a held-out graph
family that must be deferred.

Those pipelines are now treated as adapter compositions rather than a separate
preprocessing tier.

## Design rule

Evaluation should be metrics-first.

- benchmark result
- numeric score
- compact trace

The report is meant to be comparable across revisions without depending on
natural-language explanations.

## Output contract

An evaluation report should include:

- per-benchmark score
- per-benchmark metrics
- pass/fail flag
- overall score
- summary of passed and failed benchmarks

## Scope

This is not a training loop and not a general judge. It is a deterministic
regression-style evaluation harness for the v5 system surface.
