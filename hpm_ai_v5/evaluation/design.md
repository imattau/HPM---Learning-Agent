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
- ARC transformation solving

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
