# SP8: Cross-Task L4 Generalisation — Design Spec

## Goal
Validate that a globally-trained L4GenerativeHead (trained across physics tasks) outperforms
per-task L4 training on the PhyRE discrimination benchmark.

## Background
SP7 showed L4 per-task (3 train pairs) does not outperform L2L3 (61.7% vs 62.5%).
The L4 training pairs sweep showed more per-task pairs don't help either.
Hypothesis: L4 needs cross-task training to learn the L2→L3 mapping structure.

## Approach: Task-Level 80/20 Split

### Phase 1 — Fit (192 seen tasks)
- Shuffle 240 tasks with seed=42, take first 192 (80%)
- For each seen task: encode train pairs as (L2_vec, L3_vec)
- Accumulate all pairs into global L4GenerativeHead(feature_dim_in=14, feature_dim_out=12)
- Call .fit() — produces W ∈ R^{12×14}

### Phase 2 — Evaluate (48 held-out tasks)
- For each held-out task: score 5 candidates using pre-fitted global L4
- Score = -||pred_L3 - actual_L3|| (higher = better)
- Also run flat, l2l3, per_task_l4 for comparison

## Components

### benchmarks/phyre_cross_task_l4.py
- fit_global_l4(seen_tasks, l2_enc, l3_enc) -> L4GenerativeHead
- score_cross_task_l4(task, global_l4, l2_enc, l3_enc) -> int
- run_benchmark(tasks, seed=42, train_frac=0.8) -> dict[str, float]
- main(): loads phyre_tasks.pkl, runs benchmark, prints results

### tests/benchmarks/test_phyre_cross_task_l4.py
- Smoke: fit on 10 tasks, eval on 3, assert accuracy in [0,1]
- Assert global L4 produces valid predictions (not None) for all candidates

## No Changes To
- L4GenerativeHead (hpm/agents/l4_generative.py)
- PhyreL2Encoder, PhyreL3Encoder (hpm/encoders/phyre_encoders.py)
- data/phyre_tasks.pkl

## Success Criteria
- cross_task_l4 > per_task_l4: hypothesis validated
- cross_task_l4 >= l2l3: strong result (genuine cross-task abstraction)
- Failure mode: if cross_task_l4 ≈ per_task_l4, L2→L3 mapping is not consistent across families

## Expected Output
```
SP8 Cross-Task L4 Benchmark (48 held-out tasks, 192 seen)
Condition         Accuracy
----------------------------------
flat              ~0.22
l2l3              ~0.63
per_task_l4       ~0.62
cross_task_l4     ?
```
