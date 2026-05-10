# ATIS Benchmark B2-B4 Improvements Plan

## Objective
Improve the metrics for B2 (Slot Generalization), B3 (Consolidation Effectiveness), and B4 (Variant Match Rate) in the `TwoAgentATISBenchmark` to meet their targets (>70%, >30%, >0% respectively).

## Analysis & Root Causes

### 1. B3 & B4: Consolidation Not Triggering (0% Reduction, 0% Match Rate)
- **Root Cause**: The benchmark sets `max_patterns = 8192` and `consolidation_threshold = 0.8`. Consolidation only triggers when the pattern count exceeds `8192 * 0.8 = 6553`.
- B3 trains on a subset of 1,000 utterances, which generates ~2,400 patterns. This never hits the threshold, resulting in 0 consolidation.
- B1 (which B4 relies on) trains on ~4,000 utterances, which also doesn't reliably hit 6,553 patterns depending on view sparsity.

### 2. B2: Slot Generalization (Target >70%)
- **Root Cause**: B2 relies on the `content_view` (a 96-dim semantic unit vector) to generalize to novel entities. The `near_threshold` for `content_view` in `HPMPipeline` is set to `0.08`, which may be too tight to match substituting one city/airline for another.

## Implementation Steps

### Step 1: Adjust `max_patterns` in `TwoAgentATISBenchmark`
- Open `hpm_ai_v5/experiments/run_atis_two_agent_benchmark.py`.
- Change `max_patterns=8192` to `max_patterns=2048`. This means consolidation triggers at `2048 * 0.8 = 1638` patterns.
- This ensures that training on the full set (~4000 items) will aggressively consolidate variants, fulfilling B4's requirements.

### Step 2: Override `max_patterns` explicitly for B3
- In the `run_b3` method, the subset is `train[:1000]`. This generates ~2,400 patterns without consolidation.
- `max_patterns=2048` is perfect here since 2,400 > 1,638. Consolidation will trigger, demonstrating the reduction.

### Step 3: Loosen `content_view` Distance Threshold
- In `TwoAgentATISBenchmark.__init__`, update the `view_configs`:
  ```python
  view_configs={
      "content_view": {"near_threshold": 0.15, "exact_threshold": 0.02},
  }
  ```
- This will allow the `inference_agent` to more reliably match novel entity vectors to trained intent patterns.

### Step 4: Ensure Correct Agent Context
- In `ATISInferenceAgent.step_packet`, explicitly copy `packet.context.get("atis_route")` into `intent_specialist_mode` to perfectly mirror the `ATISIntentAgent` training logic, ensuring exact context signatures are matched if they ever become relevant to the core logic.

## Verification
- Run `uv run python -m hpm_ai_v5.experiments.run_atis_two_agent_benchmark`.
- B2 should exceed 70%.
- B3 should show >30% reduction.
- B4 should show >0% variant contribution.

## Future Exploration: Pattern Overseer Agent
- As requested by the user, we will explore introducing a dedicated **Pattern Overseer Agent** in future iterations. Currently, pattern consolidation and promotion are handled algorithmically by the `PatternManager`. An Overseer Agent could actively monitor the `PatternStore` for fragmentation, evaluate view utility over time, and trigger consolidation or pruning dynamically rather than relying on static thresholds.