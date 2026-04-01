
# `experiment_math_controller.py`

## Summary
Compares the direct synchronous HFN math loop with the new async controller layer. The controller does not change the HFN core; it sits above it and handles ingest, replay, prefetch, gap-query, and snapshot responsibilities.

## What It Does
- Builds the existing protected math world model.
- Runs the same arithmetic observation stream in two modes:
  - direct `Observer.observe()` loop
  - async `AsyncHFNController` loop
- Uses the controller for:
  - `ingest`
  - `prefetch`
  - `replay`
  - `snapshot_state`
  - `request_gap_query`
- Compares direct vs controller throughput and reports the controller snapshot state.

## What It Aims To Achieve
- Show that HFN can stay synchronous at the core while still supporting async orchestration around it.
- Measure the overhead of the adapter layer.
- Prove the controller can safely coordinate replay, prefetch, and external lookup without mutating HFN from multiple writers.

## Observed Results
Smoke and full runs both completed successfully.

The controller path preserved the same math learning behavior while adding orchestration overhead. The measured full-run numbers were:

- direct throughput: `14.2 obs/s`
- controller throughput: `12.5 obs/s`
- controller replay throughput: `9.0 obs/s`
- coverage: `100.00%` for both paths
- peak RSS delta: `69.5 MB` direct vs `4.3 MB` controller
- learned nodes surviving: `68` direct vs `69` controller
- replay count: `25`
- prefetched nodes found: `12`

The benchmark also reports controller snapshot state:
- queue size: `0`
- replayed observations: `25`
- gap queries: `0`
- last event: `replay`

## How To Run
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_math_controller.py
```

## Main Signals
- Whether the controller preserves HFN behavior.
- How much throughput the async boundary costs.
- Whether replay and prefetch can be scheduled without touching the core learning path.
- Whether snapshot/export gives a useful compact state summary.

## Insights
- The controller should be treated as orchestration, not learning.
- HFN remains the serial state machine.
- Async is useful at the boundaries: external queries, replay planning, prefetch, and state export.

## Issues / Limits
- The controller adds overhead, so it is not a pure speed optimization.
- This benchmark is math-only.
- The controller currently serializes all HFN mutations through one worker, which is the safe design, but not the fastest possible design.

## Notes
- This is the first concrete adapter-layer experiment above HFN.
- The next useful step would be to route one of the existing lexical or math curriculum experiments through this controller and compare its state snapshots over time.
