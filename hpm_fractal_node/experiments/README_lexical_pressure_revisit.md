# `experiment_lexical_pressure_revisit.py`

## Summary
Runs the same WordNet-backed forest through a lexical curriculum, then applies an explicit hot-cache shock before replaying the original seed stream. This is the experiment that most directly tests whether HFN/HPM can survive forgetting pressure and still recover earlier structure.

## What It Does
- Builds or reuses the WordNet prior forest used by the other lexical experiments.
- Runs five persistent stages in sequence:
  - `seed` with in-domain Peter Rabbit / repo text
  - `stretch` with medium-difficulty OOD vocabulary
  - `stretch_hard` with harder OOD vocabulary
  - `pressure_shock` with an aggressively reduced hot cache and frequent sweeps
  - `revisit_seed` by replaying the original seed batch after the shock
- Temporarily shrinks the hot cache and increases sweep frequency only for the pressure stage.
- Reports stage-by-stage changes in coverage, surviving learned nodes, abstraction layer, and the number of seed learned nodes still present after the shock.

## What It Aims To Achieve
- Force actual forgetting pressure instead of just harder inputs.
- See whether the revisit stage can still recover earlier concepts after the forest has been squeezed.
- Test whether learned nodes survive as cold nodes and can be reused after the shock is lifted.
- Push the lexical story closer to continual learning under memory constraints.

## Observed Results
- Smoke runs were too small to trigger much retention loss.
- Full-run results are more informative:
  - `seed`: `12` new learned nodes, `12` surviving, mean explaining layer `2.25`
  - `stretch`: `9` new learned nodes, `10` surviving, mean explaining layer `2.75`
  - `stretch_hard`: `9` new learned nodes, `10` surviving, mean explaining layer `2.94`
  - `pressure_shock`: `9` new learned nodes, `9` surviving, mean explaining layer `3.11`
  - `revisit_seed`: `9` new learned nodes, `10` surviving, mean explaining layer `2.41`
- Coverage stayed at `100%` throughout.
- The pressure stage reduced the retained seed-learned nodes to `2` before replay.
- The replay reused `2` of those seed-learned nodes and still produced `+27` more learned explanations than the seed stage.
- The replay’s mean explaining layer rose by `+0.16` versus seed, so the revisit was not a blind reset.

## How To Run
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_lexical_pressure_revisit.py
```

## Main Signals
- How many seed-learned nodes survive the pressure stage.
- Whether the replay stage reuses cold nodes or rebuilds new structure.
- Whether the replay maintains coverage while operating under less cached memory.
- The delta between the seed stage and the final replay stage in learned explanations and abstraction layer.

## Insights
- This is the first lexical experiment that explicitly adds memory pressure instead of only harder data.
- The shock stage does create real forgetting pressure: only a small subset of seed-learned nodes survived the full stress run.
- The replay stage can still recover earlier material and reuse some surviving nodes, which is the behavior you want if HFN is to function as a memory substrate.
- The model is still better at stretching than consolidating, but this experiment moves the story closer to actual retention under resource pressure.

## Issues / Limits
- The replay still reuses only a small number of seed-learned nodes, even under pressure.
- Coverage remains perfect, so the signal is in reuse and layer shifts rather than outright failure.
- The pressure settings are intentionally blunt and may overfit the experiment to hot-cache churn.
- A more selective forgetting regime may be needed to distinguish durable semantic memory from accidental retention.

## Notes
- Uses the same diagonal-sigma HFN setup as the other lexical experiments.
- Reuses the curriculum helpers for the in-domain and out-of-domain batches.
- Best interpreted as the bridge from curriculum learning and consolidation to explicit memory-pressure testing.
