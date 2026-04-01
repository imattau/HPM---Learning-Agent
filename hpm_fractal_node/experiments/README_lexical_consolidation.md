# `experiment_lexical_consolidation.py`

## Summary
Runs the same WordNet-backed forest through a curriculum of progressively harder lexical streams, then replays the original seed stream after the forest has been stretched. This is the first lexical experiment in the suite that explicitly targets retention and consolidation rather than only coverage or transfer.

## What It Does
- Builds or reuses the WordNet prior forest used by the other lexical experiments.
- Runs four persistent stages in sequence:
  - `seed` with in-domain Peter Rabbit / repo text
  - `stretch` with medium-difficulty OOD vocabulary
  - `stretch_hard` with harder OOD vocabulary
  - `revisit_seed` by replaying the original seed batch after stretching
- Keeps the same forest and observer across all stages.
- Reports stage-by-stage changes in coverage, surviving learned nodes, abstraction layer, purity, memory, and explicit reuse of seed-learned nodes.

## What It Aims To Achieve
- Show whether HFN/HPM retains what it learned after being pushed into harder lexical territory.
- Test whether revisiting earlier data triggers deeper or more reusable explanations.
- Move the narrative from incremental learning to consolidation and recall.
- Provide the closest lexical analogue so far to how a memory substrate should improve over time.

## Observed Results
- Smoke runs already showed the right consolidation shape, even though they were still small.
- Full-run results are now available and give the stronger signal:
  - `seed`: `9` new learned nodes, `9` surviving, mean explaining layer `2.15`
  - `stretch`: `9` new learned nodes, `9` surviving, mean explaining layer `2.76`
  - `stretch_hard`: `12` new learned nodes, `13` surviving, mean explaining layer `3.15`
  - `revisit_seed`: `9` new learned nodes, `9` surviving, mean explaining layer `2.21`
- Coverage stayed at `100%` throughout.
- The hardest stage pushed the mean explaining layer highest, which is the expected stretching signal.
- The revisit stage returned close to the seed-layer regime, but it did not strongly reuse seed-learned nodes: only `1` seed learned node was still retained after revisit, and `0` were directly reused in the replay explanations.
- The revisit stage still produced `+15` more learned explanations than the seed stage, so the replay is not a no-op, but the consolidation signal is weaker than the curriculum signal.

## How To Run
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_lexical_consolidation.py
```

## Main Signals
- Whether the seed learned nodes survive after the harder stages.
- Whether the revisit stage explains the same material at a higher abstraction layer.
- Whether the revisit stage reuses seed-learned nodes or creates fresh ones.
- Coverage delta and mean-layer delta between the first seed stage and the final revisit stage.

## Insights
- This is the first lexical experiment that directly probes consolidation, not just adaptation.
- The full run shows a clear stretching pattern: the harder stages push the mean explaining layer up, and the model retains more learned structure as difficulty increases.
- The replay stage does not just replicate the seed stage; it returns close to the seed-layer regime while still producing a meaningful number of learned explanations.
- Consolidation is present, but weak: the full run retained only one seed-learned node after the revisit stage and directly reused none of the seed-learned nodes in the replay explanations.
- That makes this a useful substrate signal, not a solved continual-learning story.

## Issues / Limits
- The revisit stage still does not strongly show seed-node reuse, even in the full run.
- Because coverage stays perfect, the main signal is hidden inside explanation structure rather than outright failure.
- The consolidation effect is smaller than the curriculum effect, so the model is still stretching better than it is consolidating.
- A harder revisit regime or explicit forgetting pressure may be needed to force stronger reuse and retention.

## Notes
- Uses the same diagonal-sigma HFN setup as the other lexical experiments.
- Reuses the curriculum helpers for the in-domain and out-of-domain batches.
- Best interpreted as the bridge from curriculum learning to continual learning and consolidation.
