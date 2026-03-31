# `experiment_lexical_curriculum.py`

## Summary
Runs the same WordNet-backed forest through a curriculum of progressively harder lexical streams. This is the experiment closest to the stretching-over-time story: do earlier stages leave behind reusable learned structure that later stages can build on?

## What It Does
- Builds or reuses a persistent WordNet forest.
- Runs three stages in sequence:
  - `seed` with in-domain Peter Rabbit / repo text
  - `stretch` with medium-difficulty OOD vocabulary
  - `stretch_hard` with harder OOD vocabulary
- Keeps the same forest and observer across stages.
- Reports stage-by-stage changes in coverage, surviving learned nodes, abstraction layer, purity, and RSS.

## What It Aims To Achieve
- Demonstrate incremental improvement rather than one-off coverage.
- Show whether HFN accumulates reusable structure as the data gets harder.
- Move the story from "can it explain this?" to "does it get better when stretched?"

## Observed Results
- Smoke runs already show the right shape: later stages add surviving learned nodes.
- The full `N_SAMPLES=400`, `N_PASSES=1` run gives the strongest evidence so far.
- Full-run results:
  - `seed`: `6` new learned nodes, `6` surviving, mean explaining layer `2.23`
  - `stretch`: `8` new learned nodes, `8` surviving, mean explaining layer `2.68`
  - `stretch_hard`: `9` new learned nodes, `9` surviving, mean explaining layer `3.00`
- Coverage stayed at `100%` throughout, so the key signal is not failure, but cumulative retention and rising abstraction.
- The last stage ended with `+3` surviving learned nodes versus the first stage and a mean explaining-layer delta of `+0.77`.

## How To Run
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_lexical_curriculum.py
```

## Main Signals
- Stage-to-stage survival of learned nodes.
- Changes in mean explaining layer as the stream gets harder.
- Whether the hard stage can reuse structure learned in the easy and medium stages.
- Coverage delta between the first and last stage.

## Insights
- This is the clearest evidence yet for the "stretching" story.
- Later exposure does not just replay the same prior coverage; it leaves behind extra learned nodes.
- The mean explaining layer rises with difficulty, which is a strong sign that the model is moving toward more abstract explanations.
- This is closer to human curriculum learning than any single-shot transfer run in the suite.

## Issues / Limits
- The current curriculum is still small compared with what a convincing long-term learning claim would need.
- If the ontology is too strong, coverage remains easy and the stretch signal can be subtle.
- Full runs are slower because the same forest is reused across multiple stages.
- It still needs a true consolidation / revisit phase to show retention over time, not only stage progression.

## Notes
- Uses the same diagonal-sigma HFN setup as the other lexical experiments.
- The experiment currently relies on the transfer experiment for the in-domain batch helper.
- Best interpreted as the bridge from coverage tests to continual-learning tests.
