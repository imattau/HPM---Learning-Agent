# `experiment_lexical_curriculum.py`

## Summary
Runs the same WordNet-backed forest through a curriculum of progressively harder lexical streams. This is the experiment closest to the "stretching over time" story: do earlier stages leave behind reusable learned structure that later stages can build on?

## What It Does
- Builds or reuses a persistent WordNet forest.
- Runs three stages in sequence:
  - `seed` with in-domain Peter Rabbit / repo text
  - `stretch` with medium-difficulty OOD vocabulary
  - `stretch_hard` with harder OOD vocabulary
- Keeps the same forest and observer across stages.
- Reports stage-by-stage changes in coverage, surviving learned nodes, abstraction layer, and purity.

## What It Aims To Achieve
- Demonstrate incremental improvement rather than one-off coverage.
- Show whether HFN accumulates reusable structure as the data gets harder.
- Move the story from "can it explain this?" to "does it get better when stretched?"

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
- The smoke run already shows the right qualitative shape: later stages can leave behind additional surviving learned nodes.
- This makes the experiment better aligned with the human-learning analogy than a single-shot transfer run.
- It is the clearest documentable bridge between rich priors and cumulative adaptation.

## Issues / Limits
- The current curriculum is still small compared with what a convincing long-term learning claim would need.
- If the ontology is too strong, coverage remains easy and the stretch signal can be subtle.
- Full runs are slower because the same forest is reused across multiple stages.

## Notes
- Uses the same diagonal-sigma HFN setup as the other lexical experiments.
- The experiment currently relies on the transfer experiment for the in-domain batch helper.
