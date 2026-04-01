# `experiment_lexical_math_cross_stream_replay.py`

## Summary
Runs the WordNet-backed lexical forest through a lexical curriculum, then switches to a math-text observation stream, and finally replays only the most useful seed observations before returning to the original lexical stream. This is the experiment that most directly probes cross-stream memory retention and selective replay across a different symbolic domain.

## What It Does
- Builds or reuses the WordNet prior forest used by the other lexical experiments.
- Runs seven persistent stages in sequence:
  - `seed` with in-domain Peter Rabbit / repo text
  - `stretch` with medium-difficulty OOD vocabulary
  - `stretch_hard` with harder OOD vocabulary
  - `pressure_shock` with an aggressively reduced hot cache and frequent sweeps
  - `math_shift` with arithmetic observations rendered as math-language text
  - `selective_replay` with only the seed observations attached to the highest-utility seed learned nodes
  - `revisit_seed` by replaying the original seed batch after the cross-stream interference
- Keeps the same forest and observer across the full sequence.
- Reports coverage, surviving learned nodes, abstraction layers, and how many selected seed nodes survive the math shift and are reused in selective replay.

## What It Aims To Achieve
- Test whether a lexical memory can survive a stream switch without collapsing.
- See whether a small targeted replay set can recover useful prior structure after interference from a different symbolic stream.
- Distinguish blunt full replay from selective replay of high-utility nodes.
- Move closer to a lifelong-memory story where old knowledge can be retained, reused, and reinforced across tasks.

## Observed Results
- Smoke runs were small but the full run gave the real signal.
- Full-run results:
  - `seed`: `7` new learned nodes, `7` surviving, mean explaining layer `2.26`
  - `stretch`: `10` new learned nodes, `10` surviving, mean explaining layer `2.71`
  - `stretch_hard`: `12` new learned nodes, `13` surviving, mean explaining layer `2.97`
  - `pressure_shock`: `13` new learned nodes, `13` surviving, mean explaining layer `3.19`
  - `math_shift`: `6` new learned nodes, `7` surviving, mean explaining layer `2.51`
  - `selective_replay`: `10` new learned nodes, `12` surviving, mean explaining layer `2.33`
  - `revisit_seed`: `8` new learned nodes, `9` surviving, mean explaining layer `2.40`
- Coverage stayed at `100%` throughout.
- The math shift created measurable cross-stream interference, with `136` learned explanations in that stage.
- The selected replay used `15` replay observations, left `0` of the selected seed nodes alive after the math shift, and still directly reused `1` of the selected seed nodes during selective replay.
- The final revisit still returned close to the seed layer and added `+21` learned explanations versus the seed stage.

## How To Run
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_lexical_math_cross_stream_replay.py
```

## Main Signals
- Whether seed-learned nodes survive the pressure and math-shift stages.
- Whether math-language observations cause interference or fresh structure.
- Whether selective replay recovers the chosen seed nodes better than a blunt full replay.
- Whether the final lexical revisit benefits from the cross-stream replay sequence.

## Insights
- This is the strongest multi-stream memory test in the lexical suite so far.
- The math stage does not erase the lexical substrate, but it does shift the explanation structure enough to justify replay.
- Selective replay is meaningful: it reuses at least some of the chosen seed nodes after math interference.
- The final revisit shows the forest can still come back to the original lexical stream after a pressure stage and a different symbolic stream.
- This is much closer to a reusable memory substrate than single-domain stretching alone.

## Issues / Limits
- The selective replay set is still small, so the experiment is more of a proof of concept than a full continual-learning benchmark.
- The math stream is textualized to keep the same HFN representation space, so this is a cross-stream stress test rather than a separate numeric encoder.
- Coverage remains perfect, so the strongest signal is in learned-node structure and reuse rather than accuracy failure.
- A future version should pair this with a native math encoder or a fully separate world model if we want a stricter multi-modal test.

## Notes
- Uses the same diagonal-sigma HFN setup as the other lexical experiments.
- Reuses the curriculum and pressure helpers for the lexical stages.
- Best interpreted as the bridge from retention/forgetting tests to selective, cross-stream replay under continual learning.
