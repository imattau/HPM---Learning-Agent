# `experiment_lexical_cross_domain_replay.py`

## Summary
Runs the WordNet-backed lexical forest through a lexical curriculum, then switches to code-like observations, and finally replays only the most useful seed observations before returning to the original lexical stream. This is the experiment that most directly probes cross-domain memory retention with selective replay.

## What It Does
- Builds or reuses the WordNet prior forest used by the other lexical experiments.
- Runs seven persistent stages in sequence:
  - `seed` with in-domain Peter Rabbit / repo text
  - `stretch` with medium-difficulty OOD vocabulary
  - `stretch_hard` with harder OOD vocabulary
  - `pressure_shock` with an aggressively reduced hot cache and frequent sweeps
  - `code_shift` with code-like observations derived from Python token categories
  - `selective_replay` with only the seed observations attached to the highest-utility seed learned nodes
  - `revisit_seed` by replaying the original seed batch after the cross-domain interference
- Keeps the same forest and observer across the full sequence.
- Reports coverage, surviving learned nodes, abstraction layers, and how many selected seed nodes survive the code shift and are reused in selective replay.

## What It Aims To Achieve
- Test whether a lexical memory can survive a domain switch without collapsing.
- See whether a small targeted replay set can recover useful prior structure after interference.
- Distinguish blunt full replay from selective replay of high-utility nodes.
- Move closer to a lifelong-memory story where old knowledge can be retained, reused, and reinforced across tasks.

## Observed Results
- Smoke runs were small but the full run gave the real signal.
- Full-run results:
  - `seed`: `8` new learned nodes, `8` surviving, mean explaining layer `2.30`
  - `stretch`: `9` new learned nodes, `9` surviving, mean explaining layer `2.79`
  - `stretch_hard`: `12` new learned nodes, `12` surviving, mean explaining layer `2.92`
  - `pressure_shock`: `7` new learned nodes, `7` surviving, mean explaining layer `3.19`
  - `code_shift`: `11` new learned nodes, `11` surviving, mean explaining layer `2.68`
  - `selective_replay`: `11` new learned nodes, `12` surviving, mean explaining layer `2.56`
  - `revisit_seed`: `10` new learned nodes, `12` surviving, mean explaining layer `2.40`
- Coverage stayed at `100%` throughout.
- The code shift created measurable cross-domain interference, with `112` learned explanations in that stage.
- The selected replay used `18` replay observations, kept `1` of the selected seed nodes alive after the code shift, and directly reused `1` of the selected seed nodes.
- The final revisit still returned close to the seed layer and added `+22` learned explanations versus the seed stage.

## How To Run
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_lexical_cross_domain_replay.py
```

## Main Signals
- Whether seed-learned nodes survive the pressure and code-shift stages.
- Whether code-like observations cause interference or fresh structure.
- Whether selective replay recovers the chosen seed nodes better than a blunt full replay.
- Whether the final lexical revisit benefits from the cross-domain replay sequence.

## Insights
- This is the strongest multi-task memory test in the lexical suite so far.
- The code stage does not erase the lexical substrate, but it does shift the explanation structure enough to justify replay.
- Selective replay is meaningful: it reuses at least some of the chosen seed nodes after code interference.
- The final revisit shows the forest can still come back to the original lexical stream after a domain switch and pressure stage.
- This is much closer to a reusable memory substrate than single-domain stretching alone.

## Issues / Limits
- The selective replay set is still small, so the experiment is more of a proof of concept than a full continual-learning benchmark.
- The code-like domain is synthetic and still encoded through the lexical model, so it is a cross-domain stress test rather than a fully separate code encoder.
- Coverage remains perfect, so the strongest signal is in learned-node structure and reuse rather than accuracy failure.
- A future version should use a truly different encoder or domain-specific observation pipeline if we want a harder multi-modal test.

## Notes
- Uses the same diagonal-sigma HFN setup as the other lexical experiments.
- Reuses the curriculum and pressure helpers for the lexical stages.
- Best interpreted as the bridge from retention/forgetting tests to selective, cross-domain replay under continual learning.
