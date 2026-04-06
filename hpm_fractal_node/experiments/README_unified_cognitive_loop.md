# Experiment 17: Unified Cognitive Loop (The Core Agent)

## Objective
Validate the integration of all HFN capabilities into a single autonomous agent loop. The agent must navigate an evolving environment, detect failures caused by environmental shifts, revise its beliefs, explore to discover new rules, and successfully replan to reach its goal.

## Background
The "Unified Cognitive Loop" represents the culmination of the Sovereign AI experiment series (SP17–SP41). It proves that the HFN architecture can sustain the full lifecycle of agency:
1. **Curiosity**: Exploring the world to build an initial model.
2. **Planning**: Sequencing learned rules to reach a distal goal.
3. **Action**: Executing the plan in the environment.
4. **Falsification**: Detecting when expectations fail (high surprise).
5. **Belief Revision**: Penalizing falsified rules and promoting new evidence.
6. **Adaptation**: Exploring and replanning until the goal is reached.

## Setup
- **Domain**: 10D continuous state space.
- **Rules**: Action $i$ increments Dimension $i$ by 1.0.
- **Agent**: `UnifiedAgent` with `Observer`, `Decoder`, and `WeightAwareRetriever`.
- **Goal**: Reach state `[2.0, 1.0, 0.0, ...]` from the origin.

## The Environment Shift
Mid-way through the experiment, the environment changes:
- **Action 0**: Shifts from `+1.0` to `-1.0` on Dimension 0 (Falsifying the agent's primary path).
- **Action 9**: Shifts from `+1.0` on Dimension 9 to `+1.0` on Dimension 0 (Providing a new, hidden path to the goal).

## Results
The agent successfully demonstrated autonomous adaptation:
- **Initial Success**: Reached the goal using plan `[0, 1, 0]`.
- **Detection**: Upon the rule shift, the agent perceived high surprise ($S \approx 2.5$) and correctly penalized the OLD Action 0 node (weight dropped to $0.0$).
- **Multiplicative Penalty**: Using the `WeightAwareRetriever` with scoring `dist / (weight + epsilon)`, the agent successfully suppressed the falsified rule.
- **Exploration & Discovery**: After failing its first adapted attempt, the agent entered an exploration phase, discovered the new rule for Action 9, and promoted it.
- **Final Success**: Reached the goal on the second attempt using plan `[1, 9, 9]`.

## Verification Command
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_unified_cognitive_loop.py
```
