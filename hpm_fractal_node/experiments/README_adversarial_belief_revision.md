# Experiment 19: Adversarial Belief Revision (Truth Under Conflict)

## Objective
To test the HFN's ability to unlearn entrenched, high-confidence but incorrect structural knowledge when faced with conflicting evidence.

## Background
Robust intelligence requires overcoming deeply ingrained "truths" that are later falsified. This experiment creates an "adversarial bias" by exposing the agent to a misleading environment (Phase 1) before switching to the true environment (Phase 2). It validates the Observer's ability to penalize falsified beliefs while promoting new evidence without catastrophic forgetting or oscillation.

## Setup
- **Domain**: Simple sequence environment ($A \rightarrow B \rightarrow X$).
- **Phase 1 (Adversarial)**: $B \rightarrow C$ (repeated 50 times to build high weight).
- **Phase 2 (Truth)**: $B \rightarrow D$ (contradicts Phase 1).
- **Agent**: `BeliefRevisionAgent` with explicit falsification on surprise.

## Results
The agent successfully demonstrated belief revision under conflict:
- **Entrenchment**: Belief C reached a weight of **0.6211** in Phase 1.
- **Rapid Shift**: Upon entering Phase 2, the true belief D surpassed the entrenched belief C in only **3 steps**.
- **Smooth Decay**: The incorrect belief C weight decayed exponentially (via multiplicative penalty) until it dropped below the **0.01** threshold at step **11**.
- **Minimal Conflict**: Residual conflict (both beliefs > 0.05) lasted only **2 steps**.
- **No Failure Modes**: No confirmation bias or oscillation was detected.

## Metrics Summary
| Metric | Result | Status |
|---|---|---|
| Belief Shift Time | 3 steps | SUCCESS |
| Residual Conflict | 2 steps | SUCCESS |
| Final C Weight | 0.004852 | SUCCESS (Suppressed) |
| Final D Weight | 0.203767 | SUCCESS (Dominant) |

## Verification Command
```bash
PYTHONPATH=. python3 hpm_fractal_node/experiments/experiment_adversarial_belief_revision.py
```
