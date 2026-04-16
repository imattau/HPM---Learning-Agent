# Plan: SP94 One-Shot Reaction Prediction

## Objective
Implement and validate the SP94 experiment, demonstrating the agent's ability to learn a chemical transformation (ester hydrolysis) from a single example and generalize it to a new molecule using a fingerprint-based representation.

## Key Files & Context
- **Experiment Script**: `hpm_ai_v2/experiments/experiment_sp94_reaction_prediction.py`
- **Agent Base**: `hpm_ai_v2/agents/base_agent.py`
- **Domain Config**: `hpm_ai_v2/domains/base.py`
- **HFN Core**: `hfn/hfn.py`

## Implementation Steps
- [ ] Create `hpm_ai_v2/experiments/experiment_sp94_reaction_prediction.py` with the provided self-contained content.
- [ ] Ensure the script has the correct execution permissions.

## Verification & Testing
- [ ] Run the experiment script: `PYTHONPATH=. .venv/bin/python hpm_ai_v2/experiments/experiment_sp94_reaction_prediction.py`
- [ ] Verify Phase 1: The agent successfully learns the `hydrolysis` transformation from methyl acetate.
- [ ] Verify Phase 2: The agent successfully generalizes the transformation to ethyl acetate, producing the correct output fingerprint.
- [ ] Confirm the final output displays `[SUCCESS] One-shot reaction prediction demonstrated.`
