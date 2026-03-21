# Phase 5 Predictive Synthesis Agent - Session State

## Objective
Write implementation plan for PredictiveSynthesisAgent, then review it, then offer execution choice.

## Progress
- Spec written and committed: `docs/superpowers/specs/2026-03-21-phase5-predictive-synthesis-design.md`
- Read hpm/monitor/__init__.py and hpm/agents/multi_agent.py to understand patterns
- Plan NOT yet written

## Codebase Context (from reading files)

### hpm/monitor/__init__.py (current content)
```python
from .structural_law import StructuralLawMonitor
from .recombination_strategist import RecombinationStrategist

__all__ = ["StructuralLawMonitor", "RecombinationStrategist"]
```
Need to add: `from .predictive_synthesis import PredictiveSynthesisAgent` and add to `__all__`

### hpm/agents/multi_agent.py (current state)
- Constructor signature: `def __init__(self, agents, field, seed_pattern=None, groups=None, monitor=None, strategist=None, bridge=None):`
- step() returns: `{**metrics, "field_quality": field_quality, "interventions": interventions, "bridge_report": bridge_report}`
- Need to add `forecaster=None` kwarg
- Need to call `self.forecaster.step(self._t, observations, field_quality)` after `bridge_report`
- Need to add `"forecast_report": forecast_report` to return dict

## Remaining Work (IN ORDER)

1. Write implementation plan to `docs/superpowers/plans/2026-03-21-phase5-predictive-synthesis.md`
   - Use writing-plans skill format (already invoked in previous session)
   - Header with Goal/Architecture/Tech Stack
   - Task 1: PredictiveSynthesisAgent class + tests (hpm/monitor/predictive_synthesis.py + tests/monitor/test_predictive_synthesis.py)
   - Task 2: MultiAgentOrchestrator integration + __init__ export
   - Task 3: Final verification + smoke test (optional, can merge into Task 2)

2. Run plan-document-reviewer subagent on the plan

3. Fix any issues found

4. Tell user: "Plan complete and saved to `docs/superpowers/plans/2026-03-21-phase5-predictive-synthesis.md`. Two execution options: 1. Subagent-Driven (recommended) 2. Inline Execution. Which approach?"

## Key Design Details (from spec)

### New file: hpm/monitor/predictive_synthesis.py
```python
from collections import deque
import numpy as np

class PredictiveSynthesisAgent:
    def __init__(self, store, probe_k=10, probe_n=5, probe_sigma_scale=0.5,
                 fragility_threshold=1.0, min_bridge_level=4):
        self._store = store
        self.probe_k = probe_k
        self.probe_n = probe_n
        self.probe_sigma_scale = probe_sigma_scale
        self.fragility_threshold = fragility_threshold
        self.min_bridge_level = min_bridge_level
        self._obs_history = deque(maxlen=probe_k)

    def step(self, step_t: int, current_obs: dict, field_quality: dict) -> dict:
        # Fast gate
        if field_quality.get("level4plus_count", 0) == 0:
            return _empty_report()

        # Query store, filter level >= min_bridge_level, get top-weight pattern
        # Fallback to level 3 if no level 4+
        # Extract x_obs from current_obs, trim/pad to match pattern dim
        # prediction = top_pattern.mu.copy()
        # prediction_error = float(top_pattern.log_prob(x_obs))
        # self._obs_history.append(x_obs.copy())
        # Far Transfer probe if len(obs_history) >= probe_k:
        #   sigma_probe = probe_sigma_scale * sqrt(max(mean(diag(sigma)), 1e-9))
        #   for each obs in history, probe_n noisy copies, compute nlls
        #   fragility_score = mean(nlls)
        #   delta_nll = fragility_score - prediction_error
        #   fragility_flag = delta_nll > fragility_threshold
```

### 7 return keys
prediction, prediction_error, fragility_score, delta_nll, fragility_flag, top_pattern_level, top_pattern_id

### 11 test cases
- test_uncertainty_when_no_l4_patterns
- test_prediction_is_top_pattern_mu
- test_fallback_to_level3
- test_prediction_error_computed
- test_probe_none_before_k_obs
- test_robust_pattern_not_flagged
- test_fragile_pattern_flagged
- test_noise_scaled_to_pattern_sigma
- test_forecast_report_keys_always_present
- test_orchestrator_no_forecaster
- test_orchestrator_forecaster_integrated

### Spec location
docs/superpowers/specs/2026-03-21-phase5-predictive-synthesis-design.md

### Test patterns (from similar test files)
Look at tests/monitor/test_structural_law.py and tests/substrate/test_bridge.py for patterns.

## Working Directory
/home/mattthomson/workspace/HPM---Learning-Agent

execution_mode: unattended
auto_continue: true
