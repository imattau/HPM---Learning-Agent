# Agent Lifecycle and Persistence Upgrade Plan

## Objective
Enhance the `BaseHFNAgent` with automatic observation loops and periodic state persistence to support continuous, autonomous learning across tasks without manual intervention.

## Key Files & Context
- `hpm_ai_v2/agents/base_agent.py`: The core agent class where lifecycle methods will be added.

## Implementation Steps

### 1. Implement Auto-Observation Loop
**Target:** `hpm_ai_v2/agents/base_agent.py`
- Add `auto_observe_frequency: int = 0` and `replay_buffer_size: int = 100` to the `BaseHFNAgent.__init__` parameters.
- Initialize `self._observe_counter = 0` and `self._replay_buffer = []`.
- Add `_maybe_auto_observe(self, x: Optional[np.ndarray] = None)` method to periodically call `self.observer.observe(x)` or sample from the replay buffer.
- Add `observe_example(self, x: np.ndarray)` as a public method for explicit training.
- Update the `solve()` method: After a successful solution, if `auto_observe_frequency > 0`, encode the first input and call `_maybe_auto_observe(flat[:self.m_dim])`.

### 2. Implement Node Persistence (Auto-Save)
**Target:** `hpm_ai_v2/agents/base_agent.py`
- Add `auto_save_frequency: int = 0` to the `BaseHFNAgent.__init__` parameters.
- Initialize `self._solve_counter = 0`.
- Add `_maybe_save_state(self)` method to call `self.save_state()` periodically based on `auto_save_frequency`.
- Update the `solve()` method: After a successful solution, call `_maybe_save_state()`.

## Verification & Testing
- The default behavior (frequency = 0) remains fully backward compatible. Existing experiments will run without changes.
- To verify, run any agent with `auto_observe_frequency > 0` and `auto_save_frequency > 0` and assert that state files are generated periodically and the `observer` handles incoming samples.
