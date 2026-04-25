# Learning Loop Fixes — Design Spec
Date: 2026-04-23
Branch: hpm-ai-v3-dev

## Overview

Four bugs are preventing the hpm_ai_v3 training loop from producing meaningful learning signal. All are P0/P1 blockers: two corrupt the reward signal directly, one causes premature curriculum advancement before any learning occurs, and one crashes probabilistic-reasoning task evaluation. This spec details root causes, invariants, and acceptance criteria for each fix.

---

## Fix 1 (P0): Reward Signal Stuck at -1.0

### Symptom
`mean_reward=-1.000` persists across all episodes. Expected values are `1.0` (correct) or `-0.5` (wrong answer).

### Root Cause
`DiscoveryAgent.evaluate_solution()` at line 134:

```python
if solution is None: return -1.0
```

The `-1.0` sentinel is returned when `solution is None`. This is a separate code path from the binary `1.0 / -0.5` reward computed at the call site in `act()`. The agent's population patterns are producing `None` solutions (not wrong answers — they are not producing any answer), triggering the early-exit sentinel.

The sentinel value `-1.0` leaks into the reward stream, but is semantically distinct: it means "no solution produced" rather than "wrong solution". Downstream code (CurriculumManager, MetaCognitive) treats all rewards uniformly, so `-1.0` severely biases the running average below the -0.5 floor of the actual reward range.

### Invariants
- `evaluate_solution()` must only return values in `{1.0, -0.5}` for the training loop to receive valid learning signal.
- The `None` / exception cases should return `-0.5` (failed attempt), not `-1.0`.
- A separate `is_null_solution` flag or logging may track null solutions for diagnostics without corrupting reward.

### Acceptance Criteria
1. After fix, `mean_reward` converges toward -0.5 (baseline of random wrong answers) rather than -1.0.
2. `evaluate_solution(None)` returns `-0.5`.
3. Exception paths within the try/except in `evaluate_solution()` return `-0.5`, not propagate to caller as None.
4. Existing unit tests for reward range pass.

---

## Fix 2 (P0): InnateCognitiveSubstrate `list_index` Missing Argument

### Symptom
`list_index() missing 1 required positional argument`

### Root Cause
`InnateCognitiveSubstrate` exposes `find_in_list(value, lst)` but agents call it via `resolve_call()` which uses `inspect_signature()` to discover parameter types. The method signature is:

```python
def find_in_list(self, value: Any, lst: list) -> Optional[int]:
```

Two sub-problems:

1. **Alias mismatch**: Agents (and possibly curriculum hints) reference the tool as `list_index` but the method is named `find_in_list`. If a tool alias `list_index` is registered in the registry pointing at a different callable (or not registered at all), the call fails with a missing-argument error.

2. **resolve_call arg resolution**: When `resolve_call` introspects `find_in_list`, the `lst: list` parameter gets `target_type = "list"`. The pool may not contain a plain Python list (it may contain a string representation), so `coerce(val, "list")` converts a string to `list(string)` (a list of characters), which is wrong. The `value` parameter (`Any` type) may also be filled with the wrong pool item, leaving `lst` unresolved.

### Invariants
- Any name exposed to agents as `list_index` must route to `find_in_list(value, lst)`.
- `resolve_call` must not silently pass wrong-typed args; it should raise `TypeError` with a descriptive message if required args cannot be resolved.
- The substrate method itself is correct; only the registration/routing layer needs fixing.

### Acceptance Criteria
1. `innate.find_in_list(3, [1, 2, 3])` returns `2`.
2. Calling via the name `list_index` through the registry (if registered) routes correctly to `find_in_list`.
3. `resolve_call("hpm_ai_v3.tools.innate_substrate", "find_in_list", pool=[3, [1,2,3]], task_text="...")` returns `(module, "find_in_list", [3, [1,2,3]])`.
4. Calling with a missing second arg raises `TypeError: find_in_list requires 'lst' argument` (not a bare Python argument error).

---

## Fix 3 (P1): Phase Advancing Despite 0% Success

### Symptom
Agent advances through phases Sensory Priming → Structural → Sequential → Linguistic with 0% task success.

### Root Cause — Two independent paths trigger advancement:

**Path A — `CurriculumManager.update()`** (curriculum.py line 113):
```python
if avg_reward >= 0.8 and len(self.recent_rewards) >= 5:
    self.active_pattern_idx += 1
```
This is gated correctly at 0.8 — it should NOT advance at 0%. But `recent_rewards` is initialised empty and `avg_reward` defaults to `0.0`. If `update()` is called before any rewards are appended, `len(self.recent_rewards) < 5` so this path is safe.

**Path B — `MetaCognitivePattern._do_advance_phase()`** (meta_cognitive_pattern.py line 266):
```python
def _do_advance_phase(self, agent, curriculum):
    if hasattr(curriculum, "advance_phase"):
        curriculum.advance_phase()
        agent._steps_since_advance = 0
```
This executes whenever the REINFORCE policy samples `ADVANCE_PHASE` (directive index 3). The policy is randomly initialised — `xavier_uniform_` weights produce roughly uniform action probabilities (~12.5% each). With no constraint, `ADVANCE_PHASE` fires on ~1/8 meta-steps regardless of actual competency.

The meta-reward in `compute_meta_reward()` awards `+2.0` for `phase_now > self._prev_phase`, which reinforces premature advancement. This creates a feedback loop: advance early → get +2.0 → policy learns to advance more.

### Fix
Gate `_do_advance_phase` on minimum competency:

```python
def _do_advance_phase(self, agent, curriculum):
    history = getattr(agent, "_meta_success_history", [])
    current_accuracy = float(np.mean(history)) if history else 0.0
    if current_accuracy < 0.3:
        print(f"  [MetaCognitive] ADVANCE_PHASE blocked: accuracy={current_accuracy:.2f} < 0.30")
        return
    if hasattr(curriculum, "advance_phase"):
        curriculum.advance_phase()
        agent._steps_since_advance = 0
```

Threshold of 0.3 (30% success rate over recent history) is the minimum signal that the agent is doing better than chance on binary tasks.

### Invariants
- Phase advancement via MetaCognitive directive must require `current_accuracy >= 0.3`.
- The `CurriculumManager.update()` mastery threshold (0.8) is not changed — it is already correct.
- The meta-reward bonus for phase advancement should not apply to blocked attempts.

### Acceptance Criteria
1. With `_meta_success_history = [0.0] * 20`, `_do_advance_phase` does not advance curriculum.
2. With `_meta_success_history = [1.0] * 10 + [0.0] * 5` (mean=0.667 > 0.3), `_do_advance_phase` advances.
3. Training log no longer shows phase advancement in first 50 episodes with 0% success.

---

## Fix 4 (P1): Probabilistic Reasoning Task Answer Format

### Symptom
`could not convert string to float: 'Normalize [1, ...'`

### Root Cause
Inspection of `05_probabilistic_reasoning.json` shows all tasks already have correct numeric answers:

```json
{"text": "Normalize [1, 1, 2]", "answer": [0.25, 0.25, 0.5], ...}
{"text": "What is the entropy of [0.5, 0.5]?", "answer": 1.0, ...}
```

The crash is not in the JSON answers themselves — they are correct. The crash occurs in `evaluate_solution()` when the **solution returned by the agent** is a string like `"Normalize [1, 1, 2]"` (the agent echoes the task text rather than computing a result), and the evaluator calls `float(solution)` on it.

This is caused by the evaluator's float-conversion fallback path in the numeric comparison branch:

```python
# Somewhere in evaluate_solution:
return 1.0 if abs(float(solution) - float(answer)) < tol else -0.5
```

When `solution` is a non-numeric string, `float(solution)` raises `ValueError`. The fix is twofold:
1. Guard the float conversion with a try/except returning `-0.5` on parse failure.
2. For list answers, use element-wise comparison rather than casting to float.

### Invariants
- `evaluate_solution()` must never raise uncaught exceptions — all paths return a float in `[-1.0, 1.0]` (after Fix 1, `[-0.5, 1.0]`).
- List answers (`answer: [0.25, 0.25, 0.5]`) must be compared element-wise with tolerance.
- String solutions that cannot be parsed to the expected answer type return `-0.5`.

### Acceptance Criteria
1. `evaluate_solution("Normalize [1, 1, 2]")` against answer `[0.25, 0.25, 0.5]` returns `-0.5` (not crash).
2. `evaluate_solution([0.25, 0.25, 0.5])` against answer `[0.25, 0.25, 0.5]` returns `1.0`.
3. `evaluate_solution(1.0)` against answer `1.0` returns `1.0`.
4. No `ValueError` or `TypeError` is ever raised by `evaluate_solution`.

---

## Integration Invariants (All Fixes Combined)

- After all 4 fixes, `mean_reward` must be > -0.5 within 100 training episodes on the Probabilistic Reasoning phase.
- Phase index must not increase in the first 30 episodes when starting from random initialisation.
- No uncaught exceptions in `evaluate_solution()` across all curriculum task types.
- All 4 fixes have isolated unit tests that can be run individually.
