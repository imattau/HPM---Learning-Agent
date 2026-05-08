# Polygraph View Refactor: view_matches + conditional view_engines

**Date**: 2026-05-08
**Branch**: hpm-ai-v5
**Status**: Approved for implementation

---

## Problem

`HPMPipeline` creates a full `PatternEngine` instance per polygraph view — each with its
own `PatternStore`, history, and sequence state. With `NLPPolygraphGenerator` producing
up to 50+ `semantic_view_*` engines on large corpora (5 candidates × 10 tokens per
utterance), this causes view engine explosion: 24 engines after 11 utterances, thousands
after a full corpus run. Each engine runs `observe()` on every subsequent step.

Most callers only need `last_match` from a view — they never call `view_engine.act()`.
Creating a full `PatternEngine` for this is wasteful.

---

## Design

Two-tier polygraph system:

### Tier 1: view_matches (default, all benchmarks)

Replace `view_engines: dict[str, PatternEngine]` with `view_matches: dict[str, MatchResult]`.

The primary engine observes all view states via `observe(state, update_state=False)`:
- Learns patterns (adds to primary PatternStore)
- Records `last_match`
- Does NOT update `current_state` or `history`

After each view observe: `pipeline.view_matches[view.name] = engine.last_match`

All callers that read `view_engine.last_match` migrate to `view_matches.get(view_name)`.

### Tier 2: view_engines (PAB, CartPole only — when polygraph_evaluator active)

When `polygraph_evaluator` is set AND a view is selected for `act()`, create a full
view engine for that view only. This preserves PAB and CartPole behaviour unchanged.

**Condition**: full view engine only created when `selected_view` needs `act()`.

---

## PatternEngine change

Add `update_state: bool = True` parameter to `observe()`:

```python
def observe(self, state: State, *, update_state: bool = True) -> MatchResult | None:
    ...
    if update_state:
        self.current_state = state
        self.history.append(state)
    ...
```

When `update_state=False`: learns pattern, returns match, leaves `current_state` unchanged.

---

## Files to Change

| File | Change |
|------|--------|
| `hpm_ai_v5/core/engine.py` | Add `update_state=True` to `observe()` |
| `hpm_ai_v5/pipeline.py` | Add `view_matches: dict[str, MatchResult]`; primary engine observes views; full engines only for act() |
| `hpm_ai_v5/experiments/run_snlp_benchmark.py` | `view_engines.get(n).last_match` → `view_matches.get(n)` |
| `hpm_ai_v5/experiments/run_atis_benchmark.py` | Remove `view_engines.values()` iteration |
| `hpm_ai_v5/planning/code_recognition.py` | `view_engines.get(n).last_match` → `view_matches.get(n)` |
| `hpm_ai_v5/tests/test_cartpole.py` | Keep — PAB/CartPole still use full view engines |

---

## Backward Compatibility

- `pipeline.view_engines` kept for PAB/CartPole path (when evaluator active)
- `pipeline.view_matches` added as new primary interface
- Callers that only need `last_match` migrate to `view_matches`
- No changes to `Action`, `MatchResult`, `PatternEngine.act()`

---

## Success Criteria

- All existing tests pass (no regressions)
- SNLP benchmark: T1-T5 scores unchanged
- View engine count for ATIS with `StructuralNLPPolygraphGenerator`: still 0 (no full engines)
- PAB benchmark: `selected_view` still works correctly
- CartPole `test_cartpole.py:201` still passes

---

## Out of Scope

- Merging PAB view engines into primary store (separate architectural task)
- Changing how `act()` works across views
- Removing `view_engines` entirely (kept for evaluator path)
