# Task Input Perception Layer — Design Spec

**Date:** 2026-04-23
**Branch:** hpm-ai-v3-dev
**Status:** Ready for implementation

---

## Problem

Training logs show 97% of errors are caused by the `arithmetic` tool being selected for all task types (strings, booleans, lists) and failing. The agent has no mechanism to perceive what kind of input it is receiving before selecting a tool. This is an L1 perception gap — the agent cannot classify sensory input before acting.

In HPM terms: pattern dynamics (selection + execution) are running without a perception stage. The pattern evaluator (ToolSelector) boosts tools by semantic similarity, but similarity alone is insufficient when all tool descriptions share vocabulary with all task descriptions. A hard compatibility gate is needed before soft scoring.

---

## Solution Overview

Two-component solution:

1. **TaskPerceptor** — innate, always-on classifier that runs before tool selection every episode. Produces a structured percept describing the input type and operation verb.
2. **ToolCompatibilityFilter** — hard-gate added to ToolSelector that zeroes weights for tools incompatible with the percept before semantic similarity scoring applies.

A separate arithmetic validator fix is included: the current validator correctly requires at least one digit, but bare floats like `"42.5"` and integers like `"10"` should be returned directly without further arithmetic parsing. The validator needs to handle the pure-number case.

---

## Component 1: TaskPerceptor

**File:** `hpm_ai_v3/tools/task_perceptor.py`

### Input types detected

| Type | Trigger condition |
|------|-----------------|
| `"numeric"` | Text contains one or more parseable numbers (no operators) |
| `"expression"` | Text contains arithmetic operators `+`, `-`, `*`, `/` between numbers |
| `"string"` | Text references string manipulation ("Is '...' a string?", "Count words", "reverse", "upper") |
| `"boolean"` | Text is a yes/no question ("Is...", "Does...", "Are...") |
| `"list"` | Text contains comma-separated values or bracket notation `[...]` |
| `"mixed"` | Numeric values plus non-numeric text that is not a pure string task |

Detection is priority-ordered: `expression` > `list` > `boolean` > `string` > `numeric` > `mixed`.

### Operations detected (verb extraction)

| Operation | Trigger words |
|-----------|--------------|
| `"compute"` | calculate, evaluate, solve, "what is", compute |
| `"classify"` | is, check, determine, identify, type |
| `"extract"` | find, get, count, list, extract |
| `"compare"` | compare, greater, less, equal, difference |
| `"transform"` | split, join, reverse, sort, upper, lower, strip |

### Output percept

```python
{
    "input_type": str,       # one of the six types above
    "operation": str,        # one of the five operations above
    "numeric_values": list,  # all floats parsed from text (may be empty)
    "tokens": list,          # lowercased whitespace-split words
    "is_question": bool,     # True if text ends with "?"
}
```

### Design constraints

- No external dependencies — stdlib only (`re`, `ast`).
- Pure function: stateless, deterministic, no side effects.
- Must run in < 1ms on a 200-character task string.
- Returns a valid percept dict for any input including empty string.

---

## Component 2: ToolCompatibilityFilter

**Location:** `hpm_ai_v3/tools/tool_selector.py` — new method on `ToolSelector`.

### Compatibility table

| Input type | Allowed tool names |
|------------|-------------------|
| `numeric` | `arithmetic`, `float`, `int`, `extract_numbers`, `math.*` via python_call |
| `expression` | `arithmetic`, `math.*` via python_call |
| `string` | `split`, `re_findall`, `str.*` via python_call, `get_type` |
| `boolean` | `get_type`, `str.*` via python_call |
| `list` | `index`, `arithmetic` (on extracted nums), `extract_numbers`, python_call with `len` |
| `mixed` | all tools allowed (no filtering) |

### Behaviour

- `filter_by_percept(percept, weights, patterns)` returns a new weights array.
- Any pattern whose tool_name (or module.function) is NOT in the allowed set for the detected input_type has its weight multiplied by `0.0`.
- At least one pattern must survive filtering. If all patterns would be zeroed, the filter is skipped (returns original weights) — fail-safe.
- The filter runs before `apply()` semantic similarity scoring, so semantic scores are applied only to the surviving set.

---

## Component 3: Integration in act()

**File:** `hpm_ai_v3/agents/base_discovery.py`

At the start of `act()`, before weight computation:

```python
self.current_percept = self.substrate.perceive_task(task_text)
```

Then in the weight computation block, after getting raw weights and before calling `tool_selector.apply()`:

```python
if self.tool_selector is not None and self.current_percept:
    weights = self.tool_selector.filter_by_percept(
        self.current_percept, weights, self.population.patterns
    )
```

`perceive_task` is added to `InnateCognitiveSubstrate` as a Group H method — it instantiates `TaskPerceptor` once and delegates.

---

## Component 4: Arithmetic Validator Fix

**File:** `hpm_ai_v3/tools/innate.py`, function `arithmetic_eval`

Current code checks `any(c.isdigit() for c in s_expr)` but then passes the full string to `simple_eval`. When the input is a bare number like `"42.5"` or `"10"`, `simple_eval` works but the caller may see an unexpected result type. The real issue is that `"Is 'hello' a palindrome?"` contains no digits, which is caught — but strings like `"hello world"` with no digits still reach the validator and can produce confusing errors.

**Fix:** After the digit check, attempt `float(s_expr)` first. If it succeeds, return `{"result": float(s_expr), "status": "success"}` immediately. This handles bare numbers without invoking the expression evaluator.

---

## Files Summary

| Action | File | Purpose |
|--------|------|---------|
| CREATE | `hpm_ai_v3/tools/task_perceptor.py` | TaskPerceptor class |
| MODIFY | `hpm_ai_v3/tools/tool_selector.py` | add `filter_by_percept()` |
| MODIFY | `hpm_ai_v3/tools/innate_substrate.py` | add `perceive_task()` Group H |
| MODIFY | `hpm_ai_v3/agents/base_discovery.py` | call percept in `act()` |
| MODIFY | `hpm_ai_v3/tools/innate.py` | bare-number fast-path in `arithmetic_eval` |
| CREATE | `hpm_ai_v3/tests/test_task_perceptor.py` | unit test suite |

---

## Test Coverage Requirements

- TaskPerceptor: one test per input_type (6), one per operation (5), edge cases (empty string, pure number, expression with spaces)
- ToolCompatibilityFilter: one test per input_type that confirms only allowed tools survive
- arithmetic_eval fix: bare float, bare int, bare negative
- Integration smoke test: act() produces `current_percept` attribute after calling act()
