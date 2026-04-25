# Task Input Perception Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a TaskPerceptor that classifies task input type before tool selection, and a ToolCompatibilityFilter that gates incompatible tools to zero weight — eliminating the 97% arithmetic-tool-misfire error rate.

**Architecture:** TaskPerceptor runs as a pure-function classifier (stdlib only) producing a typed percept dict. ToolSelector gains a `filter_by_percept()` method that zeros weights for incompatible tools before semantic scoring. InnateCognitiveSubstrate exposes `perceive_task()` as a Group H method. base_discovery.py `act()` calls perceive at the top of each episode.

**Tech Stack:** Python stdlib (`re`, `ast`), numpy (already used in ToolSelector), pytest.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| CREATE | `hpm_ai_v3/tools/task_perceptor.py` | TaskPerceptor — input_type + operation detection |
| MODIFY | `hpm_ai_v3/tools/tool_selector.py` | add `filter_by_percept()` + `TOOL_COMPAT` table |
| MODIFY | `hpm_ai_v3/tools/innate_substrate.py` | add `perceive_task()` Group H method |
| MODIFY | `hpm_ai_v3/agents/base_discovery.py` | call percept at top of `act()`, pass to filter |
| MODIFY | `hpm_ai_v3/tools/innate.py` | bare-number fast-path in `arithmetic_eval` |
| CREATE | `hpm_ai_v3/tests/test_task_perceptor.py` | full unit test suite |

---

## Task 1: TaskPerceptor class

**Files:**
- Create: `hpm_ai_v3/tools/task_perceptor.py`
- Test: `hpm_ai_v3/tests/test_task_perceptor.py`

- [ ] **Step 1: Write failing tests for input_type detection**

Create `hpm_ai_v3/tests/test_task_perceptor.py` with this content:

```python
import pytest
from hpm_ai_v3.tools.task_perceptor import TaskPerceptor

@pytest.fixture
def p():
    return TaskPerceptor()

def test_numeric_type(p):
    percept = p.perceive("42.5")
    assert percept["input_type"] == "numeric"

def test_expression_type(p):
    percept = p.perceive("3 + 4 * 2")
    assert percept["input_type"] == "expression"

def test_string_type(p):
    percept = p.perceive("Is 'hello' a palindrome?")
    assert percept["input_type"] == "string"

def test_boolean_type(p):
    percept = p.perceive("Is 7 greater than 5?")
    assert percept["input_type"] == "boolean"

def test_list_type(p):
    percept = p.perceive("Sort the list [3, 1, 2]")
    assert percept["input_type"] == "list"

def test_mixed_type(p):
    percept = p.perceive("The temperature is 98.6 degrees")
    assert percept["input_type"] == "mixed"

def test_empty_string(p):
    percept = p.perceive("")
    assert percept["input_type"] == "mixed"
    assert percept["numeric_values"] == []
    assert percept["tokens"] == []
    assert percept["is_question"] is False
```

- [ ] **Step 2: Run tests to confirm they fail**

```
PYTHONPATH=. pytest hpm_ai_v3/tests/test_task_perceptor.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'hpm_ai_v3.tools.task_perceptor'`

- [ ] **Step 3: Write TaskPerceptor implementation**

Create `hpm_ai_v3/tools/task_perceptor.py`:

```python
"""
TaskPerceptor — innate, always-on input classifier for HPM agents.
Runs before tool selection every episode. No external dependencies.
"""
import re
from typing import Any, Dict, List

# Priority order: expression > list > boolean > string > numeric > mixed
_OPERATOR_RE = re.compile(r'\d\s*[+\-*/]\s*\d')
_LIST_RE = re.compile(r'\[.*?\]|(\d+\s*,\s*){2,}\d+')
_BOOL_STARTERS = re.compile(r'^(is|does|are|can|has|have|will|was|were|did)\b', re.IGNORECASE)
_STRING_KEYWORDS = re.compile(
    r'\b(palindrome|string|character|word|letter|sentence|upper|lower|reverse|split|join|strip|capitalize|replace|substr|prefix|suffix|concat)\b',
    re.IGNORECASE
)
_NUMBER_RE = re.compile(r'-?\d+\.?\d*')

_OPERATION_PATTERNS = [
    ("compute",   re.compile(r'\b(calculate|compute|what\s+is)\b', re.IGNORECASE)),
    ("classify",  re.compile(r'\b(determine|identify|type\s+of)\b', re.IGNORECASE)),
    ("extract",   re.compile(r'\b(find|get|count|list|extract)\b', re.IGNORECASE)),
    ("compare",   re.compile(r'\b(compare|greater|less|equal|difference|larger|smaller)\b', re.IGNORECASE)),
    ("transform", re.compile(r'\b(split|join|reverse|sort|upper|lower|strip|replace)\b', re.IGNORECASE)),
]


class TaskPerceptor:
    """Classify task text into a structured percept dict."""

    def perceive(self, text: Any) -> Dict:
        """Return percept dict for any input. Never raises."""
        s = str(text).strip() if text is not None else ""
        tokens = s.lower().split() if s else []
        numeric_values = [float(n) for n in _NUMBER_RE.findall(s)]
        is_question = s.endswith("?")

        input_type = self._detect_type(s, numeric_values)
        operation = self._detect_operation(s)

        return {
            "input_type": input_type,
            "operation": operation,
            "numeric_values": numeric_values,
            "tokens": tokens,
            "is_question": is_question,
        }

    def _detect_type(self, s: str, numeric_values: List[float]) -> str:
        if not s:
            return "mixed"
        if _OPERATOR_RE.search(s):
            return "expression"
        if _LIST_RE.search(s):
            return "list"
        if _BOOL_STARTERS.match(s) and not _STRING_KEYWORDS.search(s):
            return "boolean"
        if _STRING_KEYWORDS.search(s):
            return "string"
        stripped_nums = _NUMBER_RE.sub("", s).strip()
        non_numeric_words = [w for w in stripped_nums.split() if w not in (",", ".", "and", "the")]
        if numeric_values and len(non_numeric_words) == 0:
            return "numeric"
        if numeric_values:
            return "mixed"
        return "mixed"

    def _detect_operation(self, s: str) -> str:
        for op_name, pattern in _OPERATION_PATTERNS:
            if pattern.search(s):
                return op_name
        return "compute"
```

- [ ] **Step 4: Run tests to confirm they pass**

```
PYTHONPATH=. pytest hpm_ai_v3/tests/test_task_perceptor.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v3/tools/task_perceptor.py hpm_ai_v3/tests/test_task_perceptor.py
git commit -m "feat: add TaskPerceptor — input type and operation detection"
```

---

## Task 2: Operation detection and percept structure tests

**Files:**
- Test: `hpm_ai_v3/tests/test_task_perceptor.py` (append)

- [ ] **Step 1: Append operation and structure tests**

Add to the bottom of `hpm_ai_v3/tests/test_task_perceptor.py`:

```python
def test_operation_compute(p):
    assert p.perceive("What is 5 + 3?")["operation"] == "compute"

def test_operation_classify(p):
    assert p.perceive("Determine the type of 42")["operation"] == "classify"

def test_operation_extract(p):
    assert p.perceive("Count the words in this sentence")["operation"] == "extract"

def test_operation_compare(p):
    assert p.perceive("Is 7 greater than 5?")["operation"] == "compare"

def test_operation_transform(p):
    assert p.perceive("Reverse the string hello")["operation"] == "transform"

def test_numeric_values_extracted(p):
    percept = p.perceive("Add 10 and 20")
    assert 10.0 in percept["numeric_values"]
    assert 20.0 in percept["numeric_values"]

def test_is_question_true(p):
    assert p.perceive("Is 3 prime?")["is_question"] is True

def test_is_question_false(p):
    assert p.perceive("Calculate 3 + 4")["is_question"] is False

def test_tokens_lowercase(p):
    percept = p.perceive("Hello World")
    assert "hello" in percept["tokens"]
    assert "world" in percept["tokens"]

def test_bare_number_numeric(p):
    assert p.perceive("42")["input_type"] == "numeric"

def test_expression_with_spaces(p):
    assert p.perceive("10 + 20 - 3")["input_type"] == "expression"
```

- [ ] **Step 2: Run all perceptor tests**

```
PYTHONPATH=. pytest hpm_ai_v3/tests/test_task_perceptor.py -v
```

Expected: all 18 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v3/tests/test_task_perceptor.py
git commit -m "test: add operation detection and percept structure tests"
```

---

## Task 3: ToolCompatibilityFilter on ToolSelector

**Files:**
- Modify: `hpm_ai_v3/tools/tool_selector.py`
- Test: `hpm_ai_v3/tests/test_task_perceptor.py` (append)

- [ ] **Step 1: Write failing filter tests**

Append to `hpm_ai_v3/tests/test_task_perceptor.py`:

```python
import numpy as np
from unittest.mock import MagicMock
from hpm_ai_v3.tools.tool_selector import ToolSelector

def _make_pattern(tool_name):
    pat = MagicMock()
    pat.tool_name = tool_name
    pat.module = None
    pat.function = None
    pat.action_type = tool_name
    pat.weight = 1.0
    return pat

def _make_selector():
    lm = MagicMock()
    lm._embed.return_value = [0.1] * 10
    return ToolSelector(lm=lm, alpha=1.0)

def test_filter_numeric_blocks_split():
    sel = _make_selector()
    patterns = [_make_pattern("arithmetic"), _make_pattern("split"), _make_pattern("float")]
    weights = np.array([1.0, 1.0, 1.0])
    percept = {"input_type": "numeric", "operation": "compute",
               "numeric_values": [42.0], "tokens": [], "is_question": False}
    result = sel.filter_by_percept(percept, weights, patterns)
    assert result[0] > 0.0   # arithmetic allowed
    assert result[1] == 0.0  # split zeroed
    assert result[2] > 0.0   # float allowed

def test_filter_string_blocks_arithmetic():
    sel = _make_selector()
    patterns = [_make_pattern("arithmetic"), _make_pattern("split"), _make_pattern("re_findall")]
    weights = np.array([1.0, 1.0, 1.0])
    percept = {"input_type": "string", "operation": "transform",
               "numeric_values": [], "tokens": [], "is_question": False}
    result = sel.filter_by_percept(percept, weights, patterns)
    assert result[0] == 0.0  # arithmetic zeroed
    assert result[1] > 0.0   # split allowed
    assert result[2] > 0.0   # re_findall allowed

def test_filter_mixed_allows_all():
    sel = _make_selector()
    patterns = [_make_pattern("arithmetic"), _make_pattern("split")]
    weights = np.array([1.0, 1.0])
    percept = {"input_type": "mixed", "operation": "compute",
               "numeric_values": [], "tokens": [], "is_question": False}
    result = sel.filter_by_percept(percept, weights, patterns)
    assert result[0] > 0.0
    assert result[1] > 0.0

def test_filter_failsafe_when_all_would_zero():
    sel = _make_selector()
    # All patterns are "split" but input is numeric
    patterns = [_make_pattern("split"), _make_pattern("split")]
    weights = np.array([1.0, 1.0])
    percept = {"input_type": "numeric", "operation": "compute",
               "numeric_values": [1.0], "tokens": [], "is_question": False}
    result = sel.filter_by_percept(percept, weights, patterns)
    # Failsafe: returns original weights when all would be zeroed
    assert result[0] > 0.0
    assert result[1] > 0.0
```

- [ ] **Step 2: Run to confirm they fail**

```
PYTHONPATH=. pytest hpm_ai_v3/tests/test_task_perceptor.py -k "filter" -v
```

Expected: `AttributeError: 'ToolSelector' object has no attribute 'filter_by_percept'`

- [ ] **Step 3: Add TOOL_COMPAT table to tool_selector.py**

In `hpm_ai_v3/tools/tool_selector.py`, add this block after the `TOOL_DESCRIPTIONS` dict (before `class ToolSelector:`):

```python
# Tools allowed per input_type. None means "all allowed" (no filter).
TOOL_COMPAT = {
    "numeric":    {"arithmetic", "float", "int", "extract_numbers",
                   "math.sqrt", "math.factorial", "math.floor", "math.gcd",
                   "operator.add", "operator.mul", "sympy.sympify"},
    "expression": {"arithmetic", "math.sqrt", "math.factorial", "math.floor",
                   "math.gcd", "operator.add", "operator.mul", "sympy.sympify"},
    "string":     {"split", "re_findall", "get_type",
                   "builtins.str.split", "builtins.str.lower", "builtins.str.upper",
                   "spacy.nlp", "language_model", "textblob.TextBlob"},
    "boolean":    {"get_type", "builtins.str.split", "builtins.str.lower", "builtins.str.upper"},
    "list":       {"index", "arithmetic", "extract_numbers", "float", "int",
                   "builtins.str.split"},
    "mixed":      None,  # no filtering
}
```

- [ ] **Step 4: Add filter_by_percept method to ToolSelector class**

Add this method at the end of the `ToolSelector` class in `hpm_ai_v3/tools/tool_selector.py` (after `_pattern_description`):

```python
    def filter_by_percept(self, percept: Dict, weights: np.ndarray,
                          patterns: List[Any]) -> np.ndarray:
        """Zero weights for tools incompatible with percept input_type.

        Fail-safe: if filtering would zero all patterns, returns original weights.
        """
        input_type = percept.get("input_type", "mixed")
        allowed = TOOL_COMPAT.get(input_type)
        if allowed is None:
            return weights  # mixed — no filtering

        new_weights = weights.copy()
        for i, pat in enumerate(patterns):
            tool_name = getattr(pat, 'tool_name', None)
            module = getattr(pat, 'module', None)
            function = getattr(pat, 'function', None)
            candidates = set()
            if tool_name:
                candidates.add(tool_name)
            if module and function:
                candidates.add(f"{module}.{function}")
            if not candidates.intersection(allowed):
                new_weights[i] = 0.0

        # Fail-safe: if all zeroed, return original
        if new_weights.sum() < 1e-8:
            return weights
        return new_weights
```

- [ ] **Step 5: Run filter tests**

```
PYTHONPATH=. pytest hpm_ai_v3/tests/test_task_perceptor.py -k "filter" -v
```

Expected: all 4 filter tests PASS.

- [ ] **Step 6: Run full test suite to check for regressions**

```
PYTHONPATH=. pytest hpm_ai_v3/tests/ -v
```

Expected: previously passing tests still PASS.

- [ ] **Step 7: Commit**

```bash
git add hpm_ai_v3/tools/tool_selector.py hpm_ai_v3/tests/test_task_perceptor.py
git commit -m "feat: add ToolCompatibilityFilter to ToolSelector"
```

---

## Task 4: perceive_task() on InnateCognitiveSubstrate

**Files:**
- Modify: `hpm_ai_v3/tools/innate_substrate.py`

- [ ] **Step 1: Add Group H section with perceive_task() at end of class**

At the very end of the `InnateCognitiveSubstrate` class in `hpm_ai_v3/tools/innate_substrate.py`, add:

```python
    # ── Group H: Task Perception ───────────────────────────────────────────

    def perceive_task(self, text: str) -> dict:
        """Classify task text into a structured percept. Always returns a valid dict."""
        from hpm_ai_v3.tools.task_perceptor import TaskPerceptor
        if not hasattr(self, '_task_perceptor'):
            self._task_perceptor = TaskPerceptor()
        return self._task_perceptor.perceive(text)
```

- [ ] **Step 2: Run existing substrate tests to check for regressions**

```
PYTHONPATH=. pytest hpm_ai_v3/tests/ -v
```

Expected: all existing tests PASS.

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v3/tools/innate_substrate.py
git commit -m "feat: add perceive_task() Group H method to InnateCognitiveSubstrate"
```

---

## Task 5: Integration in base_discovery.py act()

**Files:**
- Modify: `hpm_ai_v3/agents/base_discovery.py`

- [ ] **Step 1: Update act() to call perceive and filter before semantic scoring**

In `hpm_ai_v3/agents/base_discovery.py`, find this block in `act()`:

```python
        # 1. SELECTION (Evaluator-Gated Replicator)
        weights = np.array([p.weight for p in self.population.patterns])
        if self.tool_selector is not None and self.current_task:
            weights = self.tool_selector.apply(
                self.current_task.get("text", ""),
                weights,
                self.population.patterns
            )
```

Replace with:

```python
        # 1. SELECTION (Evaluator-Gated Replicator)
        task_text_for_percept = self.current_task.get("text", "") if self.current_task else ""
        self.current_percept = self.substrate.perceive_task(task_text_for_percept)

        weights = np.array([p.weight for p in self.population.patterns])
        if self.tool_selector is not None and self.current_task:
            # Hard gate: zero incompatible tools before soft semantic scoring
            weights = self.tool_selector.filter_by_percept(
                self.current_percept, weights, self.population.patterns
            )
            weights = self.tool_selector.apply(
                self.current_task.get("text", ""),
                weights,
                self.population.patterns
            )
```

- [ ] **Step 2: Run full test suite**

```
PYTHONPATH=. pytest hpm_ai_v3/tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v3/agents/base_discovery.py
git commit -m "feat: integrate TaskPerceptor into act() — percept gates tool selection"
```

---

## Task 6: Arithmetic validator bare-number fix

**Files:**
- Modify: `hpm_ai_v3/tools/innate.py`
- Test: `hpm_ai_v3/tests/test_task_perceptor.py` (append)

- [ ] **Step 1: Write arithmetic_eval tests**

Append to `hpm_ai_v3/tests/test_task_perceptor.py`:

```python
from hpm_ai_v3.tools.innate import arithmetic_eval

def test_arithmetic_bare_float():
    result = arithmetic_eval("42.5")
    assert result["status"] == "success"
    assert result["result"] == 42.5

def test_arithmetic_bare_int():
    result = arithmetic_eval("10")
    assert result["status"] == "success"
    assert result["result"] == 10.0

def test_arithmetic_bare_negative():
    result = arithmetic_eval("-7")
    assert result["status"] == "success"
    assert result["result"] == -7.0

def test_arithmetic_expression_still_works():
    result = arithmetic_eval("3 + 4")
    assert result["status"] == "success"
    assert result["result"] == 7

def test_arithmetic_rejects_pure_text():
    result = arithmetic_eval("hello world")
    assert result["status"] == "failed"
```

- [ ] **Step 2: Run to check which tests fail**

```
PYTHONPATH=. pytest hpm_ai_v3/tests/test_task_perceptor.py -k "arithmetic" -v
```

Note the results. Proceed to apply the fix regardless.

- [ ] **Step 3: Apply bare-number fast-path in arithmetic_eval**

In `hpm_ai_v3/tools/innate.py`, in `arithmetic_eval`, find:

```python
    if not any(c.isdigit() for c in s_expr):
        return {"error": f"Expression '{s_expr}' does not appear to be arithmetic.", "status": "failed"}
```

Replace with:

```python
    if not any(c.isdigit() for c in s_expr):
        return {"error": f"Expression '{s_expr}' does not appear to be arithmetic.", "status": "failed"}

    # Fast path: bare number — convert directly without expression parsing
    try:
        bare = float(s_expr)
        return {"result": bare, "status": "success"}
    except ValueError:
        pass  # not a bare number — fall through to expression parsing
```

- [ ] **Step 4: Run arithmetic tests**

```
PYTHONPATH=. pytest hpm_ai_v3/tests/test_task_perceptor.py -k "arithmetic" -v
```

Expected: all 5 arithmetic tests PASS.

- [ ] **Step 5: Run full suite**

```
PYTHONPATH=. pytest hpm_ai_v3/tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add hpm_ai_v3/tools/innate.py hpm_ai_v3/tests/test_task_perceptor.py
git commit -m "fix: arithmetic_eval bare-number fast-path — return float directly for pure numbers"
```

---

## Task 7: Final validation smoke test

**Files:**
- Test: `hpm_ai_v3/tests/test_task_perceptor.py` (append)

- [ ] **Step 1: Append percept field validation test**

Append to `hpm_ai_v3/tests/test_task_perceptor.py`:

```python
def test_perceptor_standalone_all_fields():
    """All percept fields present, correctly typed, and values in valid sets."""
    perceptor = TaskPerceptor()
    percept = perceptor.perceive("What is 3 + 4?")
    assert isinstance(percept["input_type"], str)
    assert isinstance(percept["operation"], str)
    assert isinstance(percept["numeric_values"], list)
    assert isinstance(percept["tokens"], list)
    assert isinstance(percept["is_question"], bool)
    valid_types = {"numeric", "expression", "string", "boolean", "list", "mixed"}
    valid_ops = {"compute", "classify", "extract", "compare", "transform"}
    assert percept["input_type"] in valid_types
    assert percept["operation"] in valid_ops
    assert percept["is_question"] is True  # "What is 3 + 4?" ends with "?"
```

- [ ] **Step 2: Run final full suite**

```
PYTHONPATH=. pytest hpm_ai_v3/tests/ -v
```

Expected: all tests PASS. The `test_task_perceptor.py` file should have 24+ tests.

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v3/tests/test_task_perceptor.py
git commit -m "test: add standalone percept field validation smoke test"
```

---

## Self-Review

**Spec coverage:**
- TaskPerceptor class with input_type + operation — Tasks 1 + 2
- ToolCompatibilityFilter on ToolSelector — Task 3
- perceive_task() on InnateCognitiveSubstrate — Task 4
- Integration in act() — Task 5
- Arithmetic validator bare-number fix — Task 6
- Full test suite — Tasks 1, 2, 3, 6, 7

**Placeholder scan:** No TBDs, no "implement later", no vague instructions. Every step with code change shows the full code.

**Type consistency:**
- `TaskPerceptor.perceive()` returns dict with keys `input_type`, `operation`, `numeric_values`, `tokens`, `is_question` — same keys used in all test assertions and filter call sites.
- `ToolSelector.filter_by_percept(percept, weights, patterns)` signature matches usage in Task 5.
- `substrate.perceive_task(text)` method name matches Task 4 implementation and Task 5 call site.
- `TOOL_COMPAT` keys (`"numeric"`, `"expression"`, `"string"`, `"boolean"`, `"list"`, `"mixed"`) match exactly the six return values of `TaskPerceptor._detect_type()`.
