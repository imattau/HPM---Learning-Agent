# Innate Cognitive Substrate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace scattered argument-wiring logic in base_discovery.py with a permanent `InnateCognitiveSubstrate` that handles signature inspection, type coercion, and pattern extraction for every tool call.

**Architecture:** A singleton `InnateCognitiveSubstrate` class in `hpm_ai_v3/tools/innate_substrate.py` exposes `resolve_call(module, function, pool, task_text)` which returns validated `(module, function, resolved_args)`. `base_discovery.py` calls it in `act()` replacing `_generate_argument_pool`, `_resolve_binding`, and `trial_args` generation. The substrate's 15 innate functions are plain Python — never in the population.

**Tech Stack:** Python stdlib (`inspect`, `re`, `importlib`), existing ToolRegistry

---

### Task 1: Create innate_substrate.py with Group A (Introspection)

**Files:**
- Create: `hpm_ai_v3/tools/innate_substrate.py`
- Create: `hpm_ai_v3/tools/test_innate_substrate.py`

- [ ] **Step 1: Write failing tests for Group A**

```python
# hpm_ai_v3/tools/test_innate_substrate.py
import pytest
from hpm_ai_v3.tools.innate_substrate import InnateCognitiveSubstrate

@pytest.fixture
def substrate():
    return InnateCognitiveSubstrate()

def test_inspect_signature_math_sqrt(substrate):
    sig = substrate.inspect_signature("math", "sqrt")
    assert sig is not None
    assert any(p["name"] == "x" for p in sig["params"])

def test_inspect_signature_re_findall(substrate):
    sig = substrate.inspect_signature("re", "findall")
    assert sig is not None
    assert len(sig["params"]) >= 2

def test_get_type_int(substrate):
    assert substrate.get_type(42) == "int"

def test_get_type_float(substrate):
    assert substrate.get_type(3.14) == "float"

def test_get_type_str(substrate):
    assert substrate.get_type("hello") == "str"

def test_get_type_list(substrate):
    assert substrate.get_type([1, 2]) == "list"

def test_describe_value_numeric_str(substrate):
    d = substrate.describe_value("42")
    assert d["type"] == "str"
    assert d["numeric_value"] == 42.0

def test_describe_value_list(substrate):
    d = substrate.describe_value([1, 2, 3])
    assert d["length"] == 3

def test_list_callable_functions(substrate):
    fns = substrate.list_callable_functions("math")
    names = [f["name"] for f in fns]
    assert "sqrt" in names
    assert "factorial" in names
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python3 -m pytest hpm_ai_v3/tools/test_innate_substrate.py -v 2>&1 | tail -5
```
Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Create innate_substrate.py with Group A**

```python
# hpm_ai_v3/tools/innate_substrate.py
"""
InnateCognitiveSubstrate - Permanent cognitive infrastructure for HPM agents.
Handles signature inspection, type coercion, and pattern extraction.
Never in the population. Never subject to replicator dynamics.
"""
import re
import math
import inspect
import importlib
from typing import Any, Dict, List, Optional, Tuple


class InnateCognitiveSubstrate:
    """Permanent singleton. Wraps every tool call with correct argument resolution."""

    # ── Group A: Introspection ──────────────────────────────────────────────

    def inspect_signature(self, module: str, function: str) -> Optional[Dict]:
        """Return {params: [{name, type, default}]} for module.function."""
        try:
            mod = importlib.import_module(module)
            # Handle dotted names like str.upper
            if "." in function:
                parts = function.split(".", 1)
                type_obj = getattr(mod, parts[0], None) or getattr(__builtins__ if isinstance(__builtins__, dict) else __import__("builtins"), parts[0], None)
                fn = getattr(type_obj, parts[1], None) if type_obj else None
            else:
                fn = getattr(mod, function, None)
            if fn is None:
                return None
            sig = inspect.signature(fn)
            params = []
            for name, param in sig.parameters.items():
                if name in ("self", "cls"):
                    continue
                ann = param.annotation
                type_hint = ann.__name__ if ann != inspect.Parameter.empty and hasattr(ann, "__name__") else None
                default = None if param.default is inspect.Parameter.empty else param.default
                params.append({"name": name, "type": type_hint, "default": default})
            return {"params": params}
        except Exception:
            return None

    def get_type(self, value: Any) -> str:
        """Return string type name of value."""
        if isinstance(value, bool): return "bool"
        if isinstance(value, int): return "int"
        if isinstance(value, float): return "float"
        if isinstance(value, str): return "str"
        if isinstance(value, list): return "list"
        if isinstance(value, dict): return "dict"
        return type(value).__name__

    def describe_value(self, value: Any) -> Dict:
        """Return {type, length, preview, numeric_value}."""
        result = {"type": self.get_type(value), "preview": str(value)[:50]}
        if isinstance(value, (list, str, dict)):
            result["length"] = len(value)
        numeric = self.to_float(value)
        result["numeric_value"] = numeric
        return result

    def list_callable_functions(self, module: str) -> List[Dict]:
        """Return [{name, signature}] for all callable public functions in module."""
        try:
            mod = importlib.import_module(module)
            result = []
            for name in dir(mod):
                if name.startswith("_"):
                    continue
                obj = getattr(mod, name)
                if callable(obj):
                    try:
                        sig = str(inspect.signature(obj))
                    except (ValueError, TypeError):
                        sig = "(...)"
                    result.append({"name": name, "signature": f"{name}{sig}"})
            return result
        except Exception:
            return []
```

- [ ] **Step 4: Run Group A tests**

```bash
python3 -m pytest hpm_ai_v3/tools/test_innate_substrate.py -v 2>&1 | tail -15
```
Expected: all 9 Group A tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v3/tools/innate_substrate.py hpm_ai_v3/tools/test_innate_substrate.py
git commit -m "feat: add InnateCognitiveSubstrate Group A (introspection)"
```

---

### Task 2: Add Group B (Type Operations) to innate_substrate.py

**Files:**
- Modify: `hpm_ai_v3/tools/innate_substrate.py`
- Modify: `hpm_ai_v3/tools/test_innate_substrate.py`

- [ ] **Step 1: Add Group B tests**

Append to `hpm_ai_v3/tools/test_innate_substrate.py`:

```python
# Group B tests
def test_coerce_str_to_int(substrate):
    assert substrate.coerce("42", "int") == 42

def test_coerce_str_to_float(substrate):
    assert substrate.coerce("3.14", "float") == 3.14

def test_coerce_returns_none_on_failure(substrate):
    assert substrate.coerce("hello", "int") is None

def test_to_int_from_float(substrate):
    assert substrate.to_int(3.9) == 3

def test_to_float_from_str(substrate):
    assert substrate.to_float("2.5") == 2.5

def test_to_str(substrate):
    assert substrate.to_str(42) == "42"

def test_to_list_from_str(substrate):
    result = substrate.to_list("hello")
    assert isinstance(result, list)

def test_safe_call_success(substrate):
    result = substrate.safe_call("math", "sqrt", 144)
    assert result == 12.0

def test_safe_call_error_returns_dict(substrate):
    result = substrate.safe_call("math", "sqrt", "not_a_number")
    assert isinstance(result, dict)
    assert "error" in result
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m pytest hpm_ai_v3/tools/test_innate_substrate.py -k "group_b or coerce or to_int or to_float or to_str or to_list or safe_call" -v 2>&1 | tail -5
```
Expected: `AttributeError` — methods not yet defined.

- [ ] **Step 3: Add Group B methods to InnateCognitiveSubstrate**

Append inside the `InnateCognitiveSubstrate` class in `innate_substrate.py`:

```python
    # ── Group B: Type Operations ────────────────────────────────────────────

    def coerce(self, value: Any, target_type: Optional[str]) -> Optional[Any]:
        """Convert value to target_type. Returns None on failure."""
        if target_type is None:
            return value
        try:
            if target_type == "int":
                return int(float(str(value).strip()))
            if target_type == "float":
                return float(str(value).strip())
            if target_type == "str":
                return str(value)
            if target_type == "list":
                return self.to_list(value)
            if target_type == "bool":
                return bool(value)
        except Exception:
            return None
        return value

    def to_int(self, value: Any) -> Optional[int]:
        return self.coerce(value, "int")

    def to_float(self, value: Any) -> Optional[float]:
        try:
            return float(str(value).strip())
        except Exception:
            return None

    def to_str(self, value: Any) -> str:
        return str(value)

    def to_list(self, value: Any) -> list:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return list(value)
        if isinstance(value, (tuple, set)):
            return list(value)
        return [value]

    def safe_call(self, module: str, function: str, *args) -> Any:
        """Call module.function(*args). Returns error dict instead of raising."""
        try:
            mod = importlib.import_module(module)
            fn = getattr(mod, function)
            return fn(*args)
        except Exception as e:
            return {"error": str(e), "status": "failed"}
```

- [ ] **Step 4: Run all tests**

```bash
python3 -m pytest hpm_ai_v3/tools/test_innate_substrate.py -v 2>&1 | tail -20
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v3/tools/innate_substrate.py hpm_ai_v3/tools/test_innate_substrate.py
git commit -m "feat: add InnateCognitiveSubstrate Group B (type operations)"
```

---

### Task 3: Add Group C (Pattern Matching) and resolve_call

**Files:**
- Modify: `hpm_ai_v3/tools/innate_substrate.py`
- Modify: `hpm_ai_v3/tools/test_innate_substrate.py`

- [ ] **Step 1: Add Group C + resolve_call tests**

Append to `hpm_ai_v3/tools/test_innate_substrate.py`:

```python
# Group C tests
def test_extract_numbers_from_text(substrate):
    assert substrate.extract_numbers("The price is 42 and 7") == [42.0, 7.0]

def test_extract_numbers_empty(substrate):
    assert substrate.extract_numbers("no numbers here") == []

def test_match_regex(substrate):
    assert substrate.match_regex(r"\d+", "abc 42 def 7") == ["42", "7"]

def test_find_in_list(substrate):
    assert substrate.find_in_list(42, [1, 42, 3]) == 1

def test_find_in_list_missing(substrate):
    assert substrate.find_in_list(99, [1, 2, 3]) is None

def test_split_text_default(substrate):
    assert substrate.split_text("hello world") == ["hello", "world"]

def test_split_text_delimiter(substrate):
    assert substrate.split_text("a,b,c", ",") == ["a", "b", "c"]

def test_compare_values_greater(substrate):
    assert substrate.compare_values(10, 5) == "greater"

def test_compare_values_equal(substrate):
    assert substrate.compare_values(5, 5) == "equal"

def test_compare_values_incomparable(substrate):
    assert substrate.compare_values("hello", 5) == "incomparable"

# resolve_call tests
def test_resolve_call_math_sqrt(substrate):
    mod, fn, args = substrate.resolve_call("math", "sqrt", [144], "")
    assert mod == "math"
    assert fn == "sqrt"
    assert args == [144.0] or args == [144]

def test_resolve_call_extracts_from_text(substrate):
    mod, fn, args = substrate.resolve_call("math", "sqrt", [], "compute sqrt of 25")
    assert args[0] == 25.0

def test_resolve_call_re_findall(substrate):
    mod, fn, args = substrate.resolve_call("re", "findall", [], "The numbers are 42 and 7")
    assert mod == "re"
    assert fn == "findall"
    # first arg should be a pattern string, second the text
    assert len(args) == 2
```

- [ ] **Step 2: Run to verify they fail**

```bash
python3 -m pytest hpm_ai_v3/tools/test_innate_substrate.py -k "extract or match_regex or find_in or split_text or compare or resolve" -v 2>&1 | tail -5
```
Expected: `AttributeError`.

- [ ] **Step 3: Add Group C and resolve_call to InnateCognitiveSubstrate**

Append inside the class in `innate_substrate.py`:

```python
    # ── Group C: Pattern Matching / Perception ──────────────────────────────

    def extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text as floats."""
        matches = re.findall(r"-?\d+\.?\d*", str(text))
        result = []
        for m in matches:
            try:
                result.append(float(m))
            except ValueError:
                pass
        return result

    def match_regex(self, pattern: str, text: str) -> List[str]:
        """Return all regex matches in text."""
        try:
            return re.findall(pattern, str(text))
        except Exception:
            return []

    def find_in_list(self, value: Any, lst: list) -> Optional[int]:
        """Return index of value in lst, or None."""
        try:
            return lst.index(value)
        except (ValueError, TypeError):
            return None

    def split_text(self, text: str, delimiter: Optional[str] = None) -> List[str]:
        """Split text by delimiter (default: whitespace)."""
        if delimiter:
            return str(text).split(delimiter)
        return str(text).split()

    def compare_values(self, a: Any, b: Any) -> str:
        """Compare two values. Returns 'equal', 'greater', 'less', 'incomparable'."""
        try:
            fa, fb = float(a), float(b)
            if fa == fb: return "equal"
            return "greater" if fa > fb else "less"
        except (TypeError, ValueError):
            if a == b: return "equal"
            return "incomparable"

    # ── Core: resolve_call ──────────────────────────────────────────────────

    def resolve_call(
        self,
        module: str,
        function: str,
        pool: List[Any],
        task_text: str
    ) -> Tuple[str, str, List[Any]]:
        """
        Resolve arguments for module.function from pool and task_text.
        Returns (module, function, resolved_args_list).
        """
        sig = self.inspect_signature(module, function)
        if not sig or not sig["params"]:
            # No signature info — pass task_text as sole arg
            nums = self.extract_numbers(task_text)
            return (module, function, nums[:1] if nums else [task_text])

        resolved = []
        for param in sig["params"]:
            val = self._match_param(param, pool, task_text, function)
            resolved.append(val)

        return (module, function, resolved)

    def _match_param(
        self,
        param: Dict,
        pool: List[Any],
        task_text: str,
        function_name: str = ""
    ) -> Any:
        """Find best value from pool for a parameter."""
        target_type = param.get("type")
        param_name = param.get("name", "")

        # Special case: regex pattern parameter
        if param_name == "pattern" or "pattern" in param_name.lower():
            return r"\d+"

        # Special case: string/text parameter
        if target_type == "str" or param_name in ("string", "text", "s", "seq"):
            # Prefer task_text for string params
            for val in pool:
                if isinstance(val, str):
                    return val
            return task_text

        # Try pool values, coerce to target type
        for val in pool:
            coerced = self.coerce(val, target_type)
            if coerced is not None:
                return coerced

        # Fallback: extract numbers from task_text
        nums = self.extract_numbers(task_text)
        if nums:
            coerced = self.coerce(nums[0], target_type)
            if coerced is not None:
                return coerced

        # Last resort: task_text
        return task_text
```

- [ ] **Step 4: Run all tests**

```bash
python3 -m pytest hpm_ai_v3/tools/test_innate_substrate.py -v 2>&1 | tail -25
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm_ai_v3/tools/innate_substrate.py hpm_ai_v3/tools/test_innate_substrate.py
git commit -m "feat: add InnateCognitiveSubstrate Group C (pattern matching) and resolve_call"
```

---

### Task 4: Wire substrate into base_discovery.py

**Files:**
- Modify: `hpm_ai_v3/agents/base_discovery.py`

Remove: `_generate_argument_pool`, `_resolve_binding`, `trial_args` generation loop, `is_bound`/`arg_bindings` trial sampling.
Add: substrate instantiation in `__init__`, single `substrate.resolve_call()` call in `act()`.

- [ ] **Step 1: Write integration test**

```python
# hpm_ai_v3/tools/test_innate_substrate.py (append)
def test_substrate_resolves_math_sqrt_from_text():
    """End-to-end: substrate resolves math.sqrt args from task text."""
    from hpm_ai_v3.tools.innate_substrate import InnateCognitiveSubstrate
    s = InnateCognitiveSubstrate()
    mod, fn, args = s.resolve_call("math", "sqrt", [], "compute sqrt of 144")
    assert mod == "math"
    result = s.safe_call(mod, fn, *args)
    assert result == 12.0
```

- [ ] **Step 2: Run to verify it passes (substrate already complete)**

```bash
python3 -m pytest hpm_ai_v3/tools/test_innate_substrate.py::test_substrate_resolves_math_sqrt_from_text -v
```
Expected: PASS.

- [ ] **Step 3: Add substrate to PureAgnosticDiscoveryAgent.__init__**

In `hpm_ai_v3/agents/base_discovery.py`, add import at top:
```python
from hpm_ai_v3.tools.innate_substrate import InnateCognitiveSubstrate
```

In `PureAgnosticDiscoveryAgent.__init__`, after `self.pipeline_recomb = ...`:
```python
        self.substrate = InnateCognitiveSubstrate()
```

- [ ] **Step 4: Replace act() argument wiring**

In `act()`, replace the block from `# 2. TRIAL ARGUMENT SAMPLING` through the `trial_args` construction (currently lines ~222-234) with:

```python
        # 2. SUBSTRATE ARGUMENT RESOLUTION
        pool = self._get_pool()
        task_text = self.current_task.get("text", "") if self.current_task else ""

        if isinstance(action_pattern, ActionPattern) and action_pattern.module and action_pattern.function:
            _, _, resolved_args = self.substrate.resolve_call(
                action_pattern.module,
                action_pattern.function,
                pool,
                task_text
            )
        else:
            resolved_args = [task_text] if task_text else []
```

- [ ] **Step 5: Add _get_pool() replacing _generate_argument_pool()**

Add this method to `PureAgnosticDiscoveryAgent` (replaces the old `_generate_argument_pool`):

```python
    def _get_pool(self) -> List[Any]:
        """Collect candidate values from episodic memory and task context."""
        pool = []
        mem_res = ToolRegistry.call("episodic_get_recent", n=10)
        for event in mem_res.get("events", []):
            res = event.get("result")
            if res is not None and isinstance(res, (int, float, str, list, dict)):
                pool.append(res)
        if self.current_task:
            if "inputs" in self.current_task:
                pool.extend(self.current_task["inputs"])
            text = self.current_task.get("text", "")
            if text:
                pool.append(text)
        return pool
```

- [ ] **Step 6: Update action_pattern.sample() call to use resolved_args**

In `act()`, replace the `context` dict and `action_pattern.sample(context)` call:

```python
        context = {
            "pool": pool,
            "text": task_text,
            "resolved_args": resolved_args,
            "context_features": self.extract_features()
        }
        result = action_pattern.sample(context)
```

In `ActionPattern.sample()`, replace the args-building loop with:

```python
    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, Any]:
        resolved_args = context.get("resolved_args", [])
        task_text = context.get("text", "")

        try:
            if self.action_type == "python_call":
                result = ToolRegistry.call("python_call",
                                           module=self.module,
                                           function=self.function,
                                           args=resolved_args)
            else:
                result = ToolRegistry.call(self.action_type)
            return result
        except Exception as e:
            return {"error": str(e), "status": "failed"}
```

- [ ] **Step 7: Remove dead code**

Delete from `base_discovery.py`:
- `_generate_argument_pool()` method (replaced by `_get_pool()`)
- `_resolve_binding()` method (moved into substrate)
- `is_bound`, `arg_bindings`, `trial_args` references in `act()`
- `pattern_failure_counts`, `failure_counts` short-term inhibition (now unnecessary with correct wiring)
- `episodic_pattern_usage` curiosity rotation (causes correct patterns to be penalised — identified as HPM drift in review)

- [ ] **Step 8: Smoke test**

```bash
python3 -c "
from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
from hpm_ai_v3.curriculum import CurriculumManager
agent = UnifiedDiscoveryAgent(context_dim=64)
cm = CurriculumManager()
task = cm.get_current_task()
print('Task:', task['text'])
sol = agent.run_episode(task, max_steps=5)
print('Solution:', sol)
" 2>&1 | grep -E 'Task:|Solution:|Error|Traceback'
```
Expected: no Traceback, prints Task and Solution.

- [ ] **Step 9: Commit**

```bash
git add hpm_ai_v3/agents/base_discovery.py
git commit -m "refactor: replace scattered argument wiring with InnateCognitiveSubstrate in base_discovery"
```

---

### Task 5: Clean up discovery_agent.py

**Files:**
- Modify: `hpm_ai_v3/agents/discovery_agent.py`

- [ ] **Step 1: Remove redundant act() override argument handling**

In `UnifiedDiscoveryAgent.act()` (currently just injects environment tool), verify it only does:
```python
    def act(self, step_idx: int = 0) -> Dict[str, Any]:
        if self.hidden_fn and not ToolRegistry.get_tool_info("evaluate_at"):
            ToolRegistry.register("evaluate_at", self._env_evaluate, ["x"], "result", 0.1)
        return super().act(step_idx)
```
Remove any trial_args or pool manipulation if present.

- [ ] **Step 2: Run smoke test with 20 episodes**

```bash
python3 -c "
from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
from hpm_ai_v3.curriculum import CurriculumManager
import numpy as np
agent = UnifiedDiscoveryAgent(context_dim=64)
cm = CurriculumManager()
rewards = []
for ep in range(20):
    task = cm.get_current_task()
    sol = agent.run_episode(task, max_steps=10)
    r = agent.evaluate_solution(sol)
    cm.update(r)
    rewards.append(r)
print(f'Avg reward: {np.mean(rewards):.3f}')
print(f'Phase: {cm.phase:.1f}')
" 2>&1 | grep -E 'Avg|Phase|Error|Traceback'
```
Expected: no Traceback, avg reward > 0.

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v3/agents/discovery_agent.py
git commit -m "cleanup: remove redundant argument handling from UnifiedDiscoveryAgent"
```

---

### Task 6: Full curriculum progression test

**Files:**
- Modify: `hpm_ai_v3/task8/test_curriculum_progression.py`

- [ ] **Step 1: Add substrate integration test**

Append to `hpm_ai_v3/task8/test_curriculum_progression.py`:

```python
def test_substrate_enables_curriculum_progression():
    """Agent with substrate should make 3+ phase transitions in 150 episodes."""
    from hpm_ai_v3.agents.discovery_agent import UnifiedDiscoveryAgent
    from hpm_ai_v3.curriculum import CurriculumManager
    agent = UnifiedDiscoveryAgent(context_dim=64)
    cm = CurriculumManager()
    transitions = 0
    for ep in range(150):
        task = cm.get_current_task()
        prev = cm.phase
        sol = agent.run_episode(task, max_steps=15)
        r = agent.evaluate_solution(sol)
        cm.update(r)
        if cm.phase != prev:
            transitions += 1
    assert transitions >= 3, f"Expected 3+ transitions, got {transitions}"
```

- [ ] **Step 2: Run test**

```bash
python3 -m pytest hpm_ai_v3/task8/test_curriculum_progression.py::test_substrate_enables_curriculum_progression -v -s 2>&1 | tail -10
```
Expected: PASS with transitions >= 3.

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v3/task8/test_curriculum_progression.py
git commit -m "test: verify InnateCognitiveSubstrate enables curriculum progression"
```
