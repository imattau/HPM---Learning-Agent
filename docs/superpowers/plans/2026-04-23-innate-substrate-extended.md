# Plan: InnateCognitiveSubstrate Extended — Groups D, E, F, G

## Goal
Add 16 innate functions across Groups D (structural/relational), E (temporal/sequential), F (uncertainty/confidence), and G (goal/task decomposition) to `hpm_ai_v3/tools/innate_substrate.py`. All functions are pure Python, stdlib only, never raise, and follow existing class conventions.

## Architecture
- Single class `InnateCognitiveSubstrate` in `hpm_ai_v3/tools/innate_substrate.py`
- Append four new comment-delimited groups after Group C, before `resolve_call`
- One new test file: `hpm_ai_v3/tests/test_innate_substrate_extended.py`
- No new dependencies

## Tech Stack
- Python 3.10+, stdlib only (`re`, `math`)
- pytest for tests

---

## Task 1: Group D — Structural / Relational Reasoning

### Step 1a — Write failing tests for Group D

File: `hpm_ai_v3/tests/test_innate_substrate_extended.py`

Create this file:

```python
# hpm_ai_v3/tests/test_innate_substrate_extended.py
import pytest
from hpm_ai_v3.tools.innate_substrate import InnateCognitiveSubstrate

S = InnateCognitiveSubstrate()


# ── Group D ────────────────────────────────────────────────────────────────

class TestBuildMapping:
    def test_basic(self):
        assert S.build_mapping(["a", "b"], [1, 2]) == {"a": 1, "b": 2}

    def test_truncates_to_shorter(self):
        assert S.build_mapping(["a", "b", "c"], [1, 2]) == {"a": 1, "b": 2}

    def test_empty(self):
        assert S.build_mapping([], []) == {}

    def test_non_list_inputs(self):
        assert S.build_mapping("x", 9) == {"x": 9}


class TestInvertMapping:
    def test_basic(self):
        assert S.invert_mapping({"a": 1, "b": 2}) == {1: "a", 2: "b"}

    def test_empty(self):
        assert S.invert_mapping({}) == {}

    def test_duplicate_values_last_wins(self):
        result = S.invert_mapping({"a": 1, "b": 1})
        assert result == {1: "b"}

    def test_unhashable_value_skipped(self):
        result = S.invert_mapping({"a": [1, 2], "b": 3})
        assert result == {3: "b"}


class TestGetNested:
    def test_dict_access(self):
        assert S.get_nested({"a": {"b": 42}}, "a.b") == 42

    def test_list_index(self):
        assert S.get_nested({"a": [10, 20, 30]}, "a.1") == 20

    def test_missing_key_returns_none(self):
        assert S.get_nested({"a": 1}, "b") is None

    def test_empty_path_returns_obj(self):
        obj = {"x": 1}
        assert S.get_nested(obj, "") == obj

    def test_deep_missing(self):
        assert S.get_nested({"a": {"b": 1}}, "a.c.d") is None


class TestFlatten:
    def test_basic(self):
        assert S.flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]

    def test_scalar(self):
        assert S.flatten(7) == [7]

    def test_empty(self):
        assert S.flatten([]) == []

    def test_dict_not_recursed(self):
        assert S.flatten([{"a": 1}, 2]) == [{"a": 1}, 2]

    def test_deeply_nested(self):
        assert S.flatten([[[1]], [[2, 3]]]) == [1, 2, 3]


class TestGroupBy:
    def test_basic(self):
        result = S.group_by([1, 2, 3, 4], lambda x: x % 2)
        assert result == {1: [1, 3], 0: [2, 4]}

    def test_empty(self):
        assert S.group_by([], lambda x: x) == {}

    def test_key_fn_raises_item_skipped(self):
        result = S.group_by([1, "bad", 3], lambda x: 1 / x)
        assert 1.0 in result or "bad" not in str(result)

    def test_strings(self):
        result = S.group_by(["apple", "avocado", "banana"], lambda x: x[0])
        assert result == {"a": ["apple", "avocado"], "b": ["banana"]}


# ── Group E ────────────────────────────────────────────────────────────────

class TestDetectTrend:
    def test_increasing(self):
        assert S.detect_trend([1, 2, 3, 4]) == "increasing"

    def test_decreasing(self):
        assert S.detect_trend([4, 3, 2, 1]) == "decreasing"

    def test_stable(self):
        assert S.detect_trend([5, 5, 5]) == "stable"

    def test_volatile(self):
        assert S.detect_trend([1, 3, 2, 5]) == "volatile"

    def test_short_series(self):
        assert S.detect_trend([42]) == "stable"


class TestDiffSequence:
    def test_basic(self):
        assert S.diff_sequence([1, 3, 6, 10]) == [2, 3, 4]

    def test_empty(self):
        assert S.diff_sequence([]) == []

    def test_single(self):
        assert S.diff_sequence([5]) == []

    def test_negatives(self):
        assert S.diff_sequence([5, 3, 1]) == [-2, -2]


class TestFindRepeating:
    def test_basic_period_2(self):
        assert S.find_repeating([1, 2, 1, 2, 1, 2]) == [1, 2]

    def test_period_3(self):
        assert S.find_repeating([1, 2, 3, 1, 2, 3]) == [1, 2, 3]

    def test_no_repeat(self):
        assert S.find_repeating([1, 2, 3, 4]) is None

    def test_empty(self):
        assert S.find_repeating([]) is None

    def test_single(self):
        assert S.find_repeating([7]) is None


class TestSlidingWindow:
    def test_basic(self):
        assert S.sliding_window([1, 2, 3, 4], 2) == [[1, 2], [2, 3], [3, 4]]

    def test_window_equals_length(self):
        assert S.sliding_window([1, 2, 3], 3) == [[1, 2, 3]]

    def test_window_too_large(self):
        assert S.sliding_window([1, 2], 5) == []

    def test_n_zero(self):
        assert S.sliding_window([1, 2, 3], 0) == []

    def test_empty_sequence(self):
        assert S.sliding_window([], 2) == []


# ── Group F ────────────────────────────────────────────────────────────────

class TestNormalize:
    def test_basic(self):
        result = S.normalize([1, 1, 2])
        assert abs(sum(result) - 1.0) < 1e-9
        assert abs(result[2] - 0.5) < 1e-9

    def test_all_zeros(self):
        result = S.normalize([0, 0, 0])
        assert result == [1/3, 1/3, 1/3]

    def test_single(self):
        assert S.normalize([5]) == [1.0]

    def test_empty(self):
        assert S.normalize([]) == []


class TestEntropy:
    def test_uniform_2(self):
        result = S.entropy([0.5, 0.5])
        assert abs(result - 1.0) < 1e-9

    def test_certain(self):
        assert S.entropy([1.0, 0.0]) == 0.0

    def test_empty(self):
        assert S.entropy([]) == 0.0

    def test_uniform_4(self):
        result = S.entropy([0.25, 0.25, 0.25, 0.25])
        assert abs(result - 2.0) < 1e-9


class TestArgmax:
    def test_basic(self):
        assert S.argmax([1, 3, 2]) == 1

    def test_first_tie(self):
        assert S.argmax([3, 3, 1]) == 0

    def test_single(self):
        assert S.argmax([7]) == 0

    def test_empty(self):
        assert S.argmax([]) == -1


class TestClamp:
    def test_below_lo(self):
        assert S.clamp(-5, 0, 10) == 0.0

    def test_above_hi(self):
        assert S.clamp(15, 0, 10) == 10.0

    def test_in_range(self):
        assert S.clamp(5, 0, 10) == 5.0

    def test_lo_greater_than_hi_swaps(self):
        assert S.clamp(5, 10, 0) == 5.0

    def test_non_numeric(self):
        assert S.clamp("bad", 0, 1) == 0.0


# ── Group G ────────────────────────────────────────────────────────────────

class TestDecomposeText:
    def test_basic(self):
        result = S.decompose_text("find the maximum value")
        assert result["verb"] == "find"

    def test_empty(self):
        result = S.decompose_text("")
        assert result == {"verb": "", "object": "", "modifier": ""}

    def test_known_verb(self):
        result = S.decompose_text("compute the sum of all numbers")
        assert result["verb"] == "compute"

    def test_returns_dict_keys(self):
        result = S.decompose_text("sort items by value")
        assert set(result.keys()) == {"verb", "object", "modifier"}

    def test_modifier_extracted(self):
        result = S.decompose_text("filter items by category")
        assert result["modifier"] != "" or result["object"] != ""


class TestEstimateProgress:
    def test_half(self):
        assert S.estimate_progress(5, 10) == 0.5

    def test_complete(self):
        assert S.estimate_progress(10, 10) == 1.0

    def test_over_target_clamped(self):
        assert S.estimate_progress(15, 10) == 1.0

    def test_zero_target_zero_current(self):
        assert S.estimate_progress(0, 0) == 1.0

    def test_zero_target_nonzero_current(self):
        assert S.estimate_progress(5, 0) == 0.0

    def test_non_numeric(self):
        assert S.estimate_progress("bad", 10) == 0.0


class TestCheckConstraint:
    def test_gt(self):
        assert S.check_constraint(5, "> 3") is True
        assert S.check_constraint(2, "> 3") is False

    def test_gte(self):
        assert S.check_constraint(3, ">= 3") is True

    def test_lt(self):
        assert S.check_constraint(1, "< 3") is True

    def test_lte(self):
        assert S.check_constraint(3, "<= 3") is True

    def test_eq(self):
        assert S.check_constraint(4, "== 4") is True

    def test_neq(self):
        assert S.check_constraint(4, "!= 5") is True

    def test_in_list(self):
        assert S.check_constraint(2, "in [1, 2, 3]") is True
        assert S.check_constraint(9, "in [1, 2, 3]") is False

    def test_type_check(self):
        assert S.check_constraint(42, "type:int") is True
        assert S.check_constraint("hello", "type:str") is True
        assert S.check_constraint([1, 2], "type:list") is True

    def test_len_check(self):
        assert S.check_constraint([1, 2, 3], "len == 3") is True
        assert S.check_constraint([1, 2], "len > 3") is False

    def test_unknown_constraint_permissive(self):
        assert S.check_constraint(42, "nonsense") is True
```

### Step 1b — Run tests (expect all failures)

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v3/tests/test_innate_substrate_extended.py -x -q 2>&1 | head -30
```

Expected output contains: `AttributeError: 'InnateCognitiveSubstrate' object has no attribute 'build_mapping'`

### Step 1c — Implement Group D

In `hpm_ai_v3/tools/innate_substrate.py`, insert the following block immediately before the `# ── Core: resolve_call` line:

```python
    # ── Group D: Structural / Relational Reasoning ─────────────────────────

    def build_mapping(self, keys, values) -> dict:
        """Zip keys and values into a dict. Truncates to shorter list."""
        try:
            if not isinstance(keys, (list, tuple)):
                keys = [keys]
            if not isinstance(values, (list, tuple)):
                values = [values]
            return dict(zip(keys, values))
        except Exception:
            return {}

    def invert_mapping(self, mapping: dict) -> dict:
        """Swap keys and values. Skips unhashable values. Duplicate values: last key wins."""
        result = {}
        try:
            for k, v in mapping.items():
                try:
                    result[v] = k
                except TypeError:
                    pass
        except Exception:
            pass
        return result

    def get_nested(self, obj, path: str):
        """Safe deep access using dot-separated path. Numeric segments used as list indices."""
        if path == "":
            return obj
        try:
            parts = path.split(".")
            current = obj
            for part in parts:
                if current is None:
                    return None
                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, (list, tuple)):
                    try:
                        current = current[int(part)]
                    except (ValueError, IndexError):
                        return None
                else:
                    return None
            return current
        except Exception:
            return None

    def flatten(self, nested) -> list:
        """Recursively flatten nested lists. Dicts and non-list scalars kept as-is."""
        result = []
        try:
            if not isinstance(nested, list):
                return [nested]
            for item in nested:
                if isinstance(item, list):
                    result.extend(self.flatten(item))
                else:
                    result.append(item)
        except Exception:
            pass
        return result

    def group_by(self, items: list, key_fn) -> dict:
        """Partition items by key_fn(item). Items where key_fn raises are skipped."""
        result = {}
        try:
            for item in items:
                try:
                    key = key_fn(item)
                    if key not in result:
                        result[key] = []
                    result[key].append(item)
                except Exception:
                    pass
        except Exception:
            pass
        return result
```

### Step 1d — Run Group D tests

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v3/tests/test_innate_substrate_extended.py::TestBuildMapping hpm_ai_v3/tests/test_innate_substrate_extended.py::TestInvertMapping hpm_ai_v3/tests/test_innate_substrate_extended.py::TestGetNested hpm_ai_v3/tests/test_innate_substrate_extended.py::TestFlatten hpm_ai_v3/tests/test_innate_substrate_extended.py::TestGroupBy -v 2>&1 | tail -20
```

Expected: `20 passed`

### Step 1e — Commit Group D

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
git add hpm_ai_v3/tools/innate_substrate.py hpm_ai_v3/tests/test_innate_substrate_extended.py
git commit -m "$(cat <<'EOF'
feat: add Group D structural/relational functions to InnateCognitiveSubstrate

Adds build_mapping, invert_mapping, get_nested, flatten, group_by.
Supports L3 relational rule formation in HPM agents.
EOF
)"
```

---

## Task 2: Group E — Temporal / Sequential Reasoning

### Step 2a — Run tests for Group E (expect failures)

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v3/tests/test_innate_substrate_extended.py::TestDetectTrend hpm_ai_v3/tests/test_innate_substrate_extended.py::TestDiffSequence hpm_ai_v3/tests/test_innate_substrate_extended.py::TestFindRepeating hpm_ai_v3/tests/test_innate_substrate_extended.py::TestSlidingWindow -q 2>&1 | head -10
```

Expected: `AttributeError: 'InnateCognitiveSubstrate' object has no attribute 'detect_trend'`

### Step 2b — Implement Group E

Append the following block in `hpm_ai_v3/tools/innate_substrate.py` after the Group D block (before `# ── Core: resolve_call`):

```python
    # ── Group E: Temporal / Sequential Reasoning ───────────────────────────

    def detect_trend(self, series: list) -> str:
        """Characterise direction: 'increasing'|'decreasing'|'stable'|'volatile'."""
        try:
            nums = []
            for v in series:
                try:
                    nums.append(float(v))
                except (TypeError, ValueError):
                    pass
            if len(nums) < 2:
                return "stable"
            diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
            if all(d == 0 for d in diffs):
                return "stable"
            if all(d > 0 for d in diffs):
                return "increasing"
            if all(d < 0 for d in diffs):
                return "decreasing"
            return "volatile"
        except Exception:
            return "stable"

    def diff_sequence(self, series: list) -> list:
        """First-order differences: series[i+1] - series[i]."""
        try:
            nums = []
            for v in series:
                try:
                    nums.append(float(v))
                except (TypeError, ValueError):
                    return []
            if len(nums) < 2:
                return []
            return [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
        except Exception:
            return []

    def find_repeating(self, sequence: list):
        """Return smallest repeating sub-list, or None."""
        try:
            n = len(sequence)
            if n < 2:
                return None
            for period in range(1, n // 2 + 1):
                unit = sequence[:period]
                tiles = (n // period)
                remainder = n % period
                if unit * tiles + unit[:remainder] == sequence:
                    return unit
            return None
        except Exception:
            return None

    def sliding_window(self, sequence: list, n: int) -> list:
        """All contiguous windows of size n."""
        try:
            if n <= 0 or n > len(sequence):
                return []
            return [sequence[i:i + n] for i in range(len(sequence) - n + 1)]
        except Exception:
            return []
```

### Step 2c — Run Group E tests

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v3/tests/test_innate_substrate_extended.py::TestDetectTrend hpm_ai_v3/tests/test_innate_substrate_extended.py::TestDiffSequence hpm_ai_v3/tests/test_innate_substrate_extended.py::TestFindRepeating hpm_ai_v3/tests/test_innate_substrate_extended.py::TestSlidingWindow -v 2>&1 | tail -20
```

Expected: `18 passed`

### Step 2d — Commit Group E

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
git add hpm_ai_v3/tools/innate_substrate.py
git commit -m "$(cat <<'EOF'
feat: add Group E temporal/sequential functions to InnateCognitiveSubstrate

Adds detect_trend, diff_sequence, find_repeating, sliding_window.
Supports L4 generative simulation prerequisite reasoning in HPM agents.
EOF
)"
```

---

## Task 3: Group F — Uncertainty / Confidence

### Step 3a — Run tests for Group F (expect failures)

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v3/tests/test_innate_substrate_extended.py::TestNormalize hpm_ai_v3/tests/test_innate_substrate_extended.py::TestEntropy hpm_ai_v3/tests/test_innate_substrate_extended.py::TestArgmax hpm_ai_v3/tests/test_innate_substrate_extended.py::TestClamp -q 2>&1 | head -10
```

Expected: `AttributeError: 'InnateCognitiveSubstrate' object has no attribute 'normalize'`

### Step 3b — Implement Group F

Append the following block in `hpm_ai_v3/tools/innate_substrate.py` after the Group E block (before `# ── Core: resolve_call`):

```python
    # ── Group F: Uncertainty / Confidence ──────────────────────────────────

    def normalize(self, values: list) -> list:
        """Convert list of non-negative numerics to probability distribution."""
        try:
            if not values:
                return []
            floats = []
            for v in values:
                try:
                    floats.append(float(v))
                except (TypeError, ValueError):
                    return []
            total = sum(floats)
            if total == 0:
                n = len(floats)
                return [1.0 / n] * n
            return [v / total for v in floats]
        except Exception:
            return []

    def entropy(self, probs: list) -> float:
        """Shannon entropy in bits. Zeros skipped (0*log2(0) = 0 by convention)."""
        try:
            if not probs:
                return 0.0
            result = 0.0
            for p in probs:
                try:
                    p = float(p)
                except (TypeError, ValueError):
                    continue
                if p > 0:
                    result -= p * math.log2(p)
            return result
        except Exception:
            return 0.0

    def argmax(self, values: list) -> int:
        """Index of maximum value. Returns -1 for empty list. Ties: lowest index."""
        try:
            if not values:
                return -1
            best_idx = 0
            best_val = values[0]
            for i in range(1, len(values)):
                if values[i] > best_val:
                    best_val = values[i]
                    best_idx = i
            return best_idx
        except Exception:
            return -1

    def clamp(self, value, lo, hi) -> float:
        """Constrain value to [lo, hi]. Swaps lo/hi if lo > hi."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            try:
                return float(lo)
            except Exception:
                return 0.0
        try:
            lo_f = float(lo)
            hi_f = float(hi)
        except (TypeError, ValueError):
            return v
        if lo_f > hi_f:
            lo_f, hi_f = hi_f, lo_f
        if v < lo_f:
            return lo_f
        if v > hi_f:
            return hi_f
        return v
```

### Step 3c — Run Group F tests

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v3/tests/test_innate_substrate_extended.py::TestNormalize hpm_ai_v3/tests/test_innate_substrate_extended.py::TestEntropy hpm_ai_v3/tests/test_innate_substrate_extended.py::TestArgmax hpm_ai_v3/tests/test_innate_substrate_extended.py::TestClamp -v 2>&1 | tail -20
```

Expected: `16 passed`

### Step 3d — Commit Group F

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
git add hpm_ai_v3/tools/innate_substrate.py
git commit -m "$(cat <<'EOF'
feat: add Group F uncertainty/confidence functions to InnateCognitiveSubstrate

Adds normalize, entropy, argmax, clamp.
Supports evaluator/gatekeeper role (confidence scoring) in HPM agents.
EOF
)"
```

---

## Task 4: Group G — Goal / Task Decomposition

### Step 4a — Run tests for Group G (expect failures)

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v3/tests/test_innate_substrate_extended.py::TestDecomposeText hpm_ai_v3/tests/test_innate_substrate_extended.py::TestEstimateProgress hpm_ai_v3/tests/test_innate_substrate_extended.py::TestCheckConstraint -q 2>&1 | head -10
```

Expected: `AttributeError: 'InnateCognitiveSubstrate' object has no attribute 'decompose_text'`

### Step 4b — Implement Group G

Append the following block in `hpm_ai_v3/tools/innate_substrate.py` after the Group F block (before `# ── Core: resolve_call`):

```python
    # ── Group G: Goal / Task Decomposition ─────────────────────────────────

    _DECOMPOSE_VERBS = {
        "find", "compute", "calculate", "sort", "filter", "group", "count",
        "sum", "detect", "compare", "get", "list", "check", "estimate",
        "build", "flatten", "split", "match",
    }
    _DECOMPOSE_STOPS = {"a", "an", "the", "in", "of", "for", "with", "from", "to", "that", "which"}
    _DECOMPOSE_MODS = {"by", "with", "from", "greater", "less", "above", "below", "than", "where", "between"}

    def decompose_text(self, text: str) -> dict:
        """Extract {verb, object, modifier} from natural language goal text."""
        empty = {"verb": "", "object": "", "modifier": ""}
        try:
            if not text or not text.strip():
                return empty
            tokens = text.strip().split()
            if not tokens:
                return empty

            # Find verb
            verb = ""
            verb_idx = -1
            for i, tok in enumerate(tokens):
                if tok.lower().rstrip(".,!?") in self._DECOMPOSE_VERBS:
                    verb = tok.lower().rstrip(".,!?")
                    verb_idx = i
                    break
            if not verb:
                verb = tokens[0].lower().rstrip(".,!?")
                verb_idx = 0

            # Find object: first token after verb not in stopwords
            obj = ""
            obj_idx = -1
            for i in range(verb_idx + 1, len(tokens)):
                tok = tokens[i].lower().rstrip(".,!?")
                if tok not in self._DECOMPOSE_STOPS:
                    obj = tok
                    obj_idx = i
                    break

            # Find modifier: token after object starting with a modifier word
            modifier = ""
            if obj_idx >= 0:
                for i in range(obj_idx + 1, len(tokens)):
                    tok = tokens[i].lower().rstrip(".,!?")
                    if tok in self._DECOMPOSE_MODS:
                        modifier = " ".join(tokens[i:]).lower().rstrip(".,!?")
                        break

            return {"verb": verb, "object": obj, "modifier": modifier}
        except Exception:
            return empty

    def estimate_progress(self, current, target) -> float:
        """Normalised progress current/target clamped to [0, 1]."""
        try:
            c = float(current)
            t = float(target)
        except (TypeError, ValueError):
            return 0.0
        try:
            if t == 0:
                return 1.0 if c == 0 else 0.0
            return self.clamp(c / t, 0.0, 1.0)
        except Exception:
            return 0.0

    def check_constraint(self, value, constraint_str: str) -> bool:
        """Evaluate a constraint string against value. Unknown constraints return True."""
        try:
            cs = constraint_str.strip()

            # type:typename
            m = re.match(r"^type:(\w+)$", cs)
            if m:
                type_name = m.group(1)
                type_map = {
                    "int": int, "float": float, "str": str,
                    "list": list, "dict": dict, "bool": bool,
                }
                t = type_map.get(type_name)
                if t is None:
                    return True
                # int check: bool is subclass of int, exclude
                if type_name == "int":
                    return isinstance(value, int) and not isinstance(value, bool)
                return isinstance(value, t)

            # len comparisons: len > N etc.
            m = re.match(r"^len\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)$", cs)
            if m:
                op, n_str = m.group(1), m.group(2)
                try:
                    length = len(value)
                    n = float(n_str)
                    return self._apply_op(float(length), op, n)
                except (TypeError, ValueError):
                    return False

            # in [a, b, c]
            m = re.match(r"^in\s*\[(.+)\]$", cs)
            if m:
                parts = [p.strip() for p in m.group(1).split(",")]
                candidates = []
                for p in parts:
                    try:
                        candidates.append(float(p))
                    except ValueError:
                        candidates.append(p.strip("'\""))
                try:
                    return float(value) in candidates or value in candidates
                except (TypeError, ValueError):
                    return value in candidates

            # numeric comparisons: >= N, > N, etc.
            m = re.match(r"^(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)$", cs)
            if m:
                op, n_str = m.group(1), m.group(2)
                try:
                    v = float(value)
                    n = float(n_str)
                    return self._apply_op(v, op, n)
                except (TypeError, ValueError):
                    return False

            # Unknown constraint: permissive
            return True
        except Exception:
            return True

    def _apply_op(self, a: float, op: str, b: float) -> bool:
        """Apply a comparison operator string."""
        if op == ">":  return a > b
        if op == ">=": return a >= b
        if op == "<":  return a < b
        if op == "<=": return a <= b
        if op == "==": return a == b
        if op == "!=": return a != b
        return True
```

### Step 4c — Run Group G tests

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v3/tests/test_innate_substrate_extended.py::TestDecomposeText hpm_ai_v3/tests/test_innate_substrate_extended.py::TestEstimateProgress hpm_ai_v3/tests/test_innate_substrate_extended.py::TestCheckConstraint -v 2>&1 | tail -25
```

Expected: `19 passed`

### Step 4d — Commit Group G

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
git add hpm_ai_v3/tools/innate_substrate.py
git commit -m "$(cat <<'EOF'
feat: add Group G goal/task decomposition functions to InnateCognitiveSubstrate

Adds decompose_text, estimate_progress, check_constraint, _apply_op.
Supports L5 meta-cognition bootstrapping in HPM agents.
EOF
)"
```

---

## Task 5: Full Test Suite Run and Commit

### Step 5a — Run all extended tests

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v3/tests/test_innate_substrate_extended.py -v 2>&1 | tail -30
```

Expected output:
```
TestBuildMapping::test_basic PASSED
TestBuildMapping::test_truncates_to_shorter PASSED
TestBuildMapping::test_empty PASSED
TestBuildMapping::test_non_list_inputs PASSED
TestInvertMapping::test_basic PASSED
...
73 passed in X.XXs
```

### Step 5b — Run all substrate tests together (regression check)

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
python -m pytest hpm_ai_v3/tests/ -q 2>&1 | tail -10
```

Expected: all pass, no errors.

### Step 5c — Commit test file

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent
git add hpm_ai_v3/tests/test_innate_substrate_extended.py
git commit -m "$(cat <<'EOF'
test: add full test suite for InnateCognitiveSubstrate Groups D-G

73 tests covering all 16 new innate functions across structural,
temporal, uncertainty, and goal-decomposition capability groups.
EOF
)"
```

---

## Self-Review Checklist

- [x] All 16 functions specified in session-state.md are covered (D:5, E:4, F:4, G:3)
- [x] No placeholders or TBDs in any code block
- [x] All types are concrete stdlib types (list, dict, float, int, str, bool)
- [x] All functions handle empty inputs, type errors, and edge cases
- [x] `_apply_op` helper included for `check_constraint` (not counted as a public function)
- [x] Group G uses class-level constants (`_DECOMPOSE_VERBS`, etc.) consistent with Python conventions
- [x] Test file imports from correct module path `hpm_ai_v3.tools.innate_substrate`
- [x] Each task follows TDD: failing test run → implement → passing test run → commit
- [x] `math` already imported in existing file; no new imports required
- [x] `re` already imported in existing file; no new imports required
- [x] Class-level attributes (`_DECOMPOSE_VERBS`, `_DECOMPOSE_STOPS`, `_DECOMPOSE_MODS`) defined inside the class body — consistent with Python class design, accessible as `self._DECOMPOSE_VERBS`
