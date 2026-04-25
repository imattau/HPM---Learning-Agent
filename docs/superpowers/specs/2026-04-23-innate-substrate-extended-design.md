# Spec: InnateCognitiveSubstrate Extended — Groups D, E, F, G

Date: 2026-04-23
Status: Draft

## 1. Purpose

Extend `InnateCognitiveSubstrate` with 16 additional innate functions across four new groups. These functions give HPM agents structural, temporal, probabilistic, and goal-decomposition capabilities as permanent cognitive infrastructure — not subject to replicator dynamics, never in the population.

## 2. HPM Alignment

| Group | HPM Level | Role |
|-------|-----------|------|
| D: Structural/Relational | L3 — Relational rules | Build and traverse relational structures |
| E: Temporal/Sequential | L4 — Generative simulation | Detect and reason over sequences |
| F: Uncertainty/Confidence | Evaluator/Gatekeeper | Score confidence; inform pattern selection |
| G: Goal/Task Decomposition | L5 — Meta-cognition | Parse goals; track progress; enforce constraints |

## 3. Group D: Structural / Relational Reasoning (5 functions)

### 3.1 `build_mapping(keys, values) -> dict`
- **Purpose**: Pair a key list with a value list into a dict.
- **Inputs**: `keys: list`, `values: list`
- **Output**: `dict` — zip of keys and values. If lists differ in length, zip truncates to the shorter.
- **Edge cases**: empty lists → `{}`; non-list inputs → treat as single-element list before zipping.
- **Errors**: never raises; returns `{}` on any exception.

### 3.2 `invert_mapping(mapping) -> dict`
- **Purpose**: Swap keys and values.
- **Inputs**: `mapping: dict`
- **Output**: `dict` — `{v: k for k, v in mapping.items()}`. If values are unhashable, skip those entries silently.
- **Edge cases**: empty dict → `{}`; duplicate values → last key wins.
- **Errors**: never raises.

### 3.3 `get_nested(obj, path) -> Any`
- **Purpose**: Safe deep access into nested dicts/lists using a dotted string path.
- **Inputs**: `obj: Any`, `path: str` — dot-separated keys/indices e.g. `"a.b.0.c"`.
- **Output**: value at path, or `None` if any step fails.
- **Edge cases**: numeric path segments used as list indices; non-existent keys → `None`; `path=""` → return `obj`.
- **Errors**: never raises.

### 3.4 `flatten(nested) -> list`
- **Purpose**: Recursively flatten any nested list structure into a single flat list.
- **Inputs**: `nested: Any`
- **Output**: `list` of non-list items, depth-first left-to-right.
- **Edge cases**: non-list scalar → `[scalar]`; empty list → `[]`; dict inside list → kept as-is (not recursed into).
- **Errors**: never raises.

### 3.5 `group_by(items, key_fn) -> dict`
- **Purpose**: Partition a list into a dict of lists keyed by `key_fn(item)`.
- **Inputs**: `items: list`, `key_fn: callable`
- **Output**: `dict[Any, list]` — items sharing the same key value are grouped together, preserving order.
- **Edge cases**: empty list → `{}`; `key_fn` raises on an item → skip that item silently.
- **Errors**: never raises.

## 4. Group E: Temporal / Sequential Reasoning (4 functions)

### 4.1 `detect_trend(series) -> str`
- **Purpose**: Characterise overall direction of a numeric series.
- **Inputs**: `series: list` of numeric values (at least 2).
- **Output**: one of `"increasing"` | `"decreasing"` | `"stable"` | `"volatile"`
- **Algorithm**:
  1. Compute diffs between consecutive elements.
  2. If all diffs == 0 → `"stable"`.
  3. If all diffs > 0 → `"increasing"`.
  4. If all diffs < 0 → `"decreasing"`.
  5. Otherwise → `"volatile"`.
- **Edge cases**: series length < 2 → `"stable"`; non-numeric values → attempt `float()` conversion; skip failures.
- **Errors**: never raises; returns `"stable"` on exception.

### 4.2 `diff_sequence(series) -> list`
- **Purpose**: Compute first-order differences.
- **Inputs**: `series: list` of numeric values.
- **Output**: `list` of length `len(series) - 1` containing `series[i+1] - series[i]`.
- **Edge cases**: length 0 or 1 → `[]`; non-numeric values → attempt `float()` conversion; return `[]` on failure.
- **Errors**: never raises.

### 4.3 `find_repeating(sequence) -> Any`
- **Purpose**: Detect the smallest repeating unit in a sequence.
- **Inputs**: `sequence: list`
- **Output**: the repeating sub-sequence as a `list`, or `None` if none found.
- **Algorithm**: for period length 1..len//2, check if sequence is fully tiled by that period. Return first match.
- **Edge cases**: empty or length-1 sequence → `None`; sequence that doesn't repeat → `None`.
- **Errors**: never raises.

### 4.4 `sliding_window(sequence, n) -> list`
- **Purpose**: Extract all contiguous windows of size `n`.
- **Inputs**: `sequence: list`, `n: int`
- **Output**: `list` of `list`s, each of length `n`. Length of output: `max(0, len(sequence) - n + 1)`.
- **Edge cases**: `n <= 0` → `[]`; `n > len(sequence)` → `[]`.
- **Errors**: never raises.

## 5. Group F: Uncertainty / Confidence (4 functions)

### 5.1 `normalize(values) -> list`
- **Purpose**: Convert a list of non-negative numerics to a probability distribution (sum = 1).
- **Inputs**: `values: list` of numeric
- **Output**: `list[float]` each in [0, 1], summing to 1.0.
- **Edge cases**: all zeros → uniform distribution (1/n each); single value → `[1.0]`; empty → `[]`.
- **Errors**: never raises; returns `[]` on non-numeric input.

### 5.2 `entropy(probs) -> float`
- **Purpose**: Compute Shannon entropy (base 2) of a probability distribution.
- **Inputs**: `probs: list[float]` — should sum to ~1; zeros are skipped (0 * log(0) = 0 by convention).
- **Output**: `float` — entropy in bits. Maximum for n outcomes = log2(n).
- **Edge cases**: empty list → `0.0`; single nonzero prob → `0.0`; negative values → treated as 0.
- **Errors**: never raises.

### 5.3 `argmax(values) -> int`
- **Purpose**: Return the index of the maximum value.
- **Inputs**: `values: list` of comparable values
- **Output**: `int` index of maximum. Ties: return the first (lowest index).
- **Edge cases**: empty list → `-1`; single element → `0`.
- **Errors**: never raises; returns `-1` on exception.

### 5.4 `clamp(value, lo, hi) -> float`
- **Purpose**: Constrain a numeric value to the range [lo, hi].
- **Inputs**: `value: Any`, `lo: Any`, `hi: Any` — all coercible to float.
- **Output**: `float` — `lo` if value < lo, `hi` if value > hi, else `value`.
- **Edge cases**: `lo > hi` → swap lo and hi before clamping; non-numeric → return `lo`.
- **Errors**: never raises.

## 6. Group G: Goal / Task Decomposition (3 functions)

### 6.1 `decompose_text(text) -> dict`
- **Purpose**: Extract a simple goal structure from natural language text.
- **Inputs**: `text: str`
- **Output**: `dict` with keys `verb`, `object`, `modifier` — all strings (empty string if not found).
- **Algorithm** (pure regex / heuristic, no NLP deps):
  1. `verb`: first token that is a known action word from a small fixed vocabulary (`["find", "compute", "calculate", "sort", "filter", "group", "count", "sum", "detect", "compare", "get", "list", "check", "estimate", "build", "flatten", "split", "match"]`), case-insensitive. If none found, use first word.
  2. `object`: first noun phrase approximation — first token after the verb that is not a stopword (`["a", "an", "the", "in", "of", "for", "with", "from", "to", "that", "which"]`).
  3. `modifier`: any token after `object` starting with a preposition or qualifier (`["by", "with", "from", "greater", "less", "above", "below", "than", "where", "between"]`). Empty string if none.
- **Edge cases**: empty string → `{"verb": "", "object": "", "modifier": ""}`; single word → verb set, object/modifier empty.
- **Errors**: never raises.

### 6.2 `estimate_progress(current, target) -> float`
- **Purpose**: Compute normalised progress toward a scalar target.
- **Inputs**: `current: Any`, `target: Any` — coercible to float.
- **Output**: `float` in [0.0, 1.0].
- **Formula**: `clamp(current / target, 0, 1)` if target != 0, else `1.0` if current == target else `0.0`.
- **Edge cases**: target == 0 and current == 0 → `1.0`; target == 0 and current != 0 → `0.0`; non-numeric → `0.0`.
- **Errors**: never raises.

### 6.3 `check_constraint(value, constraint_str) -> bool`
- **Purpose**: Evaluate a simple numeric or type constraint expressed as a string.
- **Inputs**: `value: Any`, `constraint_str: str`
- **Output**: `bool`
- **Supported constraint forms** (parsed with regex):
  - `"> N"`, `">= N"`, `"< N"`, `"<= N"`, `"== N"`, `"!= N"` — numeric comparisons
  - `"in [a, b, c]"` — membership check (values parsed as floats where possible)
  - `"type:typename"` — type check e.g. `"type:int"`, `"type:str"`, `"type:list"`
  - `"len > N"`, `"len >= N"`, `"len < N"`, `"len <= N"`, `"len == N"` — length checks
- **Edge cases**: unrecognised constraint → `True` (permissive default); non-numeric value for numeric constraint → `False`.
- **Errors**: never raises.

## 7. Implementation Constraints

- All 16 functions are pure Python, stdlib only (`re`, `math`).
- All functions are methods of `InnateCognitiveSubstrate` — same class, same file.
- No function raises an exception under any input; all have safe fallbacks.
- All return types are concrete Python builtins (no `Optional` return for functions that have safe defaults).
- New groups follow the exact comment-header style of existing groups.
- `key_fn` parameter in `group_by` is a Python callable passed directly — not a string.

## 8. Test Requirements

- One test file: `hpm_ai_v3/tests/test_innate_substrate_extended.py`
- At minimum 3 test cases per function (happy path, edge case, error/boundary).
- No mocking — all tests call methods on a live `InnateCognitiveSubstrate()` instance.
- Test file uses only `pytest` and stdlib.

## 9. Files Modified / Created

| Path | Action |
|------|--------|
| `hpm_ai_v3/tools/innate_substrate.py` | Modify — append Groups D, E, F, G |
| `hpm_ai_v3/tests/test_innate_substrate_extended.py` | Create — full test suite |
