# Polygraph View Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-view PatternEngine instances with a lightweight view_matches dict, keeping full view engines only for PAB/CartPole's act() path.

**Architecture:** PatternEngine.observe() gains update_state=False mode for view observations. HPMPipeline adds view_matches dict populated by primary engine. view_engines kept only when polygraph_evaluator selects a view for act().

**Tech Stack:** Python 3.12, pytest, uv

---

## File Map

| File | Action |
|------|--------|
| `hpm_ai_v5/core/engine.py` | **Modify** — add `update_state=True` param to `observe()` |
| `hpm_ai_v5/pipeline.py` | **Modify** — add `view_matches`, primary engine observes views |
| `hpm_ai_v5/experiments/run_snlp_benchmark.py` | **Modify** — `view_engines.get(n)` → `view_matches.get(n)` |
| `hpm_ai_v5/experiments/run_atis_benchmark.py` | **Modify** — remove view_engines iteration |
| `hpm_ai_v5/planning/code_recognition.py` | **Modify** — `view_engines.get(n)` → `view_matches.get(n)` |

---

## Task 1: PatternEngine.observe(update_state=True)

**Files:**
- Modify: `hpm_ai_v5/core/engine.py`

- [ ] **Step 1: Write failing test**

```python
# Add to hpm_ai_v5/tests/test_core.py or a new tests/test_view_refactor.py
def test_observe_update_state_false_does_not_change_current_state():
    from hpm_ai_v5.core import PatternEngine, State
    engine = PatternEngine()
    s1 = State(value=(1.0, 2.0))
    engine.observe(s1)
    assert engine.current_state == s1
    s2 = State(value=(3.0, 4.0))
    engine.observe(s2, update_state=False)
    assert engine.current_state == s1  # unchanged

def test_observe_update_state_false_still_learns_pattern():
    from hpm_ai_v5.core import PatternEngine, State
    engine = PatternEngine()
    s1 = State(value=(1.0, 2.0))
    engine.observe(s1)
    engine.observe(s1)  # reinforce
    s2 = State(value=(1.1, 2.1))
    match = engine.observe(s2, update_state=False)
    assert match is not None
    assert match.status in ("exact", "near")  # matched the learned pattern
```

- [ ] **Step 2: Run test — verify it fails**

```bash
uv run pytest tests/test_view_refactor.py -v
```
Expected: AttributeError — `observe()` has no `update_state` param.

- [ ] **Step 3: Add update_state param to observe()**

In `hpm_ai_v5/core/engine.py`, find `def observe(self, state: State)` and update:

```python
def observe(self, state: State, *, update_state: bool = True) -> MatchResult | None:
```

Find the lines that update current_state and history (around line 196-237):
```python
    if self.current_state is None:
        self.current_state = state
    ...
    self.current_state = state      # line ~236
    self.history.append(state)      # line ~99
```

Wrap the state/history update at the end of observe() (line ~236 onwards):
```python
        if update_state:
            self.current_state = state
```

Also wrap the history append in `_record_pattern_name` or wherever history is appended — check that history is only updated when `update_state=True`. Pass `update_state` down if needed.

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_view_refactor.py -v
```
Expected: 2 PASSED.

- [ ] **Step 5: Run full test suite to verify no regressions**

```bash
uv run pytest hpm_ai_v5/tests/ -v 2>&1 | tail -20
```
Expected: all PASSED.

- [ ] **Step 6: Commit**

```bash
git add hpm_ai_v5/core/engine.py tests/test_view_refactor.py
git commit -m "feat: PatternEngine.observe(update_state=False) for view observations"
```

---

## Task 2: HPMPipeline.view_matches

**Files:**
- Modify: `hpm_ai_v5/pipeline.py`

- [ ] **Step 1: Read pipeline.py lines 50-130 to understand current view_engines logic**

```bash
sed -n '50,130p' hpm_ai_v5/pipeline.py
```

- [ ] **Step 2: Write failing test**

```python
# Add to tests/test_view_refactor.py
def test_pipeline_view_matches_populated():
    from hpm_ai_v5.core import PatternEngine
    from hpm_ai_v5.core.config import CoreConfig
    from hpm_ai_v5.adapter.validation_only import ValidationOnlyAdapter
    from hpm_ai_v5.pipeline import HPMPipeline
    from hpm_ai_v5.polygraphs.nlp import StructuralNLPPolygraphGenerator
    from hpm_ai_v5.adapter.nlp import NLPTokenizer, SkeletonExtractor, SkeletonNgramAdapter

    engine = PatternEngine(config=CoreConfig(max_patterns=64, near_threshold=0.4))
    pipeline = HPMPipeline(
        preprocessor=NLPTokenizer(),
        engine=engine,
        postprocessor=ValidationOnlyAdapter(),
        polygraph_generator=StructuralNLPPolygraphGenerator(),
        polygraph_confidence_skip=1.1,
    )
    pipeline.register_preprocessor(SkeletonExtractor())
    pipeline.register_preprocessor(SkeletonNgramAdapter())

    pipeline.step("What is the weather in London?")
    pipeline.step("What is the weather in Paris?")

    assert "skeleton_view" in pipeline.view_matches
    assert "skeleton_bigram_view" in pipeline.view_matches
    assert pipeline.view_matches["skeleton_view"] is not None
```

- [ ] **Step 3: Run test — verify it fails**

```bash
uv run pytest tests/test_view_refactor.py::test_pipeline_view_matches_populated -v
```
Expected: AttributeError — pipeline has no `view_matches`.

- [ ] **Step 4: Add view_matches to HPMPipeline**

In `hpm_ai_v5/pipeline.py`:

Add after `self.view_engines: dict[str, PatternEngine] = {}`:
```python
self.view_matches: dict[str, MatchResult] = {}
```

Add import for MatchResult at top if not present:
```python
from .core.store import MatchResult
```

In the view processing loop (where `view_engines.setdefault(view.name, PatternEngine())` is called), add:
```python
# Fast path: observe view through primary engine, store match
match = self.engine.observe(view.state, update_state=False)
if match is not None:
    self.view_matches[view.name] = match
```

Keep the existing `view_engines` creation for when `polygraph_evaluator` needs `act()`.

In the `clear()` / reset paths, also clear `view_matches`.

- [ ] **Step 5: Run tests — verify they pass**

```bash
uv run pytest tests/test_view_refactor.py -v
```
Expected: all PASSED.

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest hpm_ai_v5/tests/ -v 2>&1 | tail -20
```
Expected: no regressions.

- [ ] **Step 7: Commit**

```bash
git add hpm_ai_v5/pipeline.py tests/test_view_refactor.py
git commit -m "feat: add view_matches dict to HPMPipeline (lightweight view observation)"
```

---

## Task 3: Migrate SNLP benchmark to view_matches

**Files:**
- Modify: `hpm_ai_v5/experiments/run_snlp_benchmark.py`

- [ ] **Step 1: Replace all view_engines accesses**

Search and replace pattern:
```
self.pipeline.view_engines.get("skeleton_view")
→ type: MatchResultProxy (see below) OR direct MatchResult
```

Since callers do `view_engine.last_match.status`, create a tiny adapter or just
update each call site to use `view_matches` directly:

```python
# Before:
view_engine = self.pipeline.view_engines.get("skeleton_view")
if view_engine and view_engine.last_match:
    if view_engine.last_match.status in ("exact", "near"):

# After:
match = self.pipeline.view_matches.get("skeleton_view")
if match and match.status in ("exact", "near"):
```

Apply same pattern to `skeleton_bigram_view` access in T4.

For `view_engines.clear()` in `reset_for_isolation()`:
```python
# Before:
self.pipeline.view_engines.clear()
# After:
self.pipeline.view_engines.clear()
self.pipeline.view_matches.clear()
```

For `view_engines.items()` in T3:
```python
# Before:
for view_name, engine in self.pipeline.view_engines.items():
    if view_name.startswith("semantic_view_") and engine.last_match:
        if engine.last_match.status in ("exact", "near"):
# After:
for view_name, match in self.pipeline.view_matches.items():
    if view_name.startswith("semantic_view_") and match:
        if match.status in ("exact", "near"):
```

- [ ] **Step 2: Run SNLP benchmark 3 times to verify no regression**

```bash
for i in 1 2 3; do uv run python -m hpm_ai_v5.experiments.run_snlp_benchmark 2>&1 | grep "^T[1-5]:"; echo "---"; done
```
Expected: T2 >= 80%, T3 = 100%, T5 = 100% (same as before refactor).

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v5/experiments/run_snlp_benchmark.py
git commit -m "refactor: SNLP benchmark uses view_matches instead of view_engines"
```

---

## Task 4: Migrate code_recognition.py

**Files:**
- Modify: `hpm_ai_v5/planning/code_recognition.py`

- [ ] **Step 1: Update view_engines references**

```python
# Before (line ~87):
view_engine = self.pipeline.view_engines.get(view_name)
if view_engine and view_engine.last_match and view_engine.last_match.pattern:
    view_patterns[view_name].add(view_engine.last_match.pattern.name)

# After:
match = self.pipeline.view_matches.get(view_name)
if match and match.pattern:
    view_patterns[view_name].add(match.pattern.name)
```

```python
# Before (line ~137):
for ve in self.pipeline.view_engines.values():
    ...
# After: iterate view_matches
for match in self.pipeline.view_matches.values():
    ...
```

- [ ] **Step 2: Run code benchmark tests**

```bash
uv run pytest hpm_ai_v5/tests/ -k "code" -v 2>&1 | tail -20
```
Expected: no regressions.

- [ ] **Step 3: Commit**

```bash
git add hpm_ai_v5/planning/code_recognition.py
git commit -m "refactor: code_recognition uses view_matches instead of view_engines"
```

---

## Task 5: Final validation

- [ ] **Step 1: Full test suite**

```bash
uv run pytest hpm_ai_v5/tests/ tests/ -v 2>&1 | tail -30
```
Expected: all PASSED. Specifically verify:
- `test_cartpole.py::test_selected_view` — `selected_view == "test_view"` still works
- `test_pab.py` — view selection still functions
- `test_core.py` — no engine regressions

- [ ] **Step 2: SNLP 3-run check**

```bash
for i in 1 2 3; do uv run python -m hpm_ai_v5.experiments.run_snlp_benchmark 2>&1 | grep "^T[1-5]:"; echo "---"; done
```

- [ ] **Step 3: ATIS benchmark**

```bash
uv run python -m hpm_ai_v5.experiments.run_atis_benchmark 2>&1 | tail -10
```

- [ ] **Step 4: Final commit**

```bash
git add -u
git commit -m "refactor: polygraph view_matches complete — view_engines only for act() path"
```
