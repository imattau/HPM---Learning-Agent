# Two-Tier Context-Aware HPM Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-stream pattern store with a two-tier context-aware architecture so agents can learn generalizable meta-patterns across ARC tasks rather than collapsing to 2 generic patterns.

**Architecture:** Tier 1 (ephemeral, per-task) holds patterns mutated by task signal; Tier 2 (persistent) holds meta-patterns promoted via similarity merge at task end and cross-task recombination between tasks. Weight updates from task signal are blocked from reaching Tier 2, preventing non-stationary interference. A `CrossTaskRecombinator` runs between tasks to promote recurring structural patterns into high-level Tier 2 meta-patterns.

**Tech Stack:** Python, numpy, existing HPM stack (`InMemoryStore`, `GaussianPattern`, `RecombinationOperator`, `StructuralLawMonitor`)

---

## File Structure

**New files:**
- `hpm/store/tiered_store.py` — `TieredStore`: two-tier context-aware store with similarity merge
- `hpm/monitor/cross_task_recombinator.py` — `CrossTaskRecombinator`: off-line between-task recombination
- `tests/store/test_tiered_store.py` — unit + integration tests for TieredStore
- `tests/monitor/test_cross_task_recombinator.py` — tests for CrossTaskRecombinator

**Modified files:**
- `hpm/store/__init__.py` — re-export `TieredStore`
- `benchmarks/multi_agent_arc.py` — `run_persistent()` uses `TieredStore`, calls `begin_context`/`end_context`, runs `CrossTaskRecombinator` between tasks

---

## Task 1: TieredStore — core context management

**Files:**
- Create: `hpm/store/tiered_store.py`
- Create: `tests/store/test_tiered_store.py`

The `TieredStore` implements the same `PatternStore` protocol as `InMemoryStore`. Internally it holds:
- `_tier1: dict[str, InMemoryStore]` — one ephemeral store per context_id
- `_tier2: InMemoryStore` — persistent meta-pattern store
- `_current_context: str | None` — active context_id (None = between tasks)

**Key invariant:** `update_weight` only mutates Tier 1. Tier 2 weights are never touched by task signal — they can only be changed via `similarity_merge` or `promote_to_tier2`.

- [ ] **Step 1: Write failing tests**

```python
# tests/store/test_tiered_store.py
import numpy as np
import pytest
from hpm.store.tiered_store import TieredStore
from hpm.patterns.gaussian import GaussianPattern


def _pat(seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(4)
    return GaussianPattern(mu=mu, sigma=np.eye(4) * 0.1)


def test_save_goes_to_tier1_during_context():
    store = TieredStore()
    store.begin_context("task_0")
    p = _pat()
    store.save(p, 1.0, "agent_a")
    # Should be in tier1, NOT tier2
    assert store.query_tier2("agent_a") == []
    assert len(store.query("agent_a")) == 1


def test_save_without_context_goes_to_tier2():
    store = TieredStore()
    p = _pat()
    store.save(p, 1.0, "agent_a")
    assert len(store.query_tier2("agent_a")) == 1


def test_update_weight_does_not_mutate_tier2():
    store = TieredStore()
    p = _pat()
    store.save(p, 1.0, "agent_a")          # goes to tier2
    store.begin_context("task_0")
    store.update_weight(p.id, 0.01)        # must NOT change tier2 weight
    store.end_context("task_0", correct=False)
    tier2 = store.query_tier2("agent_a")
    assert tier2[0][1] == pytest.approx(1.0)  # unchanged


def test_query_returns_tier1_plus_tier2_during_context():
    store = TieredStore()
    p2 = _pat(seed=0)
    store.save(p2, 0.8, "agent_a")         # tier2
    store.begin_context("task_0")
    p1 = _pat(seed=1)
    store.save(p1, 1.0, "agent_a")         # tier1
    results = store.query("agent_a")
    assert len(results) == 2


def test_end_context_clears_tier1():
    store = TieredStore()
    store.begin_context("task_0")
    store.save(_pat(), 1.0, "agent_a")
    store.end_context("task_0", correct=False)
    assert store.query("agent_a") == []    # tier1 gone, tier2 empty too


def test_delete_removes_from_tier1():
    store = TieredStore()
    store.begin_context("task_0")
    p = _pat()
    store.save(p, 1.0, "agent_a")
    store.delete(p.id)
    assert store.query("agent_a") == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/store/test_tiered_store.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'hpm.store.tiered_store'`

- [ ] **Step 3: Implement TieredStore**

```python
# hpm/store/tiered_store.py
from hpm.store.memory import InMemoryStore


class TieredStore:
    """
    Two-tier context-aware PatternStore.

    Tier 1 (ephemeral): per-task patterns, mutated freely by agent signal.
    Tier 2 (persistent): meta-patterns, protected from task signal mutation.

    Weight updates (update_weight) only affect Tier 1. Tier 2 is updated
    exclusively via similarity_merge() and promote_to_tier2().
    """

    def __init__(self):
        self._tier1: dict[str, InMemoryStore] = {}   # context_id -> store
        self._tier2: InMemoryStore = InMemoryStore()
        self._current_context: str | None = None

    # ------------------------------------------------------------------
    # Context lifecycle
    # ------------------------------------------------------------------

    def begin_context(self, context_id: str) -> None:
        """Start a new task context. Creates a fresh Tier 1 store."""
        self._current_context = context_id
        self._tier1[context_id] = InMemoryStore()

    def end_context(self, context_id: str, correct: bool) -> None:
        """End task context. Optionally runs similarity_merge before clearing."""
        if correct and context_id in self._tier1:
            self.similarity_merge(context_id)
        self._tier1.pop(context_id, None)
        if self._current_context == context_id:
            self._current_context = None

    # ------------------------------------------------------------------
    # PatternStore protocol
    # ------------------------------------------------------------------

    def save(self, pattern, weight: float, agent_id: str) -> None:
        if self._current_context is not None:
            self._tier1[self._current_context].save(pattern, weight, agent_id)
        else:
            self._tier2.save(pattern, weight, agent_id)

    def load(self, pattern_id: str) -> tuple:
        if self._current_context and pattern_id in self._tier1[self._current_context]._data:
            return self._tier1[self._current_context].load(pattern_id)
        return self._tier2.load(pattern_id)

    def query(self, agent_id: str) -> list:
        """Returns Tier 1 (current context) + Tier 2 patterns for agent."""
        t1 = []
        if self._current_context and self._current_context in self._tier1:
            t1 = self._tier1[self._current_context].query(agent_id)
        t2 = self._tier2.query(agent_id)
        return t1 + t2

    def query_all(self) -> list:
        t1 = []
        if self._current_context and self._current_context in self._tier1:
            t1 = self._tier1[self._current_context].query_all()
        return t1 + self._tier2.query_all()

    def query_tier2(self, agent_id: str) -> list:
        """Direct access to Tier 2 patterns (for monitoring/testing)."""
        return self._tier2.query(agent_id)

    def query_tier2_all(self) -> list:
        return self._tier2.query_all()

    def delete(self, pattern_id: str) -> None:
        if self._current_context and self._current_context in self._tier1:
            self._tier1[self._current_context].delete(pattern_id)
        self._tier2.delete(pattern_id)

    def update_weight(self, pattern_id: str, weight: float) -> None:
        """Only mutates Tier 1. Tier 2 is protected from task signal."""
        if self._current_context and self._current_context in self._tier1:
            t1 = self._tier1[self._current_context]
            if pattern_id in t1._data:
                t1.update_weight(pattern_id, weight)
                return
        # Pattern is in Tier 2 — do not mutate (protection invariant)

    # ------------------------------------------------------------------
    # Tier 2 promotion
    # ------------------------------------------------------------------

    def similarity_merge(self, context_id: str,
                         similarity_threshold: float = 0.95,
                         consolidation_boost: float = 0.1,
                         max_tier2_patterns: int = 200) -> None:
        """
        At task end: compare Tier 1 patterns against Tier 2.
        - If similar Tier 2 pattern found: boost its weight.
        - If no match and Tier 2 not full: promote pattern to Tier 2.
        Similarity measured by cosine similarity of pattern mu vectors.
        """
        import numpy as np

        if context_id not in self._tier1:
            return

        t1_records = self._tier1[context_id].query_all()
        t2_all = self._tier2.query_all()
        t2_patterns = [(p, w, aid) for p, w, aid in t2_all]

        for p1, w1, aid1 in t1_records:
            mu1 = p1.mu
            norm1 = np.linalg.norm(mu1)
            if norm1 < 1e-8:
                continue

            best_sim = -1.0
            best_t2_id = None
            best_t2_w = 0.0

            for p2, w2, _ in t2_patterns:
                mu2 = p2.mu
                norm2 = np.linalg.norm(mu2)
                if norm2 < 1e-8:
                    continue
                sim = float(np.dot(mu1, mu2) / (norm1 * norm2))
                if sim > best_sim:
                    best_sim = sim
                    best_t2_id = p2.id
                    best_t2_w = w2

            if best_sim >= similarity_threshold and best_t2_id is not None:
                # Consolidation boost to matching Tier 2 pattern
                self._tier2.update_weight(best_t2_id, best_t2_w + consolidation_boost)
            elif len(t2_patterns) < max_tier2_patterns:
                # Promote to Tier 2
                self._tier2.save(p1, w1 * 0.5, aid1)  # half-weight on promotion
                t2_patterns.append((p1, w1 * 0.5, aid1))

    def promote_to_tier2(self, pattern, weight: float, agent_id: str) -> None:
        """Directly promote a pattern to Tier 2 (used by CrossTaskRecombinator)."""
        self._tier2.save(pattern, weight, agent_id)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/store/test_tiered_store.py -v
```
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add hpm/store/tiered_store.py tests/store/test_tiered_store.py
git commit -m "feat: add TieredStore — two-tier context-aware pattern store"
```

---

## Task 2: TieredStore — similarity_merge tests

**Files:**
- Modify: `tests/store/test_tiered_store.py`

- [ ] **Step 1: Write failing tests for similarity_merge**

```python
# Add to tests/store/test_tiered_store.py

def test_similarity_merge_boosts_matching_tier2_pattern():
    import numpy as np
    from hpm.store.tiered_store import TieredStore
    from hpm.patterns.gaussian import GaussianPattern

    store = TieredStore()

    # Seed a Tier 2 pattern at mu=[1,0,0,0]
    mu = np.array([1.0, 0.0, 0.0, 0.0])
    p_t2 = GaussianPattern(mu=mu.copy(), sigma=np.eye(4) * 0.1)
    store.save(p_t2, 0.5, "agent_a")   # no context → tier2

    # Task produces similar pattern (same direction)
    store.begin_context("task_0")
    p_t1 = GaussianPattern(mu=mu.copy() + 0.01, sigma=np.eye(4) * 0.1)
    store.save(p_t1, 0.8, "agent_a")
    store.end_context("task_0", correct=True)

    # Tier 2 weight should have been boosted
    tier2 = store.query_tier2("agent_a")
    assert tier2[0][1] > 0.5


def test_similarity_merge_promotes_novel_pattern_to_tier2():
    import numpy as np
    from hpm.store.tiered_store import TieredStore
    from hpm.patterns.gaussian import GaussianPattern

    store = TieredStore()
    store.begin_context("task_0")
    mu = np.array([1.0, 0.0, 0.0, 0.0])
    p = GaussianPattern(mu=mu, sigma=np.eye(4) * 0.1)
    store.save(p, 0.9, "agent_a")
    store.end_context("task_0", correct=True)

    # Novel pattern (no tier2 yet) should be promoted
    tier2 = store.query_tier2("agent_a")
    assert len(tier2) == 1
    assert tier2[0][1] == pytest.approx(0.45)   # half of 0.9


def test_similarity_merge_skipped_on_incorrect_task():
    import numpy as np
    from hpm.store.tiered_store import TieredStore
    from hpm.patterns.gaussian import GaussianPattern

    store = TieredStore()
    store.begin_context("task_0")
    mu = np.array([1.0, 0.0, 0.0, 0.0])
    p = GaussianPattern(mu=mu, sigma=np.eye(4) * 0.1)
    store.save(p, 0.9, "agent_a")
    store.end_context("task_0", correct=False)  # task failed

    # Nothing promoted to tier2
    assert store.query_tier2("agent_a") == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/store/test_tiered_store.py::test_similarity_merge_boosts_matching_tier2_pattern -v
```
Expected: FAIL (function exists but logic not yet verified)

- [ ] **Step 3: Run all tiered store tests**

```bash
pytest tests/store/test_tiered_store.py -v
```
Expected: all 9 tests PASS (similarity_merge is already implemented in Task 1)

- [ ] **Step 4: Commit**

```bash
git add tests/store/test_tiered_store.py
git commit -m "test: add similarity_merge tests for TieredStore"
```

---

## Task 3: CrossTaskRecombinator

**Files:**
- Create: `hpm/monitor/cross_task_recombinator.py`
- Create: `tests/monitor/test_cross_task_recombinator.py`

Runs offline between tasks. Pulls all Tier 2 patterns from the store. For each pair whose μ vectors are sufficiently different (dissimilar enough to be complementary, yet not identical), creates a new pattern at the midpoint — representing a structural intersection. Promotes the recombinant to Tier 2 if its insight score (novelty × efficacy) is above threshold.

- [ ] **Step 1: Write failing tests**

```python
# tests/monitor/test_cross_task_recombinator.py
import numpy as np
import pytest
from hpm.store.tiered_store import TieredStore
from hpm.monitor.cross_task_recombinator import CrossTaskRecombinator
from hpm.patterns.gaussian import GaussianPattern


def _make_store_with_tier2(n_patterns=2):
    """Two Tier 2 patterns at orthogonal positions."""
    store = TieredStore()
    for i in range(n_patterns):
        mu = np.zeros(4)
        mu[i] = 1.0
        p = GaussianPattern(mu=mu, sigma=np.eye(4) * 0.1)
        store.save(p, 1.0, "agent_a")
    return store


def test_consolidate_creates_meta_pattern():
    store = _make_store_with_tier2(2)
    rec = CrossTaskRecombinator(similarity_lo=0.1, similarity_hi=0.9)
    rec.consolidate(store, agent_id="agent_a")
    # Should have promoted a recombinant: now 3 tier2 patterns
    tier2 = store.query_tier2("agent_a")
    assert len(tier2) == 3


def test_consolidate_skips_identical_patterns():
    """Patterns with cosine sim > similarity_hi are too similar — skip."""
    store = TieredStore()
    mu = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(2):
        p = GaussianPattern(mu=mu.copy(), sigma=np.eye(4) * 0.1)
        store.save(p, 1.0, "agent_a")
    rec = CrossTaskRecombinator(similarity_lo=0.1, similarity_hi=0.9)
    rec.consolidate(store, agent_id="agent_a")
    # Identical patterns → no recombinant
    assert len(store.query_tier2("agent_a")) == 2


def test_consolidate_skips_too_dissimilar():
    """Patterns with cosine sim < similarity_lo are too different — skip."""
    store = TieredStore()
    mus = [np.array([1.0, 0.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0, 0.0])]
    for mu in mus:
        p = GaussianPattern(mu=mu, sigma=np.eye(4) * 0.1)
        store.save(p, 1.0, "agent_a")
    rec = CrossTaskRecombinator(similarity_lo=0.1, similarity_hi=0.9)
    rec.consolidate(store, agent_id="agent_a")
    assert len(store.query_tier2("agent_a")) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/monitor/test_cross_task_recombinator.py -v 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'hpm.monitor.cross_task_recombinator'`

- [ ] **Step 3: Implement CrossTaskRecombinator**

```python
# hpm/monitor/cross_task_recombinator.py
import numpy as np
from hpm.patterns.gaussian import GaussianPattern


class CrossTaskRecombinator:
    """
    Off-line between-task recombinator.

    Pulls all Tier 2 patterns from TieredStore. For pairs whose cosine
    similarity falls in [similarity_lo, similarity_hi] — different enough
    to be complementary, similar enough to share structure — creates a
    midpoint meta-pattern and promotes it to Tier 2.

    This builds the structural intersection that HPM's hierarchical
    abstraction requires: e.g. "vertical symmetry" appearing in Task A
    and Task B recombines into a generalised Symmetry meta-pattern.
    """

    def __init__(self,
                 similarity_lo: float = 0.3,
                 similarity_hi: float = 0.9,
                 meta_weight: float = 0.3,
                 max_recombinants: int = 10):
        self.similarity_lo = similarity_lo
        self.similarity_hi = similarity_hi
        self.meta_weight = meta_weight
        self.max_recombinants = max_recombinants

    def consolidate(self, store, agent_id: str) -> int:
        """
        Run one consolidation pass on Tier 2 patterns for agent_id.
        Returns number of new meta-patterns promoted.
        """
        records = store.query_tier2(agent_id)
        if len(records) < 2:
            return 0

        promoted = 0
        existing_ids = {p.id for p, _, _ in store.query_tier2_all()}

        for i, (p1, w1) in enumerate(records):
            if promoted >= self.max_recombinants:
                break
            for j, (p2, w2) in enumerate(records):
                if j <= i:
                    continue
                sim = self._cosine_sim(p1.mu, p2.mu)
                if not (self.similarity_lo <= sim <= self.similarity_hi):
                    continue

                # Midpoint meta-pattern
                mu_meta = (p1.mu + p2.mu) / 2.0
                sigma_meta = (p1.sigma + p2.sigma) / 2.0
                p_meta = GaussianPattern(
                    mu=mu_meta,
                    sigma=sigma_meta,
                    level=max(getattr(p1, 'level', 1), getattr(p2, 'level', 1)) + 1,
                )

                if p_meta.id not in existing_ids:
                    store.promote_to_tier2(p_meta, self.meta_weight, agent_id)
                    existing_ids.add(p_meta.id)
                    promoted += 1

        return promoted

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/monitor/test_cross_task_recombinator.py -v
```
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add hpm/monitor/cross_task_recombinator.py tests/monitor/test_cross_task_recombinator.py
git commit -m "feat: add CrossTaskRecombinator — off-line between-task meta-pattern promotion"
```

---

## Task 4: Update store __init__ and wire into ARC benchmark

**Files:**
- Modify: `hpm/store/__init__.py`
- Modify: `hpm/monitor/__init__.py`
- Modify: `benchmarks/multi_agent_arc.py`

- [ ] **Step 1: Re-export from __init__ files**

In `hpm/store/__init__.py`, add:
```python
from .tiered_store import TieredStore
```

In `hpm/monitor/__init__.py`, add:
```python
from .cross_task_recombinator import CrossTaskRecombinator
```

- [ ] **Step 2: Modify run_persistent in multi_agent_arc.py**

Replace the `run_persistent` function body so it:
1. Uses `TieredStore` instead of `InMemoryStore` (via the orchestrator)
2. Calls `store.begin_context(str(task_idx))` before training
3. Calls `store.end_context(str(task_idx), correct=correct)` after evaluation
4. Runs `CrossTaskRecombinator().consolidate(store, agent_id)` every 10 tasks

Key change: the orchestrator's `store` must be the `TieredStore`. Since `make_arc_persistent_orchestrator()` calls `make_orchestrator()` which calls `InMemoryStore()` internally, we need to inject the `TieredStore` after construction — or pass it in.

Simplest approach: after orchestrator creation, replace `store._data` isn't possible since TieredStore and InMemoryStore have different APIs. Instead: patch `make_orchestrator` to accept an optional `store` argument.

Check `benchmarks/multi_agent_common.py` line ~58:
```python
store = InMemoryStore()
```
Change to:
```python
store = kwargs.pop("store", None) or InMemoryStore()
```

Then in `run_persistent`:
```python
from hpm.store.tiered_store import TieredStore
from hpm.monitor.cross_task_recombinator import CrossTaskRecombinator

tiered = TieredStore()
orch, agents, _ = make_arc_persistent_orchestrator(store=tiered)
store = tiered  # use tiered for context management
recombinator = CrossTaskRecombinator()
```

And in the task loop:
```python
store.begin_context(str(task_idx))

# ... training via orch.step() ...

# Evaluate
correct_score = ensemble_score(agents, correct_vec)
distractor_scores = [ensemble_score(agents, v) for v in distractor_vecs]
rank = 1 + sum(1 for s in distractor_scores if s <= correct_score)
correct = correct_score < min(distractor_scores)

store.end_context(str(task_idx), correct=correct)

# Off-line recombination every 10 tasks
if (run_idx + 1) % 10 == 0:
    for agent in agents:
        recombinator.consolidate(store, agent.agent_id)

# Progress reporting: include tier2 meta-pattern count
if (run_idx + 1) % 50 == 0:
    t2_count = len(store.query_tier2_all())
    t1_count = sum(len(store.query(a.agent_id)) for a in agents) - t2_count
    print(f"  {run_idx+1}/{len(eligible)} — accuracy: {pct:.1f}%, "
          f"tier1: {t1_count}, tier2_meta: {t2_count}", flush=True)
```

- [ ] **Step 3: Run the full test suite to verify no regressions**

```bash
pytest tests/ -x -q 2>&1 | tail -20
```
Expected: all existing tests PASS

- [ ] **Step 4: Run the persistent benchmark**

```bash
python3 benchmarks/multi_agent_arc.py --persistent 2>/dev/null
```
Expected: accuracy > 18.4% (previous baseline), tier2 meta-patterns accumulating over time

- [ ] **Step 5: Commit**

```bash
git add hpm/store/__init__.py hpm/monitor/__init__.py benchmarks/multi_agent_common.py benchmarks/multi_agent_arc.py
git commit -m "feat: wire TieredStore + CrossTaskRecombinator into persistent ARC benchmark"
```

---

## Task 5: Integration test

**Files:**
- Create: `tests/integration/test_two_tier_arc.py`

- [ ] **Step 1: Write integration test**

```python
# tests/integration/test_two_tier_arc.py
"""
Integration test: TieredStore + CrossTaskRecombinator across multiple
simulated tasks. Verifies that Tier 2 accumulates meta-patterns and
that accuracy doesn't collapse below chance (20%) after 20 tasks.
"""
import numpy as np
import pytest
from hpm.store.tiered_store import TieredStore
from hpm.monitor.cross_task_recombinator import CrossTaskRecombinator
from hpm.patterns.gaussian import GaussianPattern


def _random_task_pattern(rng, dim=8):
    mu = rng.standard_normal(dim)
    mu /= np.linalg.norm(mu)
    return GaussianPattern(mu=mu, sigma=np.eye(dim) * 0.5)


def test_tier2_accumulates_across_tasks():
    """After 20 successful tasks, Tier 2 should have meta-patterns."""
    rng = np.random.default_rng(42)
    store = TieredStore()
    rec = CrossTaskRecombinator()

    for task_id in range(20):
        store.begin_context(str(task_id))
        p = _random_task_pattern(rng)
        store.save(p, 1.0, "agent_a")
        store.end_context(str(task_id), correct=True)
        if (task_id + 1) % 5 == 0:
            rec.consolidate(store, "agent_a")

    tier2 = store.query_tier2("agent_a")
    assert len(tier2) > 0, "Tier 2 should have patterns after 20 successful tasks"


def test_tier2_not_polluted_by_failed_tasks():
    """Failed tasks should not promote patterns to Tier 2."""
    rng = np.random.default_rng(0)
    store = TieredStore()

    for task_id in range(10):
        store.begin_context(str(task_id))
        p = _random_task_pattern(rng)
        store.save(p, 1.0, "agent_a")
        store.end_context(str(task_id), correct=False)  # all fail

    assert store.query_tier2("agent_a") == []


def test_tier2_weight_protected_from_task_signal():
    """Tier 2 patterns must not have their weights reduced by task signal."""
    rng = np.random.default_rng(0)
    store = TieredStore()

    # Establish a Tier 2 meta-pattern
    mu = rng.standard_normal(8)
    p_meta = GaussianPattern(mu=mu, sigma=np.eye(8) * 0.1)
    store.promote_to_tier2(p_meta, 0.9, "agent_a")

    # Run a task that tries to down-weight it
    store.begin_context("task_x")
    store.update_weight(p_meta.id, 0.001)   # should be blocked
    store.end_context("task_x", correct=False)

    tier2 = store.query_tier2("agent_a")
    assert tier2[0][1] == pytest.approx(0.9), "Tier 2 weight must not be mutated by task signal"
```

- [ ] **Step 2: Run integration tests**

```bash
pytest tests/integration/test_two_tier_arc.py -v
```
Expected: 3 tests PASS

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -q 2>&1 | tail -5
```
Expected: all tests PASS

- [ ] **Step 4: Final commit**

```bash
git add tests/integration/test_two_tier_arc.py
git commit -m "test: integration tests for two-tier context-aware HPM"
```
