# Negative Patterns / Inhibitory Tier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an inhibitory tier to TieredStore and PatternField so that failed tasks contribute taboo knowledge that penalises confusing candidates in ensemble scoring and propagates across agents via social inhibition.
**Architecture:** A new `_tier2_negative` InMemoryStore partition is added to TieredStore; on failure, `negative_merge()` promotes conflicting Tier 1 patterns (cosine similarity ≥ `neg_conflict_threshold` to existing positive Tier 2) to the negative store at half weight. PatternField gains a parallel `_negative` channel for broadcasting/pulling inhibitory patterns across agents, and `ensemble_score` subtracts the negative NLL contribution from the total score so candidates resembling taboo patterns rank worse.
**Tech Stack:** Python, numpy, pytest

---

## Task 1: TieredStore negative partition

**Modify:** `hpm/store/tiered_store.py`
**New test file:** `tests/store/test_tiered_store_negative.py`

### 1.1 Code changes to `hpm/store/tiered_store.py`

Replace the `__init__` method and add four new methods. The full set of changes:

**In `__init__`**, add one line after `self._tier2`:
```python
self._tier2_negative: InMemoryStore = InMemoryStore()
```

**Updated `end_context`** (add `elif` branch and forward kwargs):
```python
def end_context(self, context_id: str, correct: bool,
                neg_conflict_threshold: float = 0.7,
                max_tier2_negative: int = 100) -> None:
    """End task context. Runs similarity_merge on success, negative_merge on failure."""
    if correct and context_id in self._tier1:
        self.similarity_merge(context_id)
    elif not correct and context_id in self._tier1:
        self.negative_merge(context_id,
                            neg_conflict_threshold=neg_conflict_threshold,
                            max_tier2_negative=max_tier2_negative)
    self._tier1.pop(context_id, None)
    if self._current_context == context_id:
        self._current_context = None
```

**New method `query_negative`**:
```python
def query_negative(self, agent_id: str) -> list:
    """Return all negative Tier 2 patterns for agent_id as list[(pattern, weight)]."""
    return self._tier2_negative.query(agent_id)
```

**New method `query_tier2_negative_all`**:
```python
def query_tier2_negative_all(self) -> list:
    """Return all negative Tier 2 records (for monitoring/testing)."""
    return self._tier2_negative.query_all()
```

**New method `negative_merge`**:
```python
def negative_merge(self, context_id: str,
                   neg_conflict_threshold: float = 0.7,
                   max_tier2_negative: int = 100) -> None:
    """
    Promote conflicting Tier 1 patterns to _tier2_negative.

    A Tier 1 pattern from a failed task is promoted if its cosine similarity
    to any positive Tier 2 pattern >= neg_conflict_threshold. This encodes the
    taboo: a pattern shape highly similar to a known-good meta-pattern was
    present during failure — it is misleadingly similar (anti-predictive).

    Patterns with zero norm are skipped (same guard as similarity_merge).
    New patterns are silently dropped when cap is reached.
    """
    import numpy as np

    if context_id not in self._tier1:
        return

    t1_records = self._tier1[context_id].query_all()
    t2_all = self._tier2.query_all()

    for p1, w1, aid1 in t1_records:
        mu1 = p1.mu
        norm1 = np.linalg.norm(mu1)
        if norm1 < 1e-8:
            continue

        best_sim = -1.0
        for p2, w2, _ in t2_all:
            mu2 = p2.mu
            norm2 = np.linalg.norm(mu2)
            if norm2 < 1e-8:
                continue
            sim = float(np.dot(mu1, mu2) / (norm1 * norm2))
            if sim > best_sim:
                best_sim = sim

        if best_sim >= neg_conflict_threshold:
            current_neg = self._tier2_negative.query_all()
            if len(current_neg) < max_tier2_negative:
                self._tier2_negative.save(p1, w1 * 0.5, aid1)
```

### 1.2 New test file `tests/store/test_tiered_store_negative.py`

```python
import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern
from hpm.store.tiered_store import TieredStore


def _pat(mu, dim=4):
    return GaussianPattern(mu=np.array(mu, dtype=float), sigma=np.eye(dim) * 0.1)


# ── Test 1: empty by default ────────────────────────────────────────────────

def test_query_negative_empty_by_default():
    store = TieredStore()
    assert store.query_negative("agent_a") == []
    assert store.query_tier2_negative_all() == []


# ── Test 2: end_context(correct=False) promotes conflicting pattern ──────────

def test_negative_merge_promotes_conflicting():
    store = TieredStore()
    # Seed a positive Tier 2 pattern at mu=[1,0,0,0]
    p_pos = _pat([1.0, 0.0, 0.0, 0.0])
    store.save(p_pos, 1.0, "agent_a")  # no context → tier2

    # Failed task: Tier 1 pattern highly similar to the positive (cosine ~0.9998)
    store.begin_context("task_fail")
    p_neg = _pat([1.0, 0.05, 0.0, 0.0])
    store.save(p_neg, 0.8, "agent_a")
    store.end_context("task_fail", correct=False)

    results = store.query_negative("agent_a")
    assert len(results) == 1
    pattern, weight = results[0]
    assert weight == pytest.approx(0.4)  # 0.8 * 0.5


# ── Test 3: end_context(correct=False) discards non-conflicting pattern ──────

def test_negative_merge_discards_non_conflicting():
    store = TieredStore()
    # Seed a positive Tier 2 pattern at mu=[1,0,0,0]
    p_pos = _pat([1.0, 0.0, 0.0, 0.0])
    store.save(p_pos, 1.0, "agent_a")  # tier2

    # Failed task: Tier 1 pattern orthogonal to positive (cosine = 0.0)
    store.begin_context("task_fail")
    p_ortho = _pat([0.0, 1.0, 0.0, 0.0])
    store.save(p_ortho, 0.8, "agent_a")
    store.end_context("task_fail", correct=False)

    # Orthogonal → no conflict detected → discarded
    assert store.query_negative("agent_a") == []


# ── Test 4: max_tier2_negative cap is respected ──────────────────────────────

def test_negative_merge_respects_cap():
    store = TieredStore()
    # Seed one positive Tier 2 pattern
    p_pos = _pat([1.0, 0.0, 0.0, 0.0])
    store.save(p_pos, 1.0, "agent_a")

    cap = 3

    # Fill _tier2_negative to the cap via direct saves (bypassing merge)
    for i in range(cap):
        dummy = _pat([1.0, 0.0, 0.0, 0.0])
        store._tier2_negative.save(dummy, 0.5, "agent_a")

    assert len(store.query_tier2_negative_all()) == cap

    # Now a failed task with a conflicting pattern — cap already reached
    store.begin_context("task_overflow")
    p_conflict = _pat([1.0, 0.02, 0.0, 0.0])
    store.save(p_conflict, 0.9, "agent_a")
    store.end_context("task_overflow", correct=False,
                      max_tier2_negative=cap)

    # Still exactly cap patterns — new one silently dropped
    assert len(store.query_tier2_negative_all()) == cap


# ── Test 5: end_context(correct=True) does NOT populate negative store ───────

def test_end_context_correct_no_negative_merge():
    store = TieredStore()
    p_pos = _pat([1.0, 0.0, 0.0, 0.0])
    store.save(p_pos, 1.0, "agent_a")  # tier2

    store.begin_context("task_ok")
    p_t1 = _pat([1.0, 0.01, 0.0, 0.0])
    store.save(p_t1, 0.8, "agent_a")
    store.end_context("task_ok", correct=True)

    assert store.query_negative("agent_a") == []


# ── Test 6: query() (positive) does NOT return negative patterns ─────────────

def test_positive_query_excludes_negative_patterns():
    store = TieredStore()
    p_pos = _pat([1.0, 0.0, 0.0, 0.0])
    store.save(p_pos, 1.0, "agent_a")  # tier2

    store.begin_context("task_fail")
    p_neg = _pat([1.0, 0.05, 0.0, 0.0])
    store.save(p_neg, 0.8, "agent_a")
    store.end_context("task_fail", correct=False)

    positive_results = store.query("agent_a")
    neg_results = store.query_negative("agent_a")

    assert len(neg_results) == 1
    neg_id = neg_results[0][0].id
    positive_ids = {p.id for p, _ in positive_results}
    assert neg_id not in positive_ids


# ── Test 7: Tier 2 positive patterns unaffected by negative_merge ────────────

def test_tier2_positive_unaffected_by_negative_merge():
    store = TieredStore()
    p_pos = _pat([1.0, 0.0, 0.0, 0.0])
    store.save(p_pos, 1.0, "agent_a")  # tier2

    store.begin_context("task_fail")
    p_t1 = _pat([1.0, 0.02, 0.0, 0.0])
    store.save(p_t1, 0.8, "agent_a")
    store.end_context("task_fail", correct=False)

    # Tier 2 positive weight must remain unchanged at 1.0
    tier2_pos = store.query_tier2("agent_a")
    assert len(tier2_pos) == 1
    assert tier2_pos[0][1] == pytest.approx(1.0)
```

### 1.3 Run commands

```
cd /home/mattthomson/workspace/HPM---Learning-Agent && python -m pytest tests/store/test_tiered_store_negative.py -v
```

Expected output:
```
tests/store/test_tiered_store_negative.py::test_query_negative_empty_by_default PASSED
tests/store/test_tiered_store_negative.py::test_negative_merge_promotes_conflicting PASSED
tests/store/test_tiered_store_negative.py::test_negative_merge_discards_non_conflicting PASSED
tests/store/test_tiered_store_negative.py::test_negative_merge_respects_cap PASSED
tests/store/test_tiered_store_negative.py::test_end_context_correct_no_negative_merge PASSED
tests/store/test_tiered_store_negative.py::test_positive_query_excludes_negative_patterns PASSED
tests/store/test_tiered_store_negative.py::test_tier2_positive_unaffected_by_negative_merge PASSED

7 passed in 0.XXs
```

Also verify existing store tests still pass:
```
cd /home/mattthomson/workspace/HPM---Learning-Agent && python -m pytest tests/store/test_tiered_store.py -v
```

Expected: all existing tests pass unchanged.

### 1.4 Commit

```
cd /home/mattthomson/workspace/HPM---Learning-Agent && git add hpm/store/tiered_store.py tests/store/test_tiered_store_negative.py && git commit -m "feat(store): add _tier2_negative partition, negative_merge, query_negative"
```

---

## Task 2: AgentConfig new fields

**Modify:** `hpm/config.py`

### 2.1 Code changes to `hpm/config.py`

Add three new fields at the end of the `AgentConfig` dataclass, after `min_recomb_level`:

```python
# Negative / inhibitory tier (inhibitory pattern learning)
gamma_neg: float = 0.3             # social inhibition attenuation (0 = off)
neg_conflict_threshold: float = 0.7  # cosine sim threshold for negative_merge
max_tier2_negative: int = 100      # cap on _tier2_negative store size
```

The full block added to `AgentConfig` (appended after line 62 `min_recomb_level: int = 4`):

```python
    # Negative / inhibitory tier
    gamma_neg: float = 0.3               # social inhibition attenuation (0 = off)
    neg_conflict_threshold: float = 0.7  # cosine sim threshold for negative_merge
    max_tier2_negative: int = 100        # cap on _tier2_negative store size
```

### 2.2 Tests (inline, add to `tests/test_config.py` if it exists, else create)

Create `tests/test_config_negative.py`:

```python
from hpm.config import AgentConfig


def test_agent_config_negative_defaults():
    cfg = AgentConfig(agent_id="a", feature_dim=4)
    assert cfg.gamma_neg == 0.3
    assert cfg.neg_conflict_threshold == 0.7
    assert cfg.max_tier2_negative == 100


def test_agent_config_negative_fields_overridable():
    cfg = AgentConfig(agent_id="a", feature_dim=4,
                      gamma_neg=0.0,
                      neg_conflict_threshold=0.5,
                      max_tier2_negative=50)
    assert cfg.gamma_neg == 0.0
    assert cfg.neg_conflict_threshold == 0.5
    assert cfg.max_tier2_negative == 50


def test_neg_conflict_threshold_distinct_from_conflict_threshold():
    """neg_conflict_threshold (0.7) must not collide with conflict_threshold (0.1)."""
    cfg = AgentConfig(agent_id="a", feature_dim=4)
    assert cfg.conflict_threshold == 0.1        # existing field unchanged
    assert cfg.neg_conflict_threshold == 0.7    # new field
    assert cfg.conflict_threshold != cfg.neg_conflict_threshold
```

### 2.3 Run commands

```
cd /home/mattthomson/workspace/HPM---Learning-Agent && python -m pytest tests/test_config_negative.py -v
```

Expected output:
```
tests/test_config_negative.py::test_agent_config_negative_defaults PASSED
tests/test_config_negative.py::test_agent_config_negative_fields_overridable PASSED
tests/test_config_negative.py::test_neg_conflict_threshold_distinct_from_conflict_threshold PASSED

3 passed in 0.XXs
```

### 2.4 Commit

```
cd /home/mattthomson/workspace/HPM---Learning-Agent && git add hpm/config.py tests/test_config_negative.py && git commit -m "feat(config): add gamma_neg, neg_conflict_threshold, max_tier2_negative to AgentConfig"
```

---

## Task 3: PatternField inhibitory channel

**Modify:** `hpm/field/field.py`
**New test file:** `tests/field/test_field_negative.py`

### 3.1 Code changes to `hpm/field/field.py`

**In `__init__`**, add one new attribute after `self._broadcast_queue`:

```python
# Inhibitory channel: maps agent_id -> [(pattern, weight), ...]
# Represents current negative pattern state (not cumulative; reset before each broadcast).
self._negative: dict[str, list[tuple]] = {}
```

**New method `broadcast_negative`**:

```python
def broadcast_negative(self, pattern, weight: float, agent_id: str) -> None:
    """
    Register a negative pattern from agent_id into the inhibitory channel.
    Appends to the agent's current step registration (caller must clear before
    re-broadcasting each step to avoid accumulation).
    """
    if agent_id not in self._negative:
        self._negative[agent_id] = []
    self._negative[agent_id].append((pattern, weight))
```

**New method `pull_negative`**:

```python
def pull_negative(self, agent_id: str, gamma_neg: float) -> list:
    """
    Return all negative patterns from OTHER agents, attenuated by gamma_neg.
    Returns list[(pattern, attenuated_weight)].
    Does not include patterns broadcast by agent_id itself.
    """
    result = []
    for src_id, records in self._negative.items():
        if src_id == agent_id:
            continue
        for pattern, weight in records:
            result.append((pattern, weight * gamma_neg))
    return result
```

### 3.2 New test file `tests/field/test_field_negative.py`

```python
import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern
from hpm.field.field import PatternField


def _pat(seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(4)
    return GaussianPattern(mu=mu, sigma=np.eye(4) * 0.1)


# ── Test 1: broadcast stores pattern; another agent can pull it ──────────────

def test_broadcast_negative_stores_pattern():
    field = PatternField()
    p = _pat(0)
    field.broadcast_negative(p, 1.0, "agent_a")

    pulled = field.pull_negative("agent_b", 1.0)
    assert len(pulled) == 1
    pulled_pattern, pulled_weight = pulled[0]
    assert pulled_pattern is p
    assert pulled_weight == pytest.approx(1.0)


# ── Test 2: agent does NOT pull its own patterns back ────────────────────────

def test_pull_negative_excludes_own():
    field = PatternField()
    p = _pat(0)
    field.broadcast_negative(p, 1.0, "agent_a")

    # agent_a pulls — should not see its own pattern
    pulled = field.pull_negative("agent_a", 1.0)
    assert pulled == []


# ── Test 3: gamma_neg attenuates weight ──────────────────────────────────────

def test_pull_negative_attenuates_weight():
    field = PatternField()
    p = _pat(0)
    field.broadcast_negative(p, 1.0, "agent_a")

    pulled = field.pull_negative("agent_b", 0.3)
    assert len(pulled) == 1
    _, weight = pulled[0]
    assert weight == pytest.approx(0.3)


# ── Test 4: multiple agents broadcast; third agent pulls all ─────────────────

def test_multiple_agents_broadcast():
    field = PatternField()
    p_a = _pat(0)
    p_b = _pat(1)
    field.broadcast_negative(p_a, 1.0, "agent_a")
    field.broadcast_negative(p_b, 0.8, "agent_b")

    pulled_by_c = field.pull_negative("agent_c", 0.5)
    assert len(pulled_by_c) == 2
    weights = {w for _, w in pulled_by_c}
    assert pytest.approx(0.5) in weights   # 1.0 * 0.5
    assert pytest.approx(0.4) in weights   # 0.8 * 0.5


# ── Test 5: gamma_neg=0 returns patterns with zero weight ───────────────────

def test_pull_negative_gamma_zero():
    field = PatternField()
    p = _pat(0)
    field.broadcast_negative(p, 1.0, "agent_a")

    pulled = field.pull_negative("agent_b", 0.0)
    assert len(pulled) == 1
    _, weight = pulled[0]
    assert weight == pytest.approx(0.0)


# ── Test 6: _negative channel is independent from positive channel ───────────

def test_negative_channel_independent_from_positive():
    field = PatternField()
    p = _pat(0)
    # Register positive pattern
    field.register("agent_a", [(p.id, 1.0)])
    # Broadcast negative pattern (different object)
    p_neg = _pat(1)
    field.broadcast_negative(p_neg, 1.0, "agent_a")

    # Positive pull (via freq) is unaffected
    assert field.freq(p.id) > 0.0
    # Negative channel returns exactly one entry
    pulled = field.pull_negative("agent_b", 1.0)
    assert len(pulled) == 1
    assert pulled[0][0] is p_neg


# ── Test 7: clearing _negative before re-broadcast prevents accumulation ─────

def test_negative_not_cumulative_after_reset():
    field = PatternField()
    p1 = _pat(0)
    p2 = _pat(1)

    # Step 1: agent_a broadcasts p1
    field._negative["agent_a"] = []
    field.broadcast_negative(p1, 1.0, "agent_a")

    # Step 2: agent_a resets and broadcasts p2 only
    field._negative["agent_a"] = []
    field.broadcast_negative(p2, 1.0, "agent_a")

    pulled = field.pull_negative("agent_b", 1.0)
    assert len(pulled) == 1
    assert pulled[0][0] is p2  # only p2, not p1 + p2
```

### 3.3 Run commands

```
cd /home/mattthomson/workspace/HPM---Learning-Agent && python -m pytest tests/field/test_field_negative.py -v
```

Expected output:
```
tests/field/test_field_negative.py::test_broadcast_negative_stores_pattern PASSED
tests/field/test_field_negative.py::test_pull_negative_excludes_own PASSED
tests/field/test_field_negative.py::test_pull_negative_attenuates_weight PASSED
tests/field/test_field_negative.py::test_multiple_agents_broadcast PASSED
tests/field/test_field_negative.py::test_pull_negative_gamma_zero PASSED
tests/field/test_field_negative.py::test_negative_channel_independent_from_positive PASSED
tests/field/test_field_negative.py::test_negative_not_cumulative_after_reset PASSED

7 passed in 0.XXs
```

### 3.4 Commit

```
cd /home/mattthomson/workspace/HPM---Learning-Agent && git add hpm/field/field.py tests/field/test_field_negative.py && git commit -m "feat(field): add inhibitory channel broadcast_negative and pull_negative"
```

---

## Task 4: Agent integration

**Modify:** `hpm/agents/agent.py`
**New test file:** `tests/agents/test_agent_negative.py`

### 4.1 Code changes to `hpm/agents/agent.py`

At the end of the `step()` method, after the existing `communicated_out = 0` block and before the `return` statement, add the two negative channel operations:

```python
        # Inhibitory channel: Step A — pull negative patterns from other agents
        if self.field is not None and hasattr(self.store, '_tier2_negative'):
            neg_incoming = self.field.pull_negative(self.agent_id, self.config.gamma_neg)
            for pattern, weight in neg_incoming:
                self.store._tier2_negative.save(pattern, weight, self.agent_id)

        # Inhibitory channel: Step B — broadcast own negative patterns to field
        if self.field is not None and hasattr(self.store, 'query_negative'):
            self.field._negative[self.agent_id] = []   # reset before re-broadcasting
            for pattern, weight in self.store.query_negative(self.agent_id):
                self.field.broadcast_negative(pattern, weight, self.agent_id)
```

This block is inserted immediately before the `return {` statement at line 301, after the `communicated_out` assignment block (lines 297-299 in the current file).

### 4.2 New test file `tests/agents/test_agent_negative.py`

```python
import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.field.field import PatternField
from hpm.store.tiered_store import TieredStore
from hpm.patterns.gaussian import GaussianPattern


def _cfg(agent_id, gamma_neg=0.3):
    return AgentConfig(
        agent_id=agent_id,
        feature_dim=4,
        gamma_neg=gamma_neg,
        gamma_soc=0.5,
    )


def _obs(seed=0):
    return np.random.default_rng(seed).standard_normal(4)


# ── Test 1: Agent with InMemoryStore does NOT crash (backward compatible) ─────

def test_step_backward_compatible_no_tiered_store():
    """Agents using InMemoryStore (no _tier2_negative) must still work."""
    from hpm.store.memory import InMemoryStore
    cfg = _cfg("agent_a")
    agent = Agent(cfg, store=InMemoryStore())
    # No field — pure single agent, no negative channel
    result = agent.step(_obs(0))
    assert "t" in result
    assert result["t"] == 1


# ── Test 2: Agent pulls negative patterns from field into _tier2_negative ─────

def test_agent_pulls_negative_from_field():
    field = PatternField()
    store_a = TieredStore()
    store_b = TieredStore()

    agent_a = Agent(_cfg("agent_a"), store=store_a, field=field)
    agent_b = Agent(_cfg("agent_b"), store=store_b, field=field)

    # Seed agent_a's negative store directly
    p_taboo = GaussianPattern(mu=np.array([1.0, 0.0, 0.0, 0.0]),
                              sigma=np.eye(4) * 0.1)
    store_a._tier2_negative.save(p_taboo, 0.8, "agent_a")

    # agent_a steps: broadcasts its negative to field
    agent_a.step(_obs(0))

    # agent_b steps: pulls negative from field into its own _tier2_negative
    agent_b.step(_obs(1))

    neg_b = store_b.query_negative("agent_b")
    assert len(neg_b) >= 1
    # Weight should be attenuated by gamma_neg=0.3: 0.8 * 0.3 = 0.24
    weights = [w for _, w in neg_b]
    assert any(abs(w - pytest.approx(0.24, abs=1e-6)) < 0.01 for w in weights)


# ── Test 3: Agent does NOT pull its own negative patterns back ────────────────

def test_agent_does_not_pull_own_negative():
    field = PatternField()
    store_a = TieredStore()

    agent_a = Agent(_cfg("agent_a"), store=store_a, field=field)

    p_taboo = GaussianPattern(mu=np.array([1.0, 0.0, 0.0, 0.0]),
                              sigma=np.eye(4) * 0.1)
    store_a._tier2_negative.save(p_taboo, 0.8, "agent_a")

    before_count = len(store_a.query_negative("agent_a"))
    agent_a.step(_obs(0))  # broadcasts own patterns; should NOT pull own back

    after_count = len(store_a.query_negative("agent_a"))
    # After step, count should be same or less (not doubled)
    # pull_negative excludes own agent_id, so no self-import occurs
    assert after_count == before_count


# ── Test 4: gamma_neg=0 prevents any taboo import ────────────────────────────

def test_gamma_neg_zero_prevents_taboo_import():
    field = PatternField()
    store_a = TieredStore()
    store_b = TieredStore()

    agent_a = Agent(_cfg("agent_a", gamma_neg=0.0), store=store_a, field=field)
    agent_b = Agent(_cfg("agent_b", gamma_neg=0.0), store=store_b, field=field)

    p_taboo = GaussianPattern(mu=np.array([1.0, 0.0, 0.0, 0.0]),
                              sigma=np.eye(4) * 0.1)
    store_a._tier2_negative.save(p_taboo, 0.8, "agent_a")

    agent_a.step(_obs(0))   # broadcasts
    agent_b.step(_obs(1))   # pulls with gamma_neg=0.0

    # Patterns arrive with weight 0.0 — still saved (weight=0), or none saved.
    # Key invariant: no non-zero weight patterns added.
    neg_b = store_b.query_negative("agent_b")
    assert all(w == pytest.approx(0.0) for _, w in neg_b)


# ── Test 5: field _negative cleared and re-broadcast each step (not cumulative)

def test_negative_broadcast_not_cumulative():
    field = PatternField()
    store_a = TieredStore()

    agent_a = Agent(_cfg("agent_a"), store=store_a, field=field)

    p1 = GaussianPattern(mu=np.array([1.0, 0.0, 0.0, 0.0]), sigma=np.eye(4) * 0.1)
    store_a._tier2_negative.save(p1, 0.8, "agent_a")

    agent_a.step(_obs(0))   # step 1: broadcasts p1
    count_after_step1 = len(field._negative.get("agent_a", []))

    agent_a.step(_obs(1))   # step 2: clears and re-broadcasts p1 (only once)
    count_after_step2 = len(field._negative.get("agent_a", []))

    # Must not grow each step
    assert count_after_step2 == count_after_step1
```

### 4.3 Run commands

```
cd /home/mattthomson/workspace/HPM---Learning-Agent && python -m pytest tests/agents/test_agent_negative.py -v
```

Expected output:
```
tests/agents/test_agent_negative.py::test_step_backward_compatible_no_tiered_store PASSED
tests/agents/test_agent_negative.py::test_agent_pulls_negative_from_field PASSED
tests/agents/test_agent_negative.py::test_agent_does_not_pull_own_negative PASSED
tests/agents/test_agent_negative.py::test_gamma_neg_zero_prevents_taboo_import PASSED
tests/agents/test_agent_negative.py::test_negative_broadcast_not_cumulative PASSED

5 passed in 0.XXs
```

### 4.4 Commit

```
cd /home/mattthomson/workspace/HPM---Learning-Agent && git add hpm/agents/agent.py tests/agents/test_agent_negative.py && git commit -m "feat(agent): wire inhibitory field pull/broadcast into Agent.step()"
```

---

## Task 5: ARC scoring — ensemble_score inhibitory subtraction

**Modify:** `benchmarks/multi_agent_arc.py`
**New test file:** `tests/integration/test_negative_arc_scoring.py`

### 5.1 Code changes to `benchmarks/multi_agent_arc.py`

Replace the existing `ensemble_score` function (lines 96-112) with:

```python
def ensemble_score(agents, vec: np.ndarray) -> float:
    """
    Compute ensemble score for a candidate vector.

    Positive patterns contribute to the total (higher NLL = less probable = worse).
    Negative patterns are subtracted: a candidate close to a taboo pattern has low
    NLL under that pattern, so subtracting a small value leaves the score HIGH (worse).
    A candidate far from taboo patterns has high NLL, so subtracting a large value
    makes the score LOWER (better rank).

    Formula:
        total += pos_weight * pos_NLL   (existing behaviour)
        total -= neg_weight * neg_NLL   (new inhibitory term)

    Lower total = more preferred candidate (consistent with existing ranking).
    Returns 0.0 if all stores are empty (backward compatible).

    Sign convention: GaussianPattern.log_prob returns NLL (lower = more probable).
    """
    total = 0.0
    any_records = False
    for agent in agents:
        pos = agent.store.query(agent.agent_id)
        if pos:
            any_records = True
            total += sum(w * p.log_prob(vec) for p, w in pos)
        if hasattr(agent.store, 'query_negative'):
            neg = agent.store.query_negative(agent.agent_id)
            if neg:
                any_records = True
                total -= sum(w * p.log_prob(vec) for p, w in neg)
    return total if any_records else 0.0
```

### 5.2 New test file `tests/integration/test_negative_arc_scoring.py`

```python
import numpy as np
import pytest
from hpm.patterns.gaussian import GaussianPattern
from hpm.store.tiered_store import TieredStore
from hpm.config import AgentConfig
from hpm.agents.agent import Agent


# Import ensemble_score from benchmark module
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from benchmarks.multi_agent_arc import ensemble_score


def _agent_with_stores(agent_id, pos_mu=None, neg_mu=None, feature_dim=4):
    """Helper: create agent with TieredStore, optionally seeded with patterns."""
    cfg = AgentConfig(agent_id=agent_id, feature_dim=feature_dim)
    store = TieredStore()
    # Seed positive tier2 directly (no context needed)
    if pos_mu is not None:
        p = GaussianPattern(mu=np.array(pos_mu, dtype=float),
                            sigma=np.eye(feature_dim) * 0.5)
        store.save(p, 1.0, agent_id)  # no context → tier2
    # Seed negative tier2 directly
    if neg_mu is not None:
        p_neg = GaussianPattern(mu=np.array(neg_mu, dtype=float),
                                sigma=np.eye(feature_dim) * 0.5)
        store._tier2_negative.save(p_neg, 1.0, agent_id)
    agent = Agent.__new__(Agent)
    agent.config = cfg
    agent.agent_id = agent_id
    agent.store = store
    return agent


# ── Test 1: no negative patterns → score identical to legacy behaviour ────────

def test_inhibitory_no_effect_when_empty():
    """Backward compatibility: empty negative store must not change score."""
    agent = _agent_with_stores("a", pos_mu=[1.0, 0.0, 0.0, 0.0])
    vec = np.array([1.0, 0.1, 0.0, 0.0])

    # Compute score using only positive (simulate old behaviour)
    pos_records = agent.store.query("a")
    expected = sum(w * p.log_prob(vec) for p, w in pos_records)

    actual = ensemble_score([agent], vec)
    assert actual == pytest.approx(expected)


# ── Test 2: negative pattern raises score for close candidate ─────────────────

def test_inhibitory_raises_score_for_close_candidate():
    """
    A candidate close to the negative pattern (low neg_NLL) should have a
    *higher* total score (worse rank) compared to baseline without inhibition.
    """
    pos_mu = [1.0, 0.0, 0.0, 0.0]
    # Negative pattern centred ON the distractor
    distractor_vec = np.array([0.0, 1.0, 0.0, 0.0])
    neg_mu = [0.0, 1.0, 0.0, 0.0]

    agent_baseline = _agent_with_stores("a", pos_mu=pos_mu)
    agent_inhibited = _agent_with_stores("b", pos_mu=pos_mu, neg_mu=neg_mu)

    score_baseline = ensemble_score([agent_baseline], distractor_vec)
    score_inhibited = ensemble_score([agent_inhibited], distractor_vec)

    # Close to taboo → subtracting small neg_NLL → score stays higher
    # (or at least no improvement for the distractor)
    # The inhibited score should be >= baseline (taboo penalises close candidates)
    assert score_inhibited >= score_baseline


# ── Test 3: correct candidate wins with inhibitory term ───────────────────────

def test_correct_wins_with_inhibitory():
    """
    Correct candidate is far from taboo; distractor is close to taboo.
    With inhibitory term, correct_score < distractor_score.
    """
    dim = 4
    # Positive pattern centred at [1,0,0,0]
    pos_mu = [1.0, 0.0, 0.0, 0.0]
    # Taboo pattern centred at [0,1,0,0]
    neg_mu = [0.0, 1.0, 0.0, 0.0]

    # Correct output vector: similar to positive, far from taboo
    correct_vec = np.array([0.9, 0.0, 0.0, 0.0])
    # Distractor: close to taboo pattern
    distractor_vec = np.array([0.05, 0.95, 0.0, 0.0])

    agent = _agent_with_stores("a", pos_mu=pos_mu, neg_mu=neg_mu)

    correct_score = ensemble_score([agent], correct_vec)
    distractor_score = ensemble_score([agent], distractor_vec)

    # Correct should rank better (lower score) than distractor
    assert correct_score < distractor_score, (
        f"Expected correct_score ({correct_score:.4f}) < distractor_score ({distractor_score:.4f})"
    )


# ── Test 4: returns 0.0 when all stores empty ─────────────────────────────────

def test_ensemble_score_returns_zero_when_empty():
    cfg = AgentConfig(agent_id="a", feature_dim=4)
    store = TieredStore()
    agent = Agent.__new__(Agent)
    agent.config = cfg
    agent.agent_id = "a"
    agent.store = store
    # Both positive and negative stores are empty — seed removed
    store._tier2 = store._tier2   # just access to confirm it's an InMemoryStore

    # Manually ensure completely empty
    vec = np.zeros(4)
    result = ensemble_score([agent], vec)
    assert result == 0.0


# ── Test 5: InMemoryStore agents (no query_negative) backward compatible ──────

def test_ensemble_score_backward_compatible_inmemory_store():
    """Agents using InMemoryStore (no query_negative method) must not crash."""
    from hpm.store.memory import InMemoryStore
    cfg = AgentConfig(agent_id="a", feature_dim=4)
    store = InMemoryStore()
    p = GaussianPattern(mu=np.array([1.0, 0.0, 0.0, 0.0]), sigma=np.eye(4) * 0.5)
    store.save(p, 1.0, "a")

    agent = Agent.__new__(Agent)
    agent.config = cfg
    agent.agent_id = "a"
    agent.store = store

    vec = np.array([1.0, 0.1, 0.0, 0.0])
    result = ensemble_score([agent], vec)
    assert isinstance(result, float)
    assert result > 0.0
```

### 5.3 Run commands

```
cd /home/mattthomson/workspace/HPM---Learning-Agent && python -m pytest tests/integration/test_negative_arc_scoring.py -v
```

Expected output:
```
tests/integration/test_negative_arc_scoring.py::test_inhibitory_no_effect_when_empty PASSED
tests/integration/test_negative_arc_scoring.py::test_inhibitory_raises_score_for_close_candidate PASSED
tests/integration/test_negative_arc_scoring.py::test_correct_wins_with_inhibitory PASSED
tests/integration/test_negative_arc_scoring.py::test_ensemble_score_returns_zero_when_empty PASSED
tests/integration/test_negative_arc_scoring.py::test_ensemble_score_backward_compatible_inmemory_store PASSED

5 passed in 0.XXs
```

### 5.4 Commit

```
cd /home/mattthomson/workspace/HPM---Learning-Agent && git add benchmarks/multi_agent_arc.py tests/integration/test_negative_arc_scoring.py && git commit -m "feat(arc): ensemble_score subtracts negative NLL; backward compatible"
```

---

## Task 6: StructuralLawMonitor metrics and RecombinationStrategist Fear Reset

**Modify:** `hpm/monitor/structural_law.py` and `hpm/monitor/recombination_strategist.py`

### 6.1 Code changes to `hpm/monitor/structural_law.py`

**Add a module-level helper function** after the `_sigmoid` function:

```python
def _taboo_overlap(agents) -> float:
    """
    Fraction of negative pattern UUIDs present in ALL agents vs any agent (Jaccard).
    Returns 0.0 if no agents have negative patterns.
    """
    neg_id_sets = [
        {p.id for p, _ in agent.store.query_negative(agent.agent_id)}
        for agent in agents
        if hasattr(agent.store, 'query_negative')
    ]
    # Filter out agents with empty negative stores for union/intersection purposes
    non_empty = [s for s in neg_id_sets if s]
    if not non_empty:
        return 0.0
    union = non_empty[0].union(*non_empty[1:])
    if not union:
        return 0.0
    intersection = non_empty[0].intersection(*non_empty[1:])
    return len(intersection) / len(union)
```

**Update the `step` method signature** to accept `agents` as a required parameter (it already receives `agents` — verify the call site passes it). The `step` method already receives `agents: list` in the current implementation.

**Update `_compute_light`** to accept and compute `negative_count` and `taboo_overlap`. Modify `step()` to compute these and pass them through:

In `step()`, after computing `light`, add:

```python
        negative_count = sum(
            len(agent.store.query_negative(agent.agent_id))
            for agent in agents
            if hasattr(agent.store, 'query_negative')
        )
        taboo_ov = _taboo_overlap(agents)
        light["negative_count"] = negative_count
        light["taboo_overlap"] = taboo_ov
```

**Update `_print_table`** to include the new metrics. Modify the `cols` list and `row` dict:

```python
    def _print_table(self, step_t, light, diversity, redundancy):
        title = f"Field Quality Report (step {step_t})"
        cols = ["Patterns", "L4+", "L4+ Weight", "Diversity", "Redundancy",
                "Conflict", "Stable", "NegCount", "TabooOvlp"]
        row = {
            "Patterns": str(light["pattern_count"]),
            "L4+": str(light["level4plus_count"]),
            "L4+ Weight": f"{light['level4plus_mean_weight']:.2f}",
            "Diversity": f"{diversity:.2f}" if diversity is not None else "—",
            "Redundancy": f"{redundancy:.2f}" if redundancy is not None else "—",
            "Conflict": f"{light['conflict']:.2f}",
            "Stable": f"{light['stability_mean']:.2f}",
            "NegCount": str(light.get("negative_count", 0)),
            "TabooOvlp": f"{light.get('taboo_overlap', 0.0):.2f}",
        }
        col_widths = {c: max(len(c), len(row[c])) for c in cols}
        sep = "   "
        header = sep.join(c.ljust(col_widths[c]) for c in cols)
        total_width = max(len(title), len(header))
        print()
        print(title)
        print("─" * total_width)
        print(header)
        print("─" * total_width)
        print(sep.join(row[c].ljust(col_widths[c]) for c in cols))
        print()
```

**Update `_log_json`** to include the new metrics:

```python
    def _log_json(self, step_t, light, diversity, redundancy):
        entry = {
            "step": step_t,
            **light,
            "level_distribution": {str(k): v for k, v in light["level_distribution"].items()},
            "diversity": diversity,
            "redundancy": redundancy,
            "negative_count": light.get("negative_count", 0),
            "taboo_overlap": light.get("taboo_overlap", 0.0),
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

**Updated `step` method return** — add the new metrics to the returned dict:

```python
        return {
            **light,
            "diversity": heavy_diversity,
            "redundancy": heavy_redundancy,
            "negative_count": light.get("negative_count", 0),
            "taboo_overlap": light.get("taboo_overlap", 0.0),
        }
```

### 6.2 Code changes to `hpm/monitor/recombination_strategist.py`

**Add Fear Reset fields to `__init__`**. After the existing `self._conflict_persistent_cycles` attribute, add:

```python
        # Fear Reset state
        self.fear_threshold: float = 0.8
        self.fear_reset_duration: int = 20
        self._fear_reset_remaining: int = 0
        self._saved_gamma_neg: dict = {}  # agent_id -> saved gamma_neg
```

**Add Fear Reset logic to `step()` method**. After the existing `if diversity is not None:` block and before the `return` statement:

```python
        # Fear Reset: if taboo_overlap exceeds threshold, zero gamma_neg temporarily
        taboo = float(field_quality.get("taboo_overlap", 0.0))
        fear_reset_fired = False

        if taboo > self.fear_threshold and self._fear_reset_remaining == 0:
            # Trigger Fear Reset
            self._fear_reset_remaining = self.fear_reset_duration
            for agent in agents:
                self._saved_gamma_neg[agent.agent_id] = getattr(
                    agent.config, 'gamma_neg', 0.3
                )
                if hasattr(agent.config, 'gamma_neg'):
                    agent.config.gamma_neg = 0.0
            fear_reset_fired = True

        if self._fear_reset_remaining > 0:
            self._fear_reset_remaining -= 1
            if self._fear_reset_remaining == 0:
                # Restore gamma_neg for all agents
                for agent in agents:
                    if agent.agent_id in self._saved_gamma_neg:
                        if hasattr(agent.config, 'gamma_neg'):
                            agent.config.gamma_neg = self._saved_gamma_neg[agent.agent_id]
                self._saved_gamma_neg = {}
```

**Update the return dict** of `step()` to include Fear Reset status:

```python
        return {
            "burst_active": self._burst_steps_remaining > 0,
            "kappa_0": kappa_0_applied,
            "beta_c_scaled": beta_c_scaled,
            "stagnation_count": self._stagnation_count,
            "cooldown_remaining": self._cooldown_steps_remaining,
            "fear_reset_active": self._fear_reset_remaining > 0,
            "fear_reset_fired": fear_reset_fired,
        }
```

### 6.3 Tests — extend monitor tests

Create `tests/monitor/test_monitor_negative.py`:

```python
import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.store.tiered_store import TieredStore
from hpm.field.field import PatternField
from hpm.monitor.structural_law import StructuralLawMonitor, _taboo_overlap
from hpm.monitor.recombination_strategist import RecombinationStrategist
from hpm.patterns.gaussian import GaussianPattern


def _make_agent(agent_id, gamma_neg=0.3):
    cfg = AgentConfig(agent_id=agent_id, feature_dim=4, gamma_neg=gamma_neg)
    store = TieredStore()
    agent = Agent(cfg, store=store)
    return agent


def _seed_negative(agent, mu_list):
    """Seed negative patterns directly into _tier2_negative."""
    for mu in mu_list:
        p = GaussianPattern(mu=np.array(mu, dtype=float), sigma=np.eye(4) * 0.1)
        agent.store._tier2_negative.save(p, 0.5, agent.agent_id)


# ── Test 1: negative_count in monitor report ──────────────────────────────────

def test_monitor_reports_negative_count():
    agents = [_make_agent("a"), _make_agent("b")]
    _seed_negative(agents[0], [[1.0, 0.0, 0.0, 0.0]])
    _seed_negative(agents[1], [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    store = agents[0].store
    monitor = StructuralLawMonitor(store=store, T_monitor=1, verbose=False)
    result = monitor.step(step_t=1, agents=agents, total_conflict=0.0)

    assert "negative_count" in result
    assert result["negative_count"] == 3  # 1 + 2


# ── Test 2: taboo_overlap computes correctly ──────────────────────────────────

def test_taboo_overlap_no_shared_patterns():
    """Two agents with fully distinct negative UUIDs → overlap = 0.0."""
    agents = [_make_agent("a"), _make_agent("b")]
    _seed_negative(agents[0], [[1.0, 0.0, 0.0, 0.0]])
    _seed_negative(agents[1], [[0.0, 1.0, 0.0, 0.0]])

    overlap = _taboo_overlap(agents)
    # Different UUIDs (GaussianPattern generates fresh UUID each time)
    assert overlap == pytest.approx(0.0)


def test_taboo_overlap_fully_shared():
    """Two agents sharing the same pattern object → overlap = 1.0."""
    agents = [_make_agent("a"), _make_agent("b")]
    p = GaussianPattern(mu=np.array([1.0, 0.0, 0.0, 0.0]), sigma=np.eye(4) * 0.1)
    agents[0].store._tier2_negative.save(p, 0.5, "a")
    agents[1].store._tier2_negative.save(p, 0.5, "b")  # same UUID

    overlap = _taboo_overlap(agents)
    assert overlap == pytest.approx(1.0)


def test_taboo_overlap_empty_stores():
    """No agents have negative patterns → overlap = 0.0."""
    agents = [_make_agent("a"), _make_agent("b")]
    overlap = _taboo_overlap(agents)
    assert overlap == pytest.approx(0.0)


# ── Test 3: taboo_overlap in monitor report ───────────────────────────────────

def test_monitor_reports_taboo_overlap():
    agents = [_make_agent("a"), _make_agent("b")]
    store = agents[0].store
    monitor = StructuralLawMonitor(store=store, T_monitor=1, verbose=False)
    result = monitor.step(step_t=1, agents=agents, total_conflict=0.0)

    assert "taboo_overlap" in result
    assert 0.0 <= result["taboo_overlap"] <= 1.0


# ── Test 4: RecombinationStrategist Fear Reset fires on high taboo_overlap ────

def test_fear_reset_fires_when_taboo_overlap_high():
    strategist = RecombinationStrategist()
    strategist.fear_threshold = 0.8

    agents = [_make_agent("a", gamma_neg=0.3), _make_agent("b", gamma_neg=0.3)]
    field_quality = {"taboo_overlap": 0.9, "diversity": None, "conflict": 0.0}

    result = strategist.step(step_t=1, field_quality=field_quality, agents=agents)

    assert result["fear_reset_fired"] is True
    # All agents should have gamma_neg zeroed
    for agent in agents:
        assert agent.config.gamma_neg == pytest.approx(0.0)


# ── Test 5: Fear Reset restores gamma_neg after duration ─────────────────────

def test_fear_reset_restores_gamma_neg():
    strategist = RecombinationStrategist()
    strategist.fear_threshold = 0.8
    strategist.fear_reset_duration = 3

    agents = [_make_agent("a", gamma_neg=0.3)]
    high_taboo = {"taboo_overlap": 0.9, "diversity": None, "conflict": 0.0}
    low_taboo = {"taboo_overlap": 0.0, "diversity": None, "conflict": 0.0}

    # Fire reset
    strategist.step(step_t=1, field_quality=high_taboo, agents=agents)
    assert agents[0].config.gamma_neg == pytest.approx(0.0)

    # Tick through duration (3 steps)
    for t in range(2, 5):
        strategist.step(step_t=t, field_quality=low_taboo, agents=agents)

    # After duration, gamma_neg restored to 0.3
    assert agents[0].config.gamma_neg == pytest.approx(0.3)


# ── Test 6: Fear Reset does not fire when taboo_overlap below threshold ───────

def test_fear_reset_does_not_fire_below_threshold():
    strategist = RecombinationStrategist()
    strategist.fear_threshold = 0.8

    agents = [_make_agent("a", gamma_neg=0.3)]
    field_quality = {"taboo_overlap": 0.5, "diversity": None, "conflict": 0.0}

    result = strategist.step(step_t=1, field_quality=field_quality, agents=agents)

    assert result["fear_reset_fired"] is False
    assert agents[0].config.gamma_neg == pytest.approx(0.3)


# ── Test 7: Fear Reset result keys present even when not fired ────────────────

def test_fear_reset_result_keys_always_present():
    strategist = RecombinationStrategist()
    agents = [_make_agent("a")]
    field_quality = {"taboo_overlap": 0.0, "diversity": None, "conflict": 0.0}

    result = strategist.step(step_t=1, field_quality=field_quality, agents=agents)

    assert "fear_reset_active" in result
    assert "fear_reset_fired" in result
```

### 6.4 Run commands

```
cd /home/mattthomson/workspace/HPM---Learning-Agent && python -m pytest tests/monitor/test_monitor_negative.py -v
```

Expected output:
```
tests/monitor/test_monitor_negative.py::test_monitor_reports_negative_count PASSED
tests/monitor/test_monitor_negative.py::test_taboo_overlap_no_shared_patterns PASSED
tests/monitor/test_monitor_negative.py::test_taboo_overlap_fully_shared PASSED
tests/monitor/test_monitor_negative.py::test_taboo_overlap_empty_stores PASSED
tests/monitor/test_monitor_negative.py::test_monitor_reports_taboo_overlap PASSED
tests/monitor/test_monitor_negative.py::test_fear_reset_fires_when_taboo_overlap_high PASSED
tests/monitor/test_monitor_negative.py::test_fear_reset_restores_gamma_neg PASSED
tests/monitor/test_monitor_negative.py::test_fear_reset_does_not_fire_below_threshold PASSED
tests/monitor/test_monitor_negative.py::test_fear_reset_result_keys_always_present PASSED

9 passed in 0.XXs
```

Also verify existing monitor tests still pass:
```
cd /home/mattthomson/workspace/HPM---Learning-Agent && python -m pytest tests/monitor/ -v
```

### 6.5 Commit

```
cd /home/mattthomson/workspace/HPM---Learning-Agent && git add hpm/monitor/structural_law.py hpm/monitor/recombination_strategist.py tests/monitor/test_monitor_negative.py && git commit -m "feat(monitor): add negative_count, taboo_overlap metrics and Fear Reset intervention"
```

---

## Full regression run

After all tasks are committed, run the full test suite to verify no regressions:

```
cd /home/mattthomson/workspace/HPM---Learning-Agent && python -m pytest tests/ -v --tb=short
```

Expected: all new tests pass; all pre-existing tests continue to pass.

---

## Dependency order

Tasks must be executed in order — each task depends on the previous:

1. **Task 1** (TieredStore) — standalone, no dependencies.
2. **Task 2** (AgentConfig) — standalone, no dependencies. Can run in parallel with Task 1.
3. **Task 3** (PatternField) — standalone, no dependencies. Can run in parallel with Tasks 1–2.
4. **Task 4** (Agent integration) — requires Tasks 1, 2, and 3 complete.
5. **Task 5** (ARC scoring) — requires Task 1 complete (uses `query_negative`).
6. **Task 6** (Monitor + Strategist) — requires Tasks 1 and 2 complete; `taboo_overlap` uses `query_negative`.

---

## Spec coverage checklist

- [x] §3 TieredStore: `_tier2_negative`, `query_negative`, `negative_merge`, `query_tier2_negative_all`, updated `end_context` — **Task 1**
- [x] §4 AgentConfig: `gamma_neg`, `neg_conflict_threshold`, `max_tier2_negative` — **Task 2**
- [x] §4 PatternField: `_negative` channel, `broadcast_negative`, `pull_negative` — **Task 3**
- [x] §4 Agent.step(): pull negative (Step A) + broadcast negative (Step B) — **Task 4**
- [x] §5 ensemble_score: subtract negative NLL, backward compatible — **Task 5**
- [x] §6 StructuralLawMonitor: `negative_count`, `taboo_overlap` metrics — **Task 6**
- [x] §6 RecombinationStrategist: Fear Reset on `taboo_overlap > fear_threshold` — **Task 6**
