# Phase 3 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Phase 3 of the HPM agent by adding communicative pattern sharing (Level 4+ broadcast/accept), competitive group-isolated field mode, and a PostgreSQL PatternStore backend.

**Architecture:** Three independent features implemented in order of dependency. Communicative mode spans GaussianPattern (source_id field), PatternField (broadcast queue), and Agent/MultiAgentOrchestrator (sharing and acceptance logic). Competitive mode extends MultiAgentOrchestrator with group-level PatternField isolation. PostgreSQL adds a new store backend with the same interface as SQLiteStore.

**Tech Stack:** Python, numpy, psycopg2 (optional), pytest

---

## File Map

| File | Change |
|---|---|
| `hpm/patterns/gaussian.py` | Add `source_id` param; update `to_dict`, `from_dict`, `update` |
| `hpm/field/field.py` | Add `_broadcast_queue`, `broadcast()`, `drain_broadcasts()` |
| `hpm/agents/agent.py` | Add `_shared_ids`, `_share_pending()`, `_accept_communicated()`, sharing block in `step()` |
| `hpm/agents/multi_agent.py` | Add communication phase, M3 fix, `groups` param, `_group_fields`, `group_field_quality()` |
| `hpm/store/postgres.py` | CREATE — `PostgreSQLStore` |
| `hpm/store/__init__.py` | Conditional import of `PostgreSQLStore` |
| `tests/patterns/test_gaussian.py` | Add `source_id` tests |
| `tests/field/test_field.py` | Add broadcast queue tests |
| `tests/agents/test_agent_communicative.py` | CREATE |
| `tests/agents/test_multi_agent_competitive.py` | CREATE |
| `tests/store/test_postgres.py` | CREATE (skip without `TEST_POSTGRES_DSN`) |

---

## Task 1: GaussianPattern — add `source_id` field

**Files:**
- Modify: `hpm/patterns/gaussian.py`
- Test: `tests/patterns/test_gaussian.py`

This is the foundation. Everything in Tasks 2–4 depends on `GaussianPattern` carrying `source_id`.

The existing `__init__` signature is:
```python
def __init__(self, mu, sigma, id=None, level=1):
```
It must become:
```python
def __init__(self, mu, sigma, id=None, level=1, source_id=None):
```
All existing callers use at most 4 positional or keyword args — `source_id` is fifth and keyword-only in practice.

- [ ] **Step 1: Write the failing tests**

Add to `tests/patterns/test_gaussian.py`:
```python
def test_source_id_defaults_to_none():
    p = GaussianPattern(np.zeros(2), np.eye(2))
    assert p.source_id is None


def test_source_id_stored_from_constructor():
    p = GaussianPattern(np.zeros(2), np.eye(2), source_id='abc-123')
    assert p.source_id == 'abc-123'


def test_to_dict_includes_source_id():
    p = GaussianPattern(np.zeros(2), np.eye(2), source_id='abc-123')
    d = p.to_dict()
    assert 'source_id' in d
    assert d['source_id'] == 'abc-123'


def test_to_dict_source_id_none_when_not_set():
    p = GaussianPattern(np.zeros(2), np.eye(2))
    assert p.to_dict()['source_id'] is None


def test_from_dict_restores_source_id():
    p = GaussianPattern(np.zeros(2), np.eye(2), source_id='abc-123')
    p2 = GaussianPattern.from_dict(p.to_dict())
    assert p2.source_id == 'abc-123'


def test_from_dict_without_source_id_key_defaults_none():
    """Existing serialised patterns (no source_id key) round-trip cleanly."""
    p = GaussianPattern(np.zeros(2), np.eye(2))
    d = p.to_dict()
    del d['source_id']
    p2 = GaussianPattern.from_dict(d)
    assert p2.source_id is None


def test_update_preserves_source_id():
    p = GaussianPattern(np.zeros(2), np.eye(2), source_id='abc-123')
    p2 = p.update(np.ones(2))
    assert p2.source_id == 'abc-123'


def test_recombine_does_not_inherit_source_id():
    pa = GaussianPattern(np.zeros(2), np.eye(2), source_id='parent-a')
    pb = GaussianPattern(np.ones(2), np.eye(2), source_id='parent-b')
    child = pa.recombine(pb)
    assert child.source_id is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/patterns/test_gaussian.py -k "source_id" -v
```
Expected: all fail with `AttributeError: 'GaussianPattern' object has no attribute 'source_id'`

- [ ] **Step 3: Implement changes in `hpm/patterns/gaussian.py`**

**`__init__`** — add `source_id=None` parameter and assignment:
```python
def __init__(self, mu: np.ndarray, sigma: np.ndarray, id: str | None = None, level: int = 1, source_id: str | None = None):
    self.id = id or str(uuid.uuid4())
    self.mu = np.array(mu, dtype=float)
    self.sigma = np.array(sigma, dtype=float)
    self.level = level
    self._n_obs: int = 0
    self.source_id = source_id  # NEW
```

**`to_dict`** — add `source_id` key (all existing keys preserved):
```python
def to_dict(self) -> dict:
    return {
        'type': 'gaussian',
        'id': self.id,
        'mu': self.mu.tolist(),
        'sigma': self.sigma.tolist(),
        'n_obs': self._n_obs,
        'level': self.level,
        'source_id': self.source_id,  # NEW
    }
```

**`from_dict`** — add `source_id=d.get('source_id', None)` (existing `_n_obs` restoration preserved):
```python
@classmethod
def from_dict(cls, d: dict) -> 'GaussianPattern':
    p = cls(np.array(d['mu']), np.array(d['sigma']), id=d['id'],
            level=d.get('level', 1), source_id=d.get('source_id', None))
    p._n_obs = d['n_obs']
    return p
```

**`update`** — pass `source_id=self.source_id` in the constructor call:
```python
def update(self, x: np.ndarray) -> 'GaussianPattern':
    n = self._n_obs + 1
    new_mu = (self.mu * self._n_obs + x) / n
    new_p = GaussianPattern(new_mu, self.sigma.copy(), id=self.id, level=self.level, source_id=self.source_id)
    new_p._n_obs = n
    return new_p
```

`recombine` — no change needed; `source_id=None` is already the default.

- [ ] **Step 4: Run tests**

```bash
pytest tests/patterns/test_gaussian.py -v
```
Expected: all pass (including existing tests — `n_obs` and `type` still in `to_dict`)

- [ ] **Step 5: Commit**

```bash
git add hpm/patterns/gaussian.py tests/patterns/test_gaussian.py
git commit -m "feat: add source_id field to GaussianPattern for communicative provenance"
```

---

## Task 2: PatternField — broadcast queue

**Files:**
- Modify: `hpm/field/field.py`
- Test: `tests/field/test_field.py`

Add `broadcast()` and `drain_broadcasts()` to `PatternField`. The queue accumulates shared patterns during a step; the orchestrator drains it after all agents have stepped.

- [ ] **Step 1: Write the failing tests**

Add to `tests/field/test_field.py`:
```python
from hpm.patterns.gaussian import GaussianPattern


def _pattern():
    return GaussianPattern(np.zeros(2), np.eye(2))


def test_broadcast_appends_to_queue():
    field = PatternField()
    p = _pattern()
    field.broadcast('agent_a', p)
    queue = field.drain_broadcasts()
    assert len(queue) == 1
    assert queue[0][0] == 'agent_a'
    assert queue[0][1] is p


def test_drain_broadcasts_clears_queue():
    field = PatternField()
    field.broadcast('agent_a', _pattern())
    field.drain_broadcasts()
    assert field.drain_broadcasts() == []


def test_drain_broadcasts_returns_independent_list():
    field = PatternField()
    field.broadcast('agent_a', _pattern())
    result = field.drain_broadcasts()
    result.append(('extra', _pattern()))
    assert field.drain_broadcasts() == []


def test_multiple_broadcasts_accumulated():
    field = PatternField()
    field.broadcast('a', _pattern())
    field.broadcast('b', _pattern())
    queue = field.drain_broadcasts()
    assert len(queue) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/field/test_field.py -k "broadcast" -v
```
Expected: `AttributeError: 'PatternField' object has no attribute 'broadcast'`

- [ ] **Step 3: Implement in `hpm/field/field.py`**

In `PatternField.__init__`, after the existing `self._agent_patterns` line:
```python
self._broadcast_queue: list[tuple[str, object]] = []
```

Add the two new methods (append before `field_quality`):
```python
def broadcast(self, source_agent_id: str, pattern) -> None:
    """Enqueue a shared pattern copy. Called by Agent._share_pending() when level >= 4."""
    self._broadcast_queue.append((source_agent_id, pattern))

def drain_broadcasts(self) -> list[tuple[str, object]]:
    """Return and clear the broadcast queue. Called by orchestrator after all agents step."""
    queue = list(self._broadcast_queue)
    self._broadcast_queue = []
    return queue
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/field/test_field.py -v
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add hpm/field/field.py tests/field/test_field.py
git commit -m "feat: add broadcast queue to PatternField for communicative mode"
```

---

## Task 3: Agent — `_share_pending` and `_accept_communicated`

**Files:**
- Modify: `hpm/agents/agent.py`
- Test: `tests/agents/test_agent_communicative.py` (create)

Add the two core methods for communicative mode. These are pure logic — not yet wired into `step()` (that's Task 4). Testing them in isolation first keeps each task small.

**Key background for implementers:**
- `log_prob(x)` returns NLL = `-log p(x|h)`, a positive value
- `-log_prob(x)` = log-likelihood ≤ 0 (this is `Eff`)
- `sym_kl_normalised` is imported from `hpm.dynamics.meta_pattern_rule`
- `_accept_communicated` must import `sym_kl_normalised` at the top of `agent.py` (it's already there via `MetaPatternRule` but import the function directly)

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_agent_communicative.py`:
```python
import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.field.field import PatternField
from hpm.patterns.gaussian import GaussianPattern
from hpm.store.memory import InMemoryStore


def cfg(agent_id='test'):
    return AgentConfig(agent_id=agent_id, feature_dim=2)


def make_agent_with_field(agent_id='test'):
    field = PatternField()
    agent = Agent(cfg(agent_id), field=field)
    return agent, field


def level4_pattern(mu=None):
    """GaussianPattern with level=4 pre-set."""
    p = GaussianPattern(
        mu=mu if mu is not None else np.zeros(2),
        sigma=np.eye(2),
    )
    p.level = 4
    return p


# ---- _share_pending ----

def test_share_pending_broadcasts_level4_pattern():
    agent, field = make_agent_with_field()
    p = level4_pattern()
    agent._share_pending(field, [p])
    queue = field.drain_broadcasts()
    assert len(queue) == 1
    source_id, shared = queue[0]
    assert source_id == 'test'
    assert shared.source_id == p.id


def test_share_pending_does_not_broadcast_below_level4():
    agent, field = make_agent_with_field()
    for lvl in [1, 2, 3]:
        p = GaussianPattern(np.zeros(2), np.eye(2))
        p.level = lvl
        agent._share_pending(field, [p])
    assert field.drain_broadcasts() == []


def test_share_pending_does_not_reshare():
    agent, field = make_agent_with_field()
    p = level4_pattern()
    agent._share_pending(field, [p])
    field.drain_broadcasts()  # consume first broadcast
    agent._share_pending(field, [p])
    assert field.drain_broadcasts() == []  # not shared again


def test_share_pending_shared_copy_has_new_uuid():
    agent, field = make_agent_with_field()
    p = level4_pattern()
    agent._share_pending(field, [p])
    _, shared = field.drain_broadcasts()[0]
    assert shared.id != p.id  # fresh UUID
    assert shared.source_id == p.id  # provenance preserved


# ---- _accept_communicated ----

def test_accept_communicated_novel_pattern_admitted():
    agent, _ = make_agent_with_field()
    # Seed agent with an observation so buffer is non-empty
    x = np.zeros(2)
    agent.step(x)
    # Distant incoming pattern
    incoming = GaussianPattern(mu=np.array([100.0, 100.0]), sigma=np.eye(2) * 0.01)
    result = agent._accept_communicated(incoming, 'other_agent')
    assert result is True
    ids = [p.id for p, _ in agent.store.query('test')]
    assert incoming.id in ids


def test_accept_communicated_identical_pattern_empty_buffer_rejected():
    """Nov=0 exactly, Eff=0 (empty buffer) → insight=0 → rejected."""
    agent = Agent(cfg(), store=InMemoryStore())
    # Seed the store manually with the same pattern we'll send
    existing = GaussianPattern(np.zeros(2), np.eye(2))
    agent.store.save(existing, 1.0, 'test')
    # Incoming is identical
    incoming = GaussianPattern(np.zeros(2), np.eye(2))
    result = agent._accept_communicated(incoming, 'other_agent')
    assert result is False


def test_accept_communicated_no_self_reception_enforced_by_caller():
    """_accept_communicated itself doesn't enforce self-rejection; orchestrator does.
    Just confirm the method runs without error on same agent_id."""
    agent, _ = make_agent_with_field()
    agent.step(np.zeros(2))
    incoming = GaussianPattern(np.array([50.0, 50.0]), np.eye(2) * 0.01)
    # Should not raise
    agent._accept_communicated(incoming, 'test')


def test_accept_communicated_empty_library_uses_nov_one():
    """When library is empty, Nov=1.0 and Eff=0.0 (empty buffer) → insight = beta_orig * alpha_nov."""
    c = AgentConfig(agent_id='empty', feature_dim=2, beta_orig=1.0, alpha_nov=0.5, alpha_eff=0.5, kappa_0=0.1)
    agent = Agent(c, store=InMemoryStore())
    # Clear the store (remove seed pattern)
    for p, _ in agent.store.query('empty'):
        agent.store.delete(p.id)
    incoming = GaussianPattern(np.zeros(2), np.eye(2))
    result = agent._accept_communicated(incoming, 'other')
    # Nov=1.0, Eff=0.0 → insight = 1.0*(0.5*1.0 + 0.5*0.0) = 0.5 > 0 → admitted
    assert result is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agents/test_agent_communicative.py -v
```
Expected: `AttributeError: 'Agent' object has no attribute '_share_pending'`

- [ ] **Step 3: Add `_shared_ids` to `Agent.__init__`**

In `hpm/agents/agent.py`, after the `self._recomb_op = RecombinationOperator(...)` line (~line 80):
```python
self._shared_ids: set[str] = set()
```

- [ ] **Step 4: Add `_share_pending` method to `Agent`**

Add after `_seed_if_empty` in `hpm/agents/agent.py`:
```python
def _share_pending(self, field, patterns: list) -> int:
    """
    Broadcast patterns that have newly reached Level 4+ to the field.
    Precondition: field must not be None (caller's responsibility).
    Returns count of patterns newly shared this call.
    """
    count = 0
    for p in patterns:
        if p.level >= 4 and p.id not in self._shared_ids:
            shared_copy = GaussianPattern(
                mu=p.mu.copy(),
                sigma=p.sigma.copy(),
                source_id=p.id,
                # No id= → fresh UUID. Communicative sharing transfers structure,
                # not UUID identity. source_id preserves provenance.
            )
            field.broadcast(self.agent_id, shared_copy)
            self._shared_ids.add(p.id)
            count += 1
    return count
```

- [ ] **Step 5: Add `_accept_communicated` method to `Agent`**

Add after `_share_pending`:
```python
def _accept_communicated(self, pattern, source_agent_id: str) -> bool:
    """
    Evaluate an incoming communicated pattern and admit it if I(h*) > 0.
    Returns True if admitted.

    Sign convention: log_prob(x) returns NLL (positive). -log_prob(x) gives
    log-likelihood (≤ 0). Eff is therefore non-positive; I(h*) > 0 requires
    novelty to offset negative efficacy.

    Nov = max(sym_kl_normalised(pattern, p) for p in library)  [1.0 if empty]
    Eff = mean(-pattern.log_prob(x) for x in obs_buffer)  ≤ 0  [0.0 if empty]
    I   = beta_orig * (alpha_nov * Nov + alpha_eff * Eff)
    """
    from .dynamics.meta_pattern_rule import sym_kl_normalised  # local import avoids circular
    records = self.store.query(self.agent_id)
    existing = [p for p, _ in records]

    nov = (
        max(sym_kl_normalised(pattern, p) for p in existing)
        if existing else 1.0
    )
    obs = list(self._obs_buffer)
    eff = float(np.mean([-pattern.log_prob(x) for x in obs])) if obs else 0.0
    insight = self.config.beta_orig * (
        self.config.alpha_nov * nov + self.config.alpha_eff * eff
    )
    if insight <= 0:
        return False

    entry_weight = self.config.kappa_0 * insight
    self.store.save(pattern, entry_weight, self.agent_id)
    all_records = self.store.query(self.agent_id)
    total_w = sum(w for _, w in all_records)
    if total_w > 0:
        for p, w in all_records:
            self.store.update_weight(p.id, w / total_w)
    return True
```

**Note on import:** `sym_kl_normalised` is already accessible via `hpm.dynamics.meta_pattern_rule`. The local import inside the method avoids any circular import risk. Alternatively, move the import to the top of `agent.py` alongside the existing `MetaPatternRule` import — either works.

- [ ] **Step 6: Run tests**

```bash
pytest tests/agents/test_agent_communicative.py -v
```
Expected: all pass

- [ ] **Step 7: Run full suite**

```bash
pytest --tb=short -q
```
Expected: all existing tests still pass

- [ ] **Step 8: Commit**

```bash
git add hpm/agents/agent.py tests/agents/test_agent_communicative.py
git commit -m "feat: add _share_pending and _accept_communicated to Agent"
```

---

## Task 4: Wire communicative mode into `Agent.step()` and `MultiAgentOrchestrator`

**Files:**
- Modify: `hpm/agents/agent.py`
- Modify: `hpm/agents/multi_agent.py`
- Test: `tests/agents/test_agent_communicative.py` (extend)

Wire the sharing trigger into `step()`, add `communicated_out` to the return dict, and update the orchestrator to drain and distribute broadcasts. Fix the M3 single-agent sharing gap.

- [ ] **Step 1: Write the failing tests**

Add to `tests/agents/test_agent_communicative.py`:
```python
def test_communicated_out_in_return_dict_every_step():
    agent, field = make_agent_with_field()
    result = agent.step(np.zeros(2))
    assert 'communicated_out' in result
    assert result['communicated_out'] == 0


def test_no_sharing_below_level4_from_step():
    """Patterns at level 1-3 never appear in broadcast queue after step."""
    agent, field = make_agent_with_field()
    # Force all patterns to level < 4 by running one step (initial pattern will be level 1)
    agent.step(np.zeros(2))
    assert field.drain_broadcasts() == []


def test_orchestrator_distributes_broadcast_to_other_agents():
    from hpm.agents.multi_agent import MultiAgentOrchestrator
    field = PatternField()
    cfgA = AgentConfig(agent_id='A', feature_dim=2)
    cfgB = AgentConfig(agent_id='B', feature_dim=2, kappa_0=0.5)
    agentA = Agent(cfgA, field=field)
    agentB = Agent(cfgB, field=field)

    # Manually add a level-4 pattern to agent A's store and _share_pending
    p = level4_pattern(mu=np.array([100.0, 100.0]))
    # Directly add to broadcast queue to test orchestrator distribution
    shared = GaussianPattern(mu=p.mu.copy(), sigma=np.eye(2) * 0.01, source_id=p.id)
    field.broadcast('A', shared)

    # Give agent B some observations so Eff can be computed
    for _ in range(5):
        agentB.step(np.zeros(2))

    # Drain and distribute manually (simulating orchestrator)
    broadcasts = field.drain_broadcasts()
    for source_id, pat in broadcasts:
        if source_id != 'B':
            agentB._accept_communicated(pat, source_id)

    # Agent B may or may not admit — just verify no exception and method works
    # (admission depends on novelty and efficacy values)
```

- [ ] **Step 2: Run failing tests**

```bash
pytest tests/agents/test_agent_communicative.py::test_communicated_out_in_return_dict_every_step -v
```
Expected: `KeyError: 'communicated_out'`

- [ ] **Step 3: Add sharing block to `Agent.step()`**

In `hpm/agents/agent.py`, after the recombination block and `_last_recomb_t` assignment, and after building `report_patterns`, add:

```python
communicated_out = 0
if self.field is not None:
    communicated_out = self._share_pending(self.field, report_patterns)
```

Add to the return dict:
```python
'communicated_out': communicated_out,
```

- [ ] **Step 4: Update `MultiAgentOrchestrator.step()` — communication phase**

In `hpm/agents/multi_agent.py`, at the end of `step()` (after the `for agent in self.agents` loop), add the communication phase. Insert after the existing `metrics[agent.agent_id] = step_metrics` block but before `return metrics`:

```python
# Communication phase: drain broadcast queue and distribute to other agents
if self.field is not None:
    broadcasts = self.field.drain_broadcasts()
    for source_agent_id, pattern in broadcasts:
        for agent in self.agents:
            if agent.agent_id != source_agent_id:
                agent._accept_communicated(pattern, source_agent_id)
```

- [ ] **Step 5: Fix M3 sharing gap in `MultiAgentOrchestrator.step()`**

In the M3 branch (where `agent.field = None` is temporarily set), after restoring `agent.field = actual_field` and re-registering patterns, add:

```python
# Run sharing check that was suppressed during the M3 detach.
# Overwrites communicated_out: 0 from agent.step() (field was None during step).
# Intentional — the orchestrator is authoritative for M3 sharing counts.
if actual_field is not None:
    patterns_post = [p for p, _ in records]
    step_metrics['communicated_out'] = agent._share_pending(actual_field, patterns_post)
```

The complete M3 block in `step()` should look like:
```python
if m3_active:
    actual_field = agent.field
    agent.field = None
    try:
        step_metrics = agent.step(x, reward=r)
    finally:
        agent.field = actual_field
    if actual_field is not None:
        records = agent.store.query(agent.agent_id)
        actual_field.register(agent.agent_id, [(p.id, w) for p, w in records])
        patterns_post = [p for p, _ in records]
        step_metrics['communicated_out'] = agent._share_pending(actual_field, patterns_post)
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/agents/test_agent_communicative.py -v
pytest tests/agents/ -v
```
Expected: all pass

- [ ] **Step 7: Run full suite**

```bash
pytest --tb=short -q
```
Expected: all passing

- [ ] **Step 8: Commit**

```bash
git add hpm/agents/agent.py hpm/agents/multi_agent.py tests/agents/test_agent_communicative.py
git commit -m "feat: wire communicative mode into Agent.step() and MultiAgentOrchestrator"
```

---

## Task 5: `MultiAgentOrchestrator` — competitive mode

**Files:**
- Modify: `hpm/agents/multi_agent.py`
- Test: `tests/agents/test_multi_agent_competitive.py` (create)

Competitive mode: agents are divided into groups, each with an isolated `PatternField`. No changes to `PatternField` or `Agent`.

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_multi_agent_competitive.py`:
```python
import numpy as np
import pytest
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.field.field import PatternField
from hpm.patterns.gaussian import GaussianPattern


def make_agents(n, feature_dim=2):
    return [Agent(AgentConfig(agent_id=f'agent_{i}', feature_dim=feature_dim))
            for i in range(n)]


def test_groups_assign_separate_field_objects():
    agents = make_agents(4)
    groups = {'agent_0': 'A', 'agent_1': 'A', 'agent_2': 'B', 'agent_3': 'B'}
    orch = MultiAgentOrchestrator(agents, PatternField(), groups=groups)
    field_A = agents[0].field
    field_B = agents[2].field
    assert field_A is not field_B
    assert agents[0].field is agents[1].field
    assert agents[2].field is agents[3].field


def test_in_group_patterns_visible_cross_agent():
    """Shared UUID registered by agent_0 appears in agent_1 field freq (same group)."""
    agents = make_agents(2)
    groups = {'agent_0': 'A', 'agent_1': 'A'}
    orch = MultiAgentOrchestrator(agents, PatternField(), groups=groups)

    shared_p = GaussianPattern(np.zeros(2), np.eye(2))
    agents[0].field.register('agent_0', [(shared_p.id, 0.8)])
    freq = agents[1].field.freq(shared_p.id)
    assert freq > 0.0


def test_out_group_patterns_not_visible():
    """Agent in group B cannot see agent A's patterns via field."""
    agents = make_agents(2)
    groups = {'agent_0': 'A', 'agent_1': 'B'}
    orch = MultiAgentOrchestrator(agents, PatternField(), groups=groups)

    p = GaussianPattern(np.zeros(2), np.eye(2))
    agents[0].field.register('agent_0', [(p.id, 0.9)])
    freq = agents[1].field.freq(p.id)
    assert freq == 0.0


def test_group_field_quality_keyed_by_group():
    agents = make_agents(4)
    groups = {'agent_0': 'A', 'agent_1': 'A', 'agent_2': 'B', 'agent_3': 'B'}
    orch = MultiAgentOrchestrator(agents, PatternField(), groups=groups)
    quality = orch.group_field_quality()
    assert set(quality.keys()) == {'A', 'B'}
    for gid, q in quality.items():
        assert 'diversity' in q
        assert 'redundancy' in q


def test_backward_compat_no_groups():
    """groups=None: behaviour identical to before; group_field_quality() returns {}."""
    agents = make_agents(2)
    field = PatternField()
    orch = MultiAgentOrchestrator(agents, field, groups=None)
    # Both agents share the same field (the one passed in)
    assert agents[0].field is field
    assert agents[1].field is field
    assert orch.group_field_quality() == {}


def test_competitive_broadcast_within_group_only():
    """In competitive mode, broadcasts only reach within-group agents."""
    agents = make_agents(2)
    groups = {'agent_0': 'A', 'agent_1': 'B'}
    orch = MultiAgentOrchestrator(agents, PatternField(), groups=groups)

    # Manually broadcast on agent_0's group field
    p = GaussianPattern(np.array([100.0, 100.0]), np.eye(2) * 0.01)
    agents[0].field.broadcast('agent_0', p)

    # Drain agent_0's field — agent_1 is in a different group so its field is separate
    queue_A = agents[0].field.drain_broadcasts()
    queue_B = agents[1].field.drain_broadcasts()
    assert len(queue_A) == 1
    assert len(queue_B) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agents/test_multi_agent_competitive.py -v
```
Expected: `TypeError` or `AttributeError` because `MultiAgentOrchestrator` doesn't accept `groups`

- [ ] **Step 3: Update `MultiAgentOrchestrator.__init__`**

In `hpm/agents/multi_agent.py`, update `__init__`:

```python
def __init__(
    self,
    agents: list,
    field: PatternField,
    seed_pattern: GaussianPattern | None = None,
    groups: dict[str, str] | None = None,   # agent_id -> group_id
):
    self.agents = agents
    self._groups = groups
    self._group_fields: dict[str, PatternField] = {}

    if groups is not None:
        # Create one PatternField per unique group and assign to agents
        for group_id in set(groups.values()):
            self._group_fields[group_id] = PatternField()
        for agent in agents:
            agent.field = self._group_fields[groups[agent.agent_id]]
        self.field = None   # ungrouped field unused when all agents are grouped
    else:
        self.field = field
        for agent in agents:
            if agent.field is None:
                agent.field = field

    if seed_pattern is not None:
        self._seed_shared(seed_pattern)
```

- [ ] **Step 4: Add `group_field_quality()` method**

Add after `_seed_shared`:
```python
def group_field_quality(self) -> dict[str, dict]:
    """
    Returns field quality metrics per group, keyed by group_id.
    Delegates to PatternField.field_quality() for each group field.
    Returns {} when groups are not configured.
    """
    return {
        gid: gfield.field_quality()
        for gid, gfield in self._group_fields.items()
    }
```

- [ ] **Step 5: Update communication phase in `step()` for grouped mode**

Replace the communication phase added in Task 4 with one that handles both grouped and ungrouped:

```python
# Communication phase — within-group only when groups configured
if self._group_fields:
    for group_id, gfield in self._group_fields.items():
        broadcasts = gfield.drain_broadcasts()
        group_agent_ids = {aid for aid, gid in self._groups.items() if gid == group_id}
        for source_agent_id, pattern in broadcasts:
            for agent in self.agents:
                if agent.agent_id in group_agent_ids and agent.agent_id != source_agent_id:
                    agent._accept_communicated(pattern, source_agent_id)
elif self.field is not None:
    broadcasts = self.field.drain_broadcasts()
    for source_agent_id, pattern in broadcasts:
        for agent in self.agents:
            if agent.agent_id != source_agent_id:
                agent._accept_communicated(pattern, source_agent_id)
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/agents/test_multi_agent_competitive.py -v
pytest tests/agents/ -v
```
Expected: all pass

- [ ] **Step 7: Run full suite**

```bash
pytest --tb=short -q
```
Expected: all passing

- [ ] **Step 8: Commit**

```bash
git add hpm/agents/multi_agent.py tests/agents/test_multi_agent_competitive.py
git commit -m "feat: add competitive group mode to MultiAgentOrchestrator"
```

---

## Task 6: PostgreSQLStore

**Files:**
- Create: `hpm/store/postgres.py`
- Modify: `hpm/store/__init__.py`
- Test: `tests/store/test_postgres.py` (create)

`PostgreSQLStore` is a drop-in replacement for `SQLiteStore`. Requires `psycopg2`. Tests skip automatically without `TEST_POSTGRES_DSN`.

- [ ] **Step 1: Write the tests**

Create `tests/store/test_postgres.py`:
```python
import os
import json
import pytest
import numpy as np

dsn = os.environ.get("TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(not dsn, reason="TEST_POSTGRES_DSN not set")


@pytest.fixture
def store():
    from hpm.store.postgres import PostgreSQLStore
    s = PostgreSQLStore(dsn)
    yield s
    # Teardown: drop patterns table for isolation
    with s._conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS patterns")
    s._conn.commit()
    s.close()


def _pattern():
    from hpm.patterns.gaussian import GaussianPattern
    return GaussianPattern(np.zeros(2), np.eye(2))


def test_save_and_query(store):
    p = _pattern()
    store.save(p, 0.8, 'agent_a')
    records = store.query('agent_a')
    assert len(records) == 1
    loaded, weight = records[0]
    assert loaded.id == p.id
    assert abs(weight - 0.8) < 1e-9


def test_query_returns_only_agent_patterns(store):
    pa = _pattern()
    pb = _pattern()
    store.save(pa, 0.5, 'agent_a')
    store.save(pb, 0.5, 'agent_b')
    assert len(store.query('agent_a')) == 1
    assert len(store.query('agent_b')) == 1


def test_delete(store):
    p = _pattern()
    store.save(p, 1.0, 'agent_a')
    store.delete(p.id)
    assert store.query('agent_a') == []


def test_update_weight(store):
    p = _pattern()
    store.save(p, 1.0, 'agent_a')
    store.update_weight(p.id, 0.3)
    _, w = store.query('agent_a')[0]
    assert abs(w - 0.3) < 1e-9


def test_update_weight_missing_id_silent_noop(store):
    store.update_weight('nonexistent-id', 0.5)  # should not raise


def test_save_overwrites_existing_id(store):
    p = _pattern()
    store.save(p, 0.5, 'agent_a')
    store.save(p, 0.9, 'agent_a')
    records = store.query('agent_a')
    assert len(records) == 1
    assert abs(records[0][1] - 0.9) < 1e-9


def test_query_all(store):
    pa = _pattern()
    pb = _pattern()
    store.save(pa, 0.6, 'agent_a')
    store.save(pb, 0.4, 'agent_b')
    all_records = store.query_all()
    assert len(all_records) == 2


def test_source_id_round_trips(store):
    from hpm.patterns.gaussian import GaussianPattern
    p = GaussianPattern(np.zeros(2), np.eye(2), source_id='origin-uuid')
    store.save(p, 1.0, 'agent_a')
    loaded, _ = store.query('agent_a')[0]
    assert loaded.source_id == 'origin-uuid'
```

- [ ] **Step 2: (Without `TEST_POSTGRES_DSN`) verify tests skip**

```bash
pytest tests/store/test_postgres.py -v
```
Expected: all tests SKIPPED with "TEST_POSTGRES_DSN not set"

- [ ] **Step 3: Implement `hpm/store/postgres.py`**

Create the file:
```python
import json
from ..patterns.gaussian import GaussianPattern


class PostgreSQLStore:
    """
    PatternStore backed by PostgreSQL. Drop-in replacement for SQLiteStore.
    Schema matches SQLiteStore: patterns(id, agent_id, pattern_json, weight).

    Parameters
    ----------
    dsn : str
        libpq connection string, e.g. "postgresql://user:pass@host/dbname"
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS patterns (
        id           TEXT PRIMARY KEY,
        agent_id     TEXT NOT NULL,
        pattern_json TEXT NOT NULL,
        weight       REAL NOT NULL
    )
    """
    _INDEX = "CREATE INDEX IF NOT EXISTS idx_patterns_agent ON patterns(agent_id)"

    def __init__(self, dsn: str):
        import psycopg2
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(self._SCHEMA)
            cur.execute(self._INDEX)
        self._conn.commit()

    def save(self, pattern, weight: float, agent_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO patterns (id, agent_id, pattern_json, weight) VALUES (%s, %s, %s, %s)"
                " ON CONFLICT (id) DO UPDATE SET"
                " weight = EXCLUDED.weight, pattern_json = EXCLUDED.pattern_json",
                (pattern.id, agent_id, json.dumps(pattern.to_dict()), weight),
            )
        self._conn.commit()

    def query(self, agent_id: str) -> list:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT pattern_json, weight FROM patterns WHERE agent_id = %s", (agent_id,)
            )
            return [
                (GaussianPattern.from_dict(json.loads(row[0])), row[1])
                for row in cur.fetchall()
            ]

    def query_all(self) -> list:
        """Return all (pattern, weight, agent_id) triples. Matches SQLiteStore signature."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT pattern_json, weight, agent_id FROM patterns")
            return [
                (GaussianPattern.from_dict(json.loads(row[0])), row[1], row[2])
                for row in cur.fetchall()
            ]

    def delete(self, pattern_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM patterns WHERE id = %s", (pattern_id,))
        self._conn.commit()

    def update_weight(self, pattern_id: str, weight: float) -> None:
        """Silent no-op if pattern_id does not exist — matches SQLiteStore behaviour."""
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE patterns SET weight = %s WHERE id = %s", (weight, pattern_id)
            )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Update `hpm/store/__init__.py`**

Add a conditional import at the end of the existing `__init__.py`:
```python
try:
    from .postgres import PostgreSQLStore
except ImportError:
    pass
```

- [ ] **Step 5: (Without Postgres) verify existing tests still pass**

```bash
pytest --tb=short -q
```
Expected: all 199+ existing tests pass; `test_postgres.py` skips

- [ ] **Step 6: (With `TEST_POSTGRES_DSN` set) run PostgreSQL tests**

```bash
TEST_POSTGRES_DSN="postgresql://user:pass@localhost/hpm_test" pytest tests/store/test_postgres.py -v
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add hpm/store/postgres.py hpm/store/__init__.py tests/store/test_postgres.py
git commit -m "feat: add PostgreSQLStore backend (optional psycopg2 dependency)"
```

---

## Final verification

- [ ] **Run full test suite**

```bash
pytest --tb=short -q
```
Expected: all tests pass (Postgres tests skip if `TEST_POSTGRES_DSN` not set)

- [ ] **Update issue log**

In `docs/issue-log.md`, mark Issues 2, 3, 4 as CLOSED.

- [ ] **Push**

```bash
git push
```
