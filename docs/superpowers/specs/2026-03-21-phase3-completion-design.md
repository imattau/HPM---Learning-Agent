# Phase 3 Completion — Design Specification

**Date:** 2026-03-21

## Goal

Complete Phase 3 of the HPM agent implementation by adding the two remaining field interaction modes (communicative and competitive) and a PostgreSQL PatternStore backend. All three features are backward-compatible and opt-in.

---

## Background

Phase 3 per the original design spec (`2026-03-20-hpm-agent-design.md §8`):

> Multi-agent, pattern field: SocialEvaluator + PatternField (observational mode first), multi-agent orchestrator with shared PatternStore (PostgreSQL for concurrent writes), metrics for §9.5, communicative and competitive modes.

Currently implemented: observational mode, SocialEvaluator, PatternField, MultiAgentOrchestrator, §9.5 social_field_convergence metric.

Missing: communicative mode, competitive mode, PostgreSQL backend.

---

## Feature 1: Communicative Mode

### HPM Motivation

The PatternField currently implements the **Registration** tier: every step, each agent pushes its pattern UUIDs and weights to the shared field, enabling population-level frequency signals (`freq_i`). This is observational learning — agents influence each other indirectly through weight dynamics.

**Sharing** is a second, distinct tier: an agent externalises the structural content of a pattern (μ, Σ) to the field when that pattern has achieved sufficient stability to be worth broadcasting. Per the HPM framework's substrate-shifting concept, Level 4+ patterns represent "published structural laws" — decoupled from surface noise, internally coherent, high density. Sharing them is the functional equivalent of placing a private cognitive model into a public, reusable substrate.

**Caveat on level classification:** `pattern.level` is assigned by `HPMLevelClassifier` inside `Agent.step()` and is agent-local — it depends on each agent's density computation, field frequency, and evaluator state. Two agents holding the same `(mu, Σ)` may classify the pattern at different levels. The sharing trigger `level >= 4` therefore fires independently per agent; a pattern may be shared by one agent but not by another holding the same parameters. This is consistent with the HPM framing that structural achievement is evaluated from the agent's own epistemic standpoint.

### Two-Tier Model

| Tier | Cadence | What is broadcast | Trigger |
|---|---|---|---|
| Registration | Every step | UUID + weight only | Automatic |
| Sharing | One-time per pattern | μ, Σ (full GaussianPattern copy) | `pattern.level >= 4` for the first time |

### Components

#### `GaussianPattern` — `source_id` field

Add `source_id: str | None = None` as the fifth parameter to `GaussianPattern.__init__`, after `level`. The existing body is preserved exactly; only the signature and the `self.source_id` assignment are new:

```python
def __init__(self, mu: np.ndarray, sigma: np.ndarray, id: str | None = None, level: int = 1, source_id: str | None = None):
    self.id = id or str(uuid.uuid4())
    self.mu = np.array(mu, dtype=float)    # existing — preserved
    self.sigma = np.array(sigma, dtype=float)  # existing — preserved
    self.level = level
    self._n_obs: int = 0                   # existing — preserved
    self.source_id = source_id             # NEW
```

All existing positional callers pass at most `mu`, `sigma`, `id`, `level` — adding `source_id` as fifth keyword parameter does not break any existing call.

**`to_dict()`** — add `source_id` alongside existing keys (all existing keys preserved):

```python
def to_dict(self) -> dict:
    return {
        'type': 'gaussian',       # existing
        'id': self.id,            # existing
        'mu': self.mu.tolist(),   # existing
        'sigma': self.sigma.tolist(),  # existing
        'n_obs': self._n_obs,     # existing
        'level': self.level,      # existing
        'source_id': self.source_id,  # NEW — None when not set
    }
```

**`from_dict()`** — add `source_id=d.get('source_id', None)` alongside existing restoration (existing `_n_obs` assignment preserved):

```python
@classmethod
def from_dict(cls, d: dict) -> 'GaussianPattern':
    p = cls(np.array(d['mu']), np.array(d['sigma']), id=d['id'],
            level=d.get('level', 1), source_id=d.get('source_id', None))  # source_id NEW
    p._n_obs = d['n_obs']   # existing
    return p
```

**`update()`** — pass `source_id=self.source_id` in the constructor call (one keyword added):

```python
def update(self, x: np.ndarray) -> 'GaussianPattern':
    n = self._n_obs + 1
    new_mu = (self.mu * self._n_obs + x) / n
    new_p = GaussianPattern(new_mu, self.sigma.copy(), id=self.id, level=self.level, source_id=self.source_id)
    new_p._n_obs = n
    return new_p
```

**`recombine()`** — the recombined child does not inherit `source_id` from either parent (it is a novel pattern, not a communicated copy); `source_id=None` is the default and requires no explicit change.

**Backward compatibility:** `from_dict` uses `.get('source_id', None)`, so existing serialised patterns without the key round-trip without error. `to_dict()` adds one new key (`source_id`) but preserves all existing keys — `from_dict` in existing store backends reads `d['n_obs']` correctly since `n_obs` is still emitted.

#### `PatternField` — broadcast queue

Add a per-step broadcast queue to `PatternField`:

```python
self._broadcast_queue: list[tuple[str, GaussianPattern]] = []
```

New methods:

```python
def broadcast(self, source_agent_id: str, pattern: GaussianPattern) -> None:
    """Enqueue a shared pattern copy. Called by Agent._share_pending() when level >= 4."""
    self._broadcast_queue.append((source_agent_id, pattern))

def drain_broadcasts(self) -> list[tuple[str, GaussianPattern]]:
    """Return and clear the broadcast queue. Called by orchestrator after all agents step."""
    queue = list(self._broadcast_queue)
    self._broadcast_queue = []
    return queue
```

The queue is cleared by the orchestrator each step — it is not a persistent store.

#### `Agent` — sharing and acceptance

**New state** (in `__init__`, after `_recomb_op`):

```python
self._shared_ids: set[str] = set()
```

**New method `_share_pending(field, patterns)`** — checks the given patterns and broadcasts newly Level 4+ ones. Extracted as a method (rather than inline in `step()`) so the orchestrator can call it after restoring the field during M3 single-agent mode.

**Precondition:** `field` must not be `None`. Callers are responsible for the `field is not None` guard before calling. The orchestrator's M3 branch wraps its call in `if actual_field is not None:`.

```python
def _share_pending(self, field, patterns: list) -> int:
    """
    Broadcast patterns that have newly reached Level 4+ to the field.
    field must not be None (caller's responsibility).
    Returns count of patterns newly shared this call.
    """
    count = 0
    for p in patterns:
        if p.level >= 4 and p.id not in self._shared_ids:
            shared_copy = GaussianPattern(
                mu=p.mu.copy(),
                sigma=p.sigma.copy(),
                source_id=p.id,
            )
            field.broadcast(self.agent_id, shared_copy)
            self._shared_ids.add(p.id)
            count += 1
    return count
```

**Sharing in `step()`** — after the recombination block, before building the return dict. Uses `report_patterns` (which includes any recombined pattern) so that a Level 4 recombined pattern can also be shared in the same step it is accepted:

```python
communicated_out = 0
if self.field is not None:
    communicated_out = self._share_pending(self.field, report_patterns)
```

**`_accept_communicated(pattern, source_agent_id)`** — evaluates and admits an incoming communicated pattern:

```python
def _accept_communicated(self, pattern: GaussianPattern, source_agent_id: str) -> bool:
    """
    Evaluate an incoming communicated pattern and admit it if I(h*) > 0.
    Returns True if admitted.

    Novelty is measured against the receiving agent's full library (max KL across
    all existing patterns), replacing the two-parent formula used in recombination.
    This is appropriate here because there is no second parent — the incoming
    pattern is evaluated on its own structural merit relative to what the agent
    already knows.

    Empty-library edge case: if the receiving agent's library is empty, Nov = 1.0
    (maximally novel) and admission depends on efficacy alone. This is consistent
    with the cold-start behaviour in RecombinationOperator (empty obs_buffer → Eff=0).

    Eff uses the agent's recent observation buffer (same as recombination), not a
    separate probe set. The two mechanisms therefore share the same kappa_0 scale,
    but numerical ranges may differ slightly because recombination novelty is
    relative to two specific parents while communicative novelty is relative to
    the full library. The shared kappa_0 default (0.1) is appropriate for both.

    Insight scoring:
      Nov = max(sym_kl_normalised(pattern, p) for p in library)  [1.0 if library empty]
      Eff = mean(-pattern.log_prob(x) for x in obs_buffer)       [0.0 if buffer empty]
      I   = beta_orig * (alpha_nov * Nov + alpha_eff * Eff)
    Entry weight = kappa_0 * I, followed by global renormalisation.
    """
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

`sym_kl_normalised` is imported from `hpm.dynamics.meta_pattern_rule`.

**Return dict addition:**

```python
'communicated_out': communicated_out,
```

#### `MultiAgentOrchestrator` — communication phase

**M3 single-agent sharing gap:** When `m3_active`, `agent.field` is temporarily set to `None` during `agent.step()`. The sharing block inside `step()` checks `if self.field is not None` and will not fire. To preserve sharing in the single-agent case, the orchestrator's M3 branch explicitly calls `_share_pending()` after restoring the field:

```python
if m3_active:
    actual_field = agent.field
    agent.field = None
    try:
        step_metrics = agent.step(x, reward=r)
    finally:
        agent.field = actual_field
    # Re-register patterns
    if actual_field is not None:
        records = agent.store.query(agent.agent_id)
        actual_field.register(agent.agent_id, [(p.id, w) for p, w in records])
        # Run sharing check that was suppressed during the M3 detach.
        # Overwrites communicated_out: 0 from agent.step() (field was None during step).
        # Intentional — the orchestrator is authoritative for M3 sharing counts.
        patterns = [p for p, _ in records]
        step_metrics['communicated_out'] = agent._share_pending(actual_field, patterns)
```

**Communication phase** — after all agents step (including M3 path), drain and distribute:

```python
if self.field is not None:
    broadcasts = self.field.drain_broadcasts()
    for source_agent_id, pattern in broadcasts:
        for agent in self.agents:
            if agent.agent_id != source_agent_id:
                agent._accept_communicated(pattern, source_agent_id)
```

For competitive mode, `self.field` is `None` when all agents are grouped (see Feature 2). In that case, each group field's broadcasts are drained separately (see competitive mode section).

---

## Feature 2: Competitive Mode

### HPM Motivation

Competitive mode implements HPM §9.5: agents are divided into groups with divergent evaluator parameters (`beta_aff`, `gamma_soc`). Each group has its own PatternField, creating natural in-group amplification — agents in the same group only see in-group pattern frequencies. The shared observation stream provides the competitive pressure; the divergent parameters produce different learning trajectories. This tests whether groups converge to distinct structural laws.

### Design

No changes to `PatternField`. Each group gets its own instance, providing implicit in-group frequency isolation.

#### `MultiAgentOrchestrator` — group support

New optional constructor parameter:

```python
def __init__(
    self,
    agents: list,
    field: PatternField,
    seed_pattern: GaussianPattern | None = None,
    groups: dict[str, str] | None = None,   # agent_id -> group_id
):
```

When `groups` is provided:

1. Build `_group_fields: dict[str, PatternField]` — one per unique group ID: `{gid: PatternField() for gid in set(groups.values())}`
2. Reassign each agent's `field` to its group's field: `agent.field = _group_fields[groups[agent.agent_id]]`
3. Set `self.field = None` — when all agents are grouped, the constructor's `field` argument is unused and orphaned. Callers should pass `field=PatternField()` as a placeholder; the grouped fields are created internally. Setting `self.field = None` prevents the orchestrator from accidentally draining the wrong queue.
4. `self._groups = groups`
5. `self._group_fields = _group_fields`

When `groups=None`, `self._groups = None`, `self._group_fields = {}`, and all behaviour is identical to today.

**Communication phase with groups** — each group field is drained independently, so broadcasts only reach within-group agents:

```python
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

**New method `group_field_quality()`:**

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

**Return dict from `orchestrator.step()`:** Unchanged — still a flat `dict[agent_id -> step_metrics]`, exactly as today. `group_field_quality()` is a **separate method** that callers invoke directly when they need group-level metrics. No change to the step return structure preserves full backward compatibility.

### Backward Compatibility

`groups=None` (default) leaves existing behaviour entirely unchanged — `self.field` remains the passed-in field, `self._group_fields = {}`, `group_field_quality()` returns `{}`.

---

## Feature 3: PostgreSQL Backend

### Design

`PostgreSQLStore` implements the `PatternStore` protocol identically to `SQLiteStore`. Drop-in replacement — no caller changes required.

#### Schema

Same as SQLite:

```sql
CREATE TABLE IF NOT EXISTS patterns (
    id           TEXT PRIMARY KEY,
    agent_id     TEXT NOT NULL,
    weight       REAL NOT NULL,
    pattern_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_patterns_agent ON patterns(agent_id);
```

#### `hpm/store/postgres.py`

```python
import json
import psycopg2
from ..patterns.gaussian import GaussianPattern


class PostgreSQLStore:
    """
    PatternStore backed by PostgreSQL. Drop-in replacement for SQLiteStore.

    Parameters
    ----------
    dsn : str
        libpq connection string, e.g. "postgresql://user:pass@host/dbname"
    """

    def __init__(self, dsn: str):
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id           TEXT PRIMARY KEY,
                    agent_id     TEXT NOT NULL,
                    weight       REAL NOT NULL,
                    pattern_json TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_agent ON patterns(agent_id)
            """)
        self._conn.commit()

    def save(self, pattern: GaussianPattern, weight: float, agent_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO patterns (id, agent_id, weight, pattern_json) VALUES (%s, %s, %s, %s)"
                " ON CONFLICT (id) DO UPDATE SET weight = EXCLUDED.weight, pattern_json = EXCLUDED.pattern_json",
                (pattern.id, agent_id, weight, json.dumps(pattern.to_dict())),
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

    def delete(self, pattern_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM patterns WHERE id = %s", (pattern_id,))
        self._conn.commit()

    def update_weight(self, pattern_id: str, weight: float) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE patterns SET weight = %s WHERE id = %s", (weight, pattern_id)
            )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
```

**Transaction semantics:** `autocommit = False` with an explicit `self._conn.commit()` after every mutating method. Each mutating call is a single atomic transaction. On error, psycopg2 automatically rolls back the implicit transaction; the caller is responsible for retrying if desired. This matches the per-call durability guarantee of `SQLiteStore`.

#### `hpm/store/__init__.py`

`PostgreSQLStore` is re-exported only if `psycopg2` is importable:

```python
try:
    from .postgres import PostgreSQLStore
except ImportError:
    pass
```

`__all__` is not redefined — downstream code that catches `ImportError` or uses `hasattr(hpm.store, 'PostgreSQLStore')` handles the optional dependency cleanly.

#### Tests — `tests/store/test_postgres.py`

All tests skip if `TEST_POSTGRES_DSN` environment variable is not set:

```python
import os, pytest
dsn = os.environ.get("TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(not dsn, reason="TEST_POSTGRES_DSN not set")
```

Same test suite as `test_sqlite.py` — save, query, delete, update_weight, cross-agent isolation, weight persistence. A `store` fixture creates a fresh `PostgreSQLStore(dsn)` and drops the `patterns` table in teardown to keep tests isolated.

---

## Components

| File | Change |
|---|---|
| `hpm/patterns/gaussian.py` | ADD `source_id` param (5th, default None); update `to_dict`, `from_dict` (`.get()`), `update()` |
| `hpm/field/field.py` | ADD `_broadcast_queue`, `broadcast()`, `drain_broadcasts()` |
| `hpm/agents/agent.py` | ADD `_shared_ids`, `_share_pending()`, `_accept_communicated()`, sharing + acceptance in `step()`, `communicated_out` in return dict |
| `hpm/agents/multi_agent.py` | ADD communication phase, M3 sharing fix, `groups` param, `_group_fields`, `group_field_quality()`, updated return dict |
| `hpm/store/postgres.py` | CREATE `PostgreSQLStore` |
| `hpm/store/__init__.py` | MODIFY conditional import of `PostgreSQLStore` |
| `tests/store/test_postgres.py` | CREATE (skip without `TEST_POSTGRES_DSN`) |
| `tests/field/test_field.py` | ADD tests for `broadcast()`, `drain_broadcasts()` |
| `tests/agents/test_agent_communicative.py` | CREATE |
| `tests/agents/test_multi_agent_competitive.py` | CREATE |

---

## Testing

### PatternField broadcast (`tests/field/test_field.py` additions)

- `test_broadcast_appends_to_queue` — after `broadcast()`, queue has one entry
- `test_drain_broadcasts_clears_queue` — after drain, queue is empty
- `test_drain_returns_copy_of_queue` — returned list is independent of internal state

### Communicative mode (`tests/agents/test_agent_communicative.py`)

- `test_no_sharing_below_level4` — patterns at Level 1-3 never appear in broadcast queue
- `test_shares_on_first_level4_promotion` — pattern promoted to Level 4 appears in queue
- `test_no_resharing` — same pattern does not appear in queue on subsequent steps
- `test_accept_novel_pattern_admitted` — novel incoming pattern with positive insight admitted to store
- `test_accept_redundant_pattern_rejected` — incoming pattern identical to an existing pattern, obs_buffer empty (so Eff=0) and Nov≈0 → insight ≈ 0 → rejected. Note: with a non-empty buffer Eff < 0 (log-likelihood is negative) can offset small positive Nov; the test uses an empty buffer to isolate the novelty gate cleanly.
- `test_accept_empty_library_uses_nov_one` — when receiving agent library is empty, Nov=1.0 and admission depends on efficacy
- `test_no_self_reception` — orchestrator loop skips `source_agent_id == agent.agent_id`
- `test_communicated_out_in_return_dict` — key present every step, value is 0 when no sharing fires
- `test_m3_sharing_fires_after_field_restore` — single-agent orchestrator correctly runs `_share_pending` after M3 detach

### Competitive mode (`tests/agents/test_multi_agent_competitive.py`)

- `test_groups_assign_separate_fields` — agents in different groups have different field objects
- `test_in_group_patterns_visible_cross_agent` — shared UUID registered by agent A appears in agent B's field freq (same group)
- `test_out_group_patterns_not_visible` — agent C (different group) sees zero freq for agent A's UUID
- `test_group_field_quality_keyed_by_group` — `group_field_quality()` keys match configured group IDs
- `test_backward_compat_no_groups` — `groups=None` produces same behaviour as before; `group_field_quality()` returns `{}`

### PostgreSQL (`tests/store/test_postgres.py`)

- Same suite as `test_sqlite.py` — save, query, delete, update_weight, cross-agent isolation, weight persistence
- All skip if `TEST_POSTGRES_DSN` not set
- Teardown drops `patterns` table to isolate tests

---

## Backward Compatibility

- `source_id=None` default and `.get('source_id', None)` in `from_dict` means all existing serialised patterns round-trip without change. No existing test asserts exact key sets of `to_dict()` output.
- All new `MultiAgentOrchestrator` parameters default to `None` / disabled
- `psycopg2` is an optional dependency — `import hpm.store` succeeds without it; `PostgreSQLStore` is simply absent from the namespace
- Existing 199 tests pass without change
