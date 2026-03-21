# Phase 5: Substrate Bridge Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `SubstrateBridgeAgent` — a cadence-gated post-step processor that anchors GaussianPattern weights to external symbolic systems via `field_frequency()`, preventing echo-chamber effects by boosting externally-grounded patterns and penalising ungrounded ones when `StructuralLawMonitor` reports high redundancy.

**Architecture:** Standalone `SubstrateBridgeAgent` class in `hpm/substrate/bridge.py`, composed into `MultiAgentOrchestrator` as optional `bridge=None` kwarg. Holds a shared `SQLiteStore` reference and an `ExternalSubstrate`. Runs every `T_substrate` steps: query Level 3+ patterns, compute `field_frequency()` (with in-memory cache), apply two-pass weight adjustment (boost then echo-chamber penalty), renormalise per agent.

**Tech Stack:** Python stdlib (`numpy`). Reads `ExternalSubstrate` protocol from `hpm/substrate/base.py`. Writes weights via `SQLiteStore.update_weight()`. No new dependencies.

---

## File Structure

```
hpm/substrate/bridge.py                  # new: SubstrateBridgeAgent class
tests/substrate/test_bridge.py           # new: unit + integration tests
hpm/substrate/__init__.py                # existing: add SubstrateBridgeAgent export
hpm/agents/multi_agent.py               # existing: add bridge=None kwarg + call
```

**Key existing files to read before starting:**
- `hpm/substrate/base.py` — `ExternalSubstrate` protocol and `hash_vectorise()`
- `hpm/store/sqlite.py` — `SQLiteStore.query_all()`, `query()`, `update_weight()` signatures
- `hpm/agents/multi_agent.py` — where `strategist.step()` is called; `bridge.step()` goes right after
- `hpm/monitor/structural_law.py` — follow this pattern for how a Phase 5 component is structured

---

### Task 1: SubstrateBridgeAgent class and unit tests

**Files:**
- Create: `hpm/substrate/bridge.py`
- Create: `tests/substrate/test_bridge.py`

- [ ] **Step 1: Write all unit tests**

Create `tests/substrate/test_bridge.py`:

```python
import numpy as np
import pytest
from hpm.store.sqlite import SQLiteStore
from hpm.patterns.gaussian import GaussianPattern
from hpm.substrate.bridge import SubstrateBridgeAgent


class StubSubstrate:
    """Deterministic stub — no HTTP calls."""
    def __init__(self, freq=0.5):
        self._freq = freq
        self.call_count = 0

    def fetch(self, query):
        return []

    def field_frequency(self, pattern):
        self.call_count += 1
        return self._freq

    def stream(self):
        return iter([])


def _make_pattern(level: int, dim: int = 4) -> GaussianPattern:
    p = GaussianPattern(mu=np.random.randn(dim), sigma=np.eye(dim))
    p.level = level
    return p


def _seed(store, agent_id, levels, weight=0.25):
    patterns = []
    for lvl in levels:
        p = _make_pattern(lvl)
        store.save(p, weight, agent_id)
        patterns.append(p)
    return patterns


def _make_bridge(substrate, store, **kwargs):
    return SubstrateBridgeAgent(substrate, store, **kwargs)


# ---- Cadence gate ----

def test_no_op_on_non_cadence_step(tmp_path):
    """Returns {} and makes no weight changes on non-cadence steps."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    patterns = _seed(store, "a1", [3], weight=0.5)
    sub = StubSubstrate(freq=0.8)
    bridge = _make_bridge(sub, store, T_substrate=10)

    result = bridge.step(step_t=1, field_quality={})

    assert result == {}
    assert sub.call_count == 0
    records = store.query("a1")
    assert records[0][1] == pytest.approx(0.5)  # weight unchanged


def test_runs_on_cadence_step(tmp_path):
    """Returns bridge_report dict on cadence step."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3])
    bridge = _make_bridge(StubSubstrate(), store, T_substrate=1)

    result = bridge.step(step_t=1, field_quality={})

    assert isinstance(result, dict)
    assert "patterns_checked" in result


# ---- Standard boost ----

def test_standard_boost_applied(tmp_path):
    """Level 3 pattern with f_freq=0.5, alpha=0.1 → weight multiplied by 1.05."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3], weight=1.0)
    bridge = _make_bridge(StubSubstrate(freq=0.5), store, T_substrate=1, alpha=0.1)

    bridge.step(step_t=1, field_quality={})

    records = store.query("a1")
    # After boost: 1.0 * (1 + 0.1 * 0.5) = 1.05; after normalise with one pattern: 1.0
    # Weight is renormalised to 1.0 (single pattern), but boost was applied
    assert records[0][1] == pytest.approx(1.0)  # single pattern normalises to 1.0


def test_standard_boost_relative(tmp_path):
    """Higher f_freq pattern gets relatively more weight after normalisation."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    p_high = _make_pattern(3)
    p_low = _make_pattern(3)
    store.save(p_high, 0.5, "a1")
    store.save(p_low, 0.5, "a1")

    sub = StubSubstrate()
    sub.field_frequency = lambda p: 0.9 if p.id == p_high.id else 0.1
    bridge = _make_bridge(sub, store, T_substrate=1, alpha=0.2)

    bridge.step(step_t=1, field_quality={})

    recs = {p.id: w for p, w in store.query("a1")}
    assert recs[p_high.id] > recs[p_low.id]


def test_zero_freq_no_boost(tmp_path):
    """f_freq=0.0 → weight multiplier is 1.0 (no boost before normalisation)."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    p1 = _make_pattern(3)
    p2 = _make_pattern(3)
    store.save(p1, 0.5, "a1")
    store.save(p2, 0.5, "a1")

    sub = StubSubstrate()
    sub.field_frequency = lambda p: 0.0
    bridge = _make_bridge(sub, store, T_substrate=1, alpha=0.2)

    bridge.step(step_t=1, field_quality={})

    recs = {p.id: w for p, w in store.query("a1")}
    # Both patterns got zero boost → weights stay equal after normalisation
    assert recs[p1.id] == pytest.approx(recs[p2.id], abs=1e-6)


# ---- Level filtering ----

def test_below_min_level_skipped(tmp_path):
    """Level 2 patterns not queried or updated."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [2], weight=0.4)
    sub = StubSubstrate(freq=0.9)
    bridge = _make_bridge(sub, store, T_substrate=1, min_bridge_level=3)

    bridge.step(step_t=1, field_quality={})

    assert sub.call_count == 0
    records = store.query("a1")
    assert records[0][1] == pytest.approx(0.4)


def test_no_candidates_returns_zeroed_report(tmp_path):
    """All patterns below min_bridge_level → patterns_checked=0, no errors."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [1, 2], weight=0.5)
    bridge = _make_bridge(StubSubstrate(), store, T_substrate=1, min_bridge_level=3)

    result = bridge.step(step_t=1, field_quality={})

    assert result["patterns_checked"] == 0
    assert result["mean_field_frequency"] == pytest.approx(0.0)


# ---- Echo-chamber audit ----

def test_echo_chamber_penalty_applied(tmp_path):
    """redundancy > threshold AND f_freq < low_threshold → penalty applied after boost."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    p_grounded = _make_pattern(3)
    p_ungrounded = _make_pattern(3)
    store.save(p_grounded, 0.5, "a1")
    store.save(p_ungrounded, 0.5, "a1")

    sub = StubSubstrate()
    sub.field_frequency = lambda p: 0.8 if p.id == p_grounded.id else 0.1
    bridge = _make_bridge(
        sub, store, T_substrate=1, alpha=0.1, gamma=0.3,
        redundancy_threshold=0.2, frequency_low_threshold=0.2
    )
    fq = {"redundancy": 0.5, "diversity": 1.0}

    bridge.step(step_t=1, field_quality=fq)

    recs = {p.id: w for p, w in store.query("a1")}
    # p_grounded boosted, p_ungrounded boosted then penalised → grounded >> ungrounded
    assert recs[p_grounded.id] > recs[p_ungrounded.id]


def test_echo_chamber_skipped_when_redundancy_none(tmp_path):
    """field_quality["redundancy"] = None → no penalty pass."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3], weight=0.5)
    sub = StubSubstrate(freq=0.05)  # below frequency_low_threshold
    bridge = _make_bridge(sub, store, T_substrate=1, gamma=0.5, redundancy_threshold=0.2)
    fq = {"redundancy": None}

    bridge.step(step_t=1, field_quality=fq)

    result_bridge = bridge.step(step_t=2, field_quality=fq)
    # Second step is non-cadence with T_substrate=1... let's reset
    # Just verify echo_chamber_penalty_applied is False for None redundancy
    bridge2 = _make_bridge(sub, store, T_substrate=1, gamma=0.5, redundancy_threshold=0.2)
    result = bridge2.step(step_t=1, field_quality={"redundancy": None})
    assert result["echo_chamber_penalty_applied"] is False


def test_echo_chamber_skipped_when_redundancy_low(tmp_path):
    """redundancy < threshold → no penalty even if f_freq is low."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3], weight=0.5)
    sub = StubSubstrate(freq=0.05)
    bridge = _make_bridge(sub, store, T_substrate=1, gamma=0.5, redundancy_threshold=0.4)
    fq = {"redundancy": 0.1}

    result = bridge.step(step_t=1, field_quality=fq)

    assert result["echo_chamber_penalty_applied"] is False


# ---- Frequency cache ----

def test_cache_hit_avoids_substrate_call(tmp_path):
    """Pattern queried once, mu unchanged → substrate.field_frequency not called second time."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3])
    sub = StubSubstrate(freq=0.5)
    bridge = _make_bridge(sub, store, T_substrate=1, cache_distance_threshold=0.1)

    bridge.step(step_t=1, field_quality={})
    assert sub.call_count == 1

    bridge.step(step_t=2, field_quality={})
    assert sub.call_count == 1  # cache hit — no new call


def test_cache_miss_on_mu_change(tmp_path):
    """Pattern mu changes beyond threshold → substrate re-queried."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    p = _make_pattern(3)
    store.save(p, 0.5, "a1")
    sub = StubSubstrate(freq=0.5)
    bridge = _make_bridge(sub, store, T_substrate=1, cache_distance_threshold=0.01)

    bridge.step(step_t=1, field_quality={})
    assert sub.call_count == 1

    # Mutate the pattern's mu significantly, delete and re-save
    store.delete(p.id)
    p2 = GaussianPattern(mu=np.ones(4) * 10.0, sigma=np.eye(4))
    p2.level = 3
    # Use same id to trigger cache check
    p2._id = p.id
    store.save(p2, 0.5, "a1")

    bridge.step(step_t=2, field_quality={})
    assert sub.call_count == 2  # cache miss — re-queried


# ---- Normalisation ----

def test_weights_normalised_per_agent(tmp_path):
    """After boost, each agent's pattern weights sum to 1.0."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3, 4, 5], weight=0.33)
    bridge = _make_bridge(StubSubstrate(freq=0.7), store, T_substrate=1, alpha=0.2)

    bridge.step(step_t=1, field_quality={})

    records = store.query("a1")
    total = sum(w for _, w in records)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_multi_agent_normalisation_independent(tmp_path):
    """Two agents normalised independently."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3, 4], weight=0.5)
    _seed(store, "a2", [3], weight=1.0)
    bridge = _make_bridge(StubSubstrate(freq=0.5), store, T_substrate=1, alpha=0.1)

    bridge.step(step_t=1, field_quality={})

    for agent_id in ["a1", "a2"]:
        records = store.query(agent_id)
        total = sum(w for _, w in records)
        assert total == pytest.approx(1.0, abs=1e-6)


# ---- Bridge report ----

def test_bridge_report_keys(tmp_path):
    """Cadence step always returns dict with all expected keys."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    _seed(store, "a1", [3])
    bridge = _make_bridge(StubSubstrate(), store, T_substrate=1)

    result = bridge.step(step_t=1, field_quality={"redundancy": 0.1})

    assert "patterns_checked" in result
    assert "cache_hits" in result
    assert "echo_chamber_penalty_applied" in result
    assert "mean_field_frequency" in result
    assert result["patterns_checked"] == 1
    assert result["mean_field_frequency"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/substrate/test_bridge.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'hpm.substrate.bridge'`

- [ ] **Step 3: Implement SubstrateBridgeAgent**

Create `hpm/substrate/bridge.py`:

```python
"""
hpm/substrate/bridge.py — Substrate Bridge Agent ("The Translator")

Anchors internal GaussianPattern weights to external symbolic systems.
Prevents echo-chamber effects by boosting externally-grounded patterns
and penalising ungrounded ones when the Librarian reports high redundancy.

Two-pass weight adjustment (every T_substrate steps):
  Pass 1 (Standard): w *= (1 + alpha * f_freq)        — boosts all Level 3+ patterns
  Pass 2 (Echo):     w *= (1 - gamma)                  — penalises low-freq patterns
                     (only when redundancy > threshold)

Weights are renormalised per agent after each full pass.
"""

import numpy as np


class SubstrateBridgeAgent:
    """
    Cadence-gated post-step processor that adjusts pattern weights based on
    field_frequency() scores from a connected ExternalSubstrate.

    Args:
        substrate:               Any ExternalSubstrate instance.
        store:                   Shared SQLiteStore (held as self._store).
        T_substrate:             Steps between substrate query passes.
        min_bridge_level:        Minimum pattern level included in frequency checks.
        alpha:                   Alignment boost scale.
        gamma:                   Echo-chamber grounding penalty.
        redundancy_threshold:    Redundancy level above which penalty pass activates.
        frequency_low_threshold: f_freq below which a pattern is "ungrounded".
        cache_distance_threshold: L2 norm below which cached f_freq is reused.
    """

    def __init__(
        self,
        substrate,
        store,
        T_substrate: int = 20,
        min_bridge_level: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.2,
        redundancy_threshold: float = 0.3,
        frequency_low_threshold: float = 0.2,
        cache_distance_threshold: float = 0.05,
    ):
        self._substrate = substrate
        self._store = store
        self.T_substrate = T_substrate
        self.min_bridge_level = min_bridge_level
        self.alpha = alpha
        self.gamma = gamma
        self.redundancy_threshold = redundancy_threshold
        self.frequency_low_threshold = frequency_low_threshold
        self.cache_distance_threshold = cache_distance_threshold
        self._t = 0
        self._freq_cache: dict = {}  # pattern_id -> (cached_mu, f_freq)

    def step(self, step_t: int, field_quality: dict) -> dict:
        """
        Called by MultiAgentOrchestrator after strategist.step().

        Returns {} on non-cadence steps.
        Returns bridge_report dict on cadence steps.
        """
        self._t += 1
        if self._t % self.T_substrate != 0:
            return {}

        # --- Snapshot ---
        all_records = self._store.query_all()  # (pattern, weight, agent_id)
        candidates = [
            (p, w, aid) for p, w, aid in all_records
            if getattr(p, "level", 1) >= self.min_bridge_level
        ]

        if not candidates:
            return {
                "patterns_checked": 0,
                "cache_hits": 0,
                "echo_chamber_penalty_applied": False,
                "mean_field_frequency": 0.0,
            }

        # --- Frequency cache + compute f_freq ---
        freq_map: dict = {}  # pattern_id -> f_freq
        cache_hits = 0
        for pattern, weight, agent_id in candidates:
            pid = pattern.id
            if pid in self._freq_cache:
                cached_mu, cached_freq = self._freq_cache[pid]
                if np.linalg.norm(pattern.mu - cached_mu) < self.cache_distance_threshold:
                    freq_map[pid] = cached_freq
                    cache_hits += 1
                    continue
            f_freq = float(self._substrate.field_frequency(pattern))
            self._freq_cache[pid] = (pattern.mu.copy(), f_freq)
            freq_map[pid] = f_freq

        # --- Standard Alignment Pass ---
        updated_weights: dict = {}  # pattern_id -> new weight after boost
        for pattern, weight, agent_id in candidates:
            pid = pattern.id
            f_freq = freq_map[pid]
            new_weight = weight * (1.0 + self.alpha * f_freq)
            self._store.update_weight(pid, new_weight)
            updated_weights[pid] = new_weight

        # --- Echo-Chamber Audit ---
        echo_penalty_applied = False
        redundancy = field_quality.get("redundancy")
        if redundancy is not None and redundancy > self.redundancy_threshold:
            for pattern, weight, agent_id in candidates:
                pid = pattern.id
                if freq_map[pid] < self.frequency_low_threshold:
                    penalised = updated_weights[pid] * (1.0 - self.gamma)
                    self._store.update_weight(pid, penalised)
                    updated_weights[pid] = penalised
            echo_penalty_applied = True

        # --- Per-Agent Normalisation ---
        agent_ids = {aid for _, _, aid in candidates}
        for aid in agent_ids:
            records = self._store.query(aid)
            total = sum(w for _, w in records)
            if total > 0:
                for p, w in records:
                    self._store.update_weight(p.id, w / total)

        mean_freq = float(np.mean(list(freq_map.values()))) if freq_map else 0.0

        return {
            "patterns_checked": len(candidates),
            "cache_hits": cache_hits,
            "echo_chamber_penalty_applied": echo_penalty_applied,
            "mean_field_frequency": mean_freq,
        }
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/substrate/test_bridge.py -v
```

Expected: all tests PASS. If `test_cache_miss_on_mu_change` fails because GaussianPattern doesn't expose `_id`, read `hpm/patterns/gaussian.py` to find the actual id attribute name and adjust the test accordingly.

- [ ] **Step 5: Run full suite**

```bash
uv run pytest -q
```

Expected: 315+ passed, 9 skipped.

- [ ] **Step 6: Commit**

```bash
git add hpm/substrate/bridge.py tests/substrate/test_bridge.py
git commit -m "feat: SubstrateBridgeAgent with alignment boost, echo-chamber penalty, and frequency cache"
```

---

### Task 2: MultiAgentOrchestrator integration + __init__ export

**Files:**
- Modify: `hpm/agents/multi_agent.py`
- Modify: `hpm/substrate/__init__.py`
- Modify: `tests/substrate/test_bridge.py` (add integration tests)

- [ ] **Step 1: Read hpm/agents/multi_agent.py**

Find:
- The `__init__` signature (current: `agents, field, seed_pattern=None, groups=None, monitor=None, strategist=None`)
- Where `strategist.step()` is called and `interventions` is computed
- The final return statement (`return {**metrics, "field_quality": field_quality, "interventions": interventions}`)

- [ ] **Step 2: Write failing integration tests**

Append to `tests/substrate/test_bridge.py`:

```python
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.field.field import PatternField
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.monitor.structural_law import StructuralLawMonitor
from hpm.substrate.bridge import SubstrateBridgeAgent


def _make_real_agent(store, dim=4):
    config = AgentConfig(feature_dim=dim, agent_id=f"agent_{id(store)}")
    return Agent(config, store=store)


def test_orchestrator_no_bridge(tmp_path):
    """Orchestrator with bridge=None returns bridge_report == {}."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = _make_real_agent(store)
    field = PatternField()
    orch = MultiAgentOrchestrator([agent], field=field, bridge=None)
    obs = {agent.agent_id: np.zeros(4)}
    result = orch.step(obs)
    assert result.get("bridge_report") == {}


def test_orchestrator_bridge_integrated(tmp_path):
    """Orchestrator with bridge returns bridge_report after T_substrate steps."""
    store = SQLiteStore(str(tmp_path / "test.db"))
    agent = _make_real_agent(store)
    field = PatternField()
    monitor = StructuralLawMonitor(store, T_monitor=100)
    bridge = SubstrateBridgeAgent(StubSubstrate(), store, T_substrate=5)
    orch = MultiAgentOrchestrator([agent], field=field, monitor=monitor, bridge=bridge)
    obs = {agent.agent_id: np.zeros(4)}

    result = None
    for _ in range(5):
        result = orch.step(obs)

    assert "bridge_report" in result
    # On step 5 (T_substrate=5), bridge_report should be non-empty dict
    br = result["bridge_report"]
    if br:  # may be empty if no Level 3+ patterns yet
        assert "patterns_checked" in br
```

Run: `uv run pytest tests/substrate/test_bridge.py::test_orchestrator_no_bridge -v`
Expected: FAIL (`MultiAgentOrchestrator` has no `bridge` parameter yet)

- [ ] **Step 3: Add bridge to MultiAgentOrchestrator**

In `hpm/agents/multi_agent.py`, update `__init__` signature:

```python
def __init__(self, agents, field: PatternField, seed_pattern=None, groups=None, monitor=None, strategist=None, bridge=None):
```

Add to `__init__` body:
```python
self.bridge = bridge
```

In `step()`, after the `interventions` computation (after `strategist.step()` call), add:

```python
bridge_report = (
    self.bridge.step(self._t, field_quality)
    if self.bridge is not None
    else {}
)
```

Update the return statement. Replace:
```python
        return {**metrics, "field_quality": field_quality, "interventions": interventions}
```
With:
```python
        return {**metrics, "field_quality": field_quality, "interventions": interventions, "bridge_report": bridge_report}
```

- [ ] **Step 4: Export SubstrateBridgeAgent from hpm/substrate/__init__.py**

In `hpm/substrate/__init__.py`, add:
```python
from .bridge import SubstrateBridgeAgent
```

- [ ] **Step 5: Run all substrate tests**

```bash
uv run pytest tests/substrate/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Run full suite**

```bash
uv run pytest -q
```

Expected: 318+ passed, 9 skipped (no regressions).

- [ ] **Step 7: Commit**

```bash
git add hpm/agents/multi_agent.py hpm/substrate/__init__.py tests/substrate/test_bridge.py
git commit -m "feat: integrate SubstrateBridgeAgent into MultiAgentOrchestrator"
```

---

### Task 3: Smoke test and final verification

**Files:** No new files.

- [ ] **Step 1: Write and run smoke test**

```python
# Save as smoke_bridge.py, delete after
import numpy as np
import tempfile, os
from hpm.store.sqlite import SQLiteStore
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.field.field import PatternField
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.monitor.structural_law import StructuralLawMonitor
from hpm.substrate.bridge import SubstrateBridgeAgent
from hpm.substrate.linguistic import LinguisticSubstrate

with tempfile.TemporaryDirectory() as tmp:
    store = SQLiteStore(os.path.join(tmp, "smoke.db"))
    config = AgentConfig(feature_dim=32, agent_id="smoke_agent")
    agents = [Agent(config, store=store)]
    field = PatternField()
    monitor = StructuralLawMonitor(store, T_monitor=10)
    # Use LinguisticSubstrate in offline mode (NLTK word list, no HTTP)
    substrate = LinguisticSubstrate(feature_dim=32)
    bridge = SubstrateBridgeAgent(substrate, store, T_substrate=5)
    orch = MultiAgentOrchestrator(
        agents, field=field, monitor=monitor, bridge=bridge
    )

    rng = np.random.default_rng(42)
    for t in range(20):
        obs = {config.agent_id: rng.normal(0, 1, 32)}
        result = orch.step(obs)

    assert "bridge_report" in result
    br = result["bridge_report"]
    # Step 20 is divisible by T_substrate=5 → non-empty report
    assert "patterns_checked" in br
    assert "mean_field_frequency" in br
    print("Smoke test PASSED")
    print(f"  patterns_checked={br['patterns_checked']}")
    print(f"  cache_hits={br['cache_hits']}")
    print(f"  mean_field_frequency={br['mean_field_frequency']:.3f}")
    print(f"  echo_chamber_penalty_applied={br['echo_chamber_penalty_applied']}")
```

Run: `uv run python smoke_bridge.py`
Expected: prints `Smoke test PASSED`

- [ ] **Step 2: Delete smoke script**

```bash
rm smoke_bridge.py
```

- [ ] **Step 3: Run full test suite one final time**

```bash
uv run pytest -q
```

Expected: 318+ passed, 9 skipped.

- [ ] **Step 4: Verify git log**

```bash
git log --oneline -3
```

Expected:
```
feat: integrate SubstrateBridgeAgent into MultiAgentOrchestrator
feat: SubstrateBridgeAgent with alignment boost, echo-chamber penalty, and frequency cache
feat: integrate RecombinationStrategist into MultiAgentOrchestrator
```
