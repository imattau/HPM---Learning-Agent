# Hierarchical Abstraction Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a 2-level abstraction stack where Level 1 HPM agents process raw observations and their outputs become the substrate for Level 2 agents, enabling hierarchical pattern learning.

**Architecture:** A new `HierarchicalOrchestrator` wraps two `MultiAgentOrchestrator` instances. Every K steps, it extracts a structured bundle `[μ, w, L]` from each Level 1 agent and delivers N separate `step()` calls to the Level 2 orchestrator. Level 2 agents use standard `GaussianPattern` with `feature_dim = l1_feature_dim + 2`.

**Tech Stack:** Python, NumPy, existing `hpm` package (`Agent`, `MultiAgentOrchestrator`, `GaussianPattern`, `InMemoryStore`, `EpistemicEvaluator`), pytest.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `hpm/agents/hierarchical.py` | Create | `LevelBundle`, `extract_bundle`, `encode_bundle`, `make_hierarchical_orchestrator`, `HierarchicalOrchestrator` |
| `hpm/agents/__init__.py` | Modify | Re-export new symbols |
| `tests/agents/test_hierarchical.py` | Create | All unit tests |
| `benchmarks/hierarchical_smoke.py` | Create | End-to-end synthetic validation |

---

## Key APIs to understand before starting

**`agent.store.query(agent_id)`** — returns `list[tuple[pattern, float]]` (2-tuples of pattern + weight). Defined in `hpm/store/memory.py`.

**`agent.epistemic._running_loss`** — dict mapping `pattern.id -> float`. The running epistemic loss per pattern. Defined in `hpm/evaluators/epistemic.py`.

**`make_orchestrator(n_agents, feature_dim, agent_ids, pattern_types, ...)`** — factory in `benchmarks/multi_agent_common.py`. Returns `(orchestrator, agents, store)`.

**`MultiAgentOrchestrator.step(observations: dict)`** — takes `{agent_id: obs_array}` and steps all agents. Returns dict of per-agent metrics.

---

### Task 1: LevelBundle dataclass + encode_bundle

**Files:**
- Create: `hpm/agents/hierarchical.py`
- Create: `tests/agents/test_hierarchical.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/agents/test_hierarchical.py
import numpy as np
import pytest
from hpm.agents.hierarchical import LevelBundle, encode_bundle


def test_encode_bundle_shape():
    bundle = LevelBundle(agent_id="a", mu=np.zeros(16), weight=0.5, epistemic_loss=0.1)
    encoded = encode_bundle(bundle)
    assert encoded.shape == (18,)


def test_encode_bundle_values():
    mu = np.ones(4)
    bundle = LevelBundle(agent_id="a", mu=mu, weight=0.7, epistemic_loss=0.3)
    encoded = encode_bundle(bundle)
    np.testing.assert_allclose(encoded[:4], mu)
    assert encoded[4] == pytest.approx(0.7)
    assert encoded[5] == pytest.approx(0.3)


def test_level_bundle_fields():
    b = LevelBundle(agent_id="x", mu=np.zeros(8), weight=1.0, epistemic_loss=0.0)
    assert b.agent_id == "x"
    assert b.mu.shape == (8,)
    assert b.weight == 1.0
    assert b.epistemic_loss == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/agents/test_hierarchical.py -v
```
Expected: `ModuleNotFoundError` or `ImportError` — `hierarchical.py` does not exist yet.

- [ ] **Step 3: Implement LevelBundle and encode_bundle**

```python
# hpm/agents/hierarchical.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class LevelBundle:
    """Structured inter-level signal: belief + confidence from one Level 1 agent."""
    agent_id: str
    mu: np.ndarray        # shape (D,) — top pattern mean
    weight: float         # top pattern's store weight
    epistemic_loss: float # running epistemic loss for that pattern


def encode_bundle(bundle: LevelBundle) -> np.ndarray:
    """Concatenate [mu, weight, epistemic_loss] into a single observation vector.

    Output shape: (D + 2,) where D = len(bundle.mu).
    This becomes the raw observation fed to Level 2 agents.
    """
    return np.concatenate([bundle.mu, [bundle.weight, bundle.epistemic_loss]])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/agents/test_hierarchical.py::test_encode_bundle_shape \
    tests/agents/test_hierarchical.py::test_encode_bundle_values \
    tests/agents/test_hierarchical.py::test_level_bundle_fields -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm/agents/hierarchical.py tests/agents/test_hierarchical.py
git commit -m "feat: add LevelBundle dataclass and encode_bundle"
```

---

### Task 2: extract_bundle

**Files:**
- Modify: `hpm/agents/hierarchical.py`
- Modify: `tests/agents/test_hierarchical.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/agents/test_hierarchical.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from hpm.agents.hierarchical import extract_bundle, LevelBundle
from hpm.config import AgentConfig
from hpm.agents.agent import Agent
from hpm.store.memory import InMemoryStore
from hpm.field.field import PatternField
from hpm.patterns.gaussian import GaussianPattern


def _make_agent(feature_dim=8, agent_id="test_agent"):
    cfg = AgentConfig(agent_id=agent_id, feature_dim=feature_dim)
    store = InMemoryStore()
    field = PatternField()
    return Agent(cfg, store=store, field=field)


def test_extract_bundle_empty_store_returns_zeros():
    agent = _make_agent(feature_dim=8)
    # Manually clear store to trigger empty-store guard
    agent.store._data.clear()
    bundle = extract_bundle(agent)
    assert bundle.agent_id == agent.agent_id
    assert bundle.mu.shape == (8,)
    np.testing.assert_allclose(bundle.mu, np.zeros(8))
    assert bundle.weight == 0.0
    assert bundle.epistemic_loss == 1.0


def test_extract_bundle_populated_returns_top_pattern():
    agent = _make_agent(feature_dim=4)
    rng = np.random.default_rng(0)
    # Step the agent a few times so it has patterns
    for _ in range(5):
        obs = rng.standard_normal(4)
        agent.step(obs)
    bundle = extract_bundle(agent)
    assert bundle.agent_id == agent.agent_id
    assert bundle.mu.shape == (4,)
    assert isinstance(bundle.weight, float)
    assert isinstance(bundle.epistemic_loss, float)
    assert np.isfinite(bundle.mu).all()


def test_extract_bundle_top_is_highest_weight():
    agent = _make_agent(feature_dim=4)
    rng = np.random.default_rng(42)
    for _ in range(20):
        agent.step(rng.standard_normal(4))
    bundle = extract_bundle(agent)
    records = agent.store.query(agent.agent_id)
    max_weight = max(w for _, w in records)
    assert bundle.weight == pytest.approx(max_weight)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/agents/test_hierarchical.py -k "extract_bundle" -v
```
Expected: `ImportError` for `extract_bundle`.

- [ ] **Step 3: Implement extract_bundle**

Add to `hpm/agents/hierarchical.py`:

```python
from hpm.agents.agent import Agent


def extract_bundle(agent: Agent) -> LevelBundle:
    """Extract a structured bundle from an agent's current state.

    Reads the top-weighted pattern from the agent's store.
    If the store is empty (only possible with manually-cleared stores in tests),
    returns a zero bundle with maximum uncertainty (epistemic_loss=1.0).
    """
    feature_dim = agent.config.feature_dim
    records = agent.store.query(agent.agent_id)

    if not records:
        return LevelBundle(
            agent_id=agent.agent_id,
            mu=np.zeros(feature_dim),
            weight=0.0,
            epistemic_loss=1.0,
        )

    top_pattern, top_weight = max(records, key=lambda r: r[1])
    epistemic_loss = agent.epistemic._running_loss.get(top_pattern.id, 0.0)

    return LevelBundle(
        agent_id=agent.agent_id,
        mu=top_pattern.mu.copy(),
        weight=float(top_weight),
        epistemic_loss=float(epistemic_loss),
    )
```

Note: `top_pattern.mu` assumes `GaussianPattern`. This is correct for Sub-project 1 where Level 1 uses Gaussian patterns. Sub-project 2 will generalise this.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/agents/test_hierarchical.py -v
```
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm/agents/hierarchical.py tests/agents/test_hierarchical.py
git commit -m "feat: add extract_bundle"
```

---

### Task 3: HierarchicalOrchestrator

**Files:**
- Modify: `hpm/agents/hierarchical.py`
- Modify: `tests/agents/test_hierarchical.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/agents/test_hierarchical.py`:

```python
from hpm.agents.hierarchical import (
    HierarchicalOrchestrator, make_hierarchical_orchestrator
)


def test_hierarchical_orchestrator_cadence_k1():
    """K=1: Level 2 steps on every Level 1 step."""
    h_orch, l1_agents, l2_agents = make_hierarchical_orchestrator(
        n_l1_agents=2, n_l2_agents=1, l1_feature_dim=8, K=1,
    )
    rng = np.random.default_rng(0)
    l2_call_count = 0
    for _ in range(10):
        result = h_orch.step(rng.standard_normal(8))
        if result["level2"]:
            l2_call_count += 1
    assert l2_call_count == 10


def test_hierarchical_orchestrator_cadence_k5():
    """K=5: Level 2 steps only at t=5,10,15,20."""
    h_orch, l1_agents, l2_agents = make_hierarchical_orchestrator(
        n_l1_agents=2, n_l2_agents=1, l1_feature_dim=8, K=5,
    )
    rng = np.random.default_rng(0)
    l2_call_count = 0
    for _ in range(20):
        result = h_orch.step(rng.standard_normal(8))
        if result["level2"]:
            l2_call_count += 1
    assert l2_call_count == 4  # steps 5, 10, 15, 20


def test_hierarchical_orchestrator_no_l2_on_noncadence():
    """level2 key is {} on non-cadence steps."""
    h_orch, _, _ = make_hierarchical_orchestrator(
        n_l1_agents=2, n_l2_agents=1, l1_feature_dim=8, K=5,
    )
    rng = np.random.default_rng(0)
    result = h_orch.step(rng.standard_normal(8))  # t=1, not a cadence step
    assert result["level2"] == {}


def test_hierarchical_orchestrator_k_larger_than_steps():
    """K > n_steps: Level 2 never steps, no error."""
    h_orch, _, _ = make_hierarchical_orchestrator(
        n_l1_agents=2, n_l2_agents=1, l1_feature_dim=8, K=100,
    )
    rng = np.random.default_rng(0)
    for _ in range(10):
        result = h_orch.step(rng.standard_normal(8))
    assert result["level2"] == {}


def test_hierarchical_orchestrator_l2_bundle_shape():
    """Level 2 receives bundles of shape (l1_feature_dim + 2,)."""
    l1_dim = 8
    h_orch, l1_agents, l2_agents = make_hierarchical_orchestrator(
        n_l1_agents=2, n_l2_agents=1, l1_feature_dim=l1_dim, K=1,
    )
    rng = np.random.default_rng(0)
    # Step once and check L2 feature_dim
    h_orch.step(rng.standard_normal(l1_dim))
    assert l2_agents[0].config.feature_dim == l1_dim + 2


def test_hierarchical_orchestrator_returns_t():
    """step() return dict includes 't' counter."""
    h_orch, _, _ = make_hierarchical_orchestrator(
        n_l1_agents=1, n_l2_agents=1, l1_feature_dim=4, K=1,
    )
    for i in range(1, 4):
        result = h_orch.step(np.zeros(4))
        assert result["t"] == i
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/agents/test_hierarchical.py -k "orchestrator" -v
```
Expected: `ImportError` — `HierarchicalOrchestrator` not yet defined.

- [ ] **Step 3: Implement HierarchicalOrchestrator and make_hierarchical_orchestrator**

Add to `hpm/agents/hierarchical.py`:

```python
import pathlib
from hpm.agents.multi_agent import MultiAgentOrchestrator


class HierarchicalOrchestrator:
    """Coordinates a 2-level abstraction stack.

    Level 1 agents process raw observations every step.
    Level 2 agents receive structured bundles from Level 1 every K steps.

    Bundle format: np.concatenate([mu, [weight, epistemic_loss]])
    shape: (l1_feature_dim + 2,)

    All Level 2 agents receive the same bundle per call.
    N Level 1 agents produce N separate step() calls to level2_orch per cadence tick.
    """

    def __init__(
        self,
        level1_orch: MultiAgentOrchestrator,
        level2_orch: MultiAgentOrchestrator,
        level1_agents: list,
        level2_agents: list,
        K: int = 1,
    ):
        self.level1_orch = level1_orch
        self.level2_orch = level2_orch
        self.level1_agents = level1_agents
        self.level2_agents = level2_agents
        self.K = K
        self._t = 0

    def step(self, obs: np.ndarray) -> dict:
        """Step the hierarchy.

        Always steps all Level 1 agents.
        Steps Level 2 only when self._t % K == 0 (after increment).
        Returns {"level1": ..., "level2": {} or last_l2_result, "t": self._t}.
        """
        l1_obs = {a.agent_id: obs for a in self.level1_agents}
        l1_result = self.level1_orch.step(l1_obs)

        self._t += 1
        l2_result = {}

        if self._t % self.K == 0:
            l2_agent_ids = [a.agent_id for a in self.level2_agents]
            for l1_agent in self.level1_agents:
                bundle = extract_bundle(l1_agent)
                encoded = encode_bundle(bundle)
                l2_obs = {aid: encoded for aid in l2_agent_ids}
                l2_result = self.level2_orch.step(l2_obs)

        return {"level1": l1_result, "level2": l2_result, "t": self._t}


def make_hierarchical_orchestrator(
    n_l1_agents: int,
    n_l2_agents: int,
    l1_feature_dim: int,
    K: int = 1,
    l1_pattern_type: str = "gaussian",
    l2_pattern_type: str = "gaussian",
    l1_agent_ids: list[str] | None = None,
    l2_agent_ids: list[str] | None = None,
) -> tuple:
    """Build a 2-level HierarchicalOrchestrator.

    Level 2 feature_dim is automatically set to l1_feature_dim + 2.
    This is the only supported construction path — do not construct
    HierarchicalOrchestrator directly with mismatched orchestrators.

    NOTE on epistemic_loss scale: _running_loss stores raw NLL (unbounded positive
    float). The empty-store guard returns 1.0 as a sentinel. Real values may be
    much larger. Level 2 agents receive these as-is; the existing MetaPatternRule
    will still learn from them but L2 patterns will reflect NLL-scale inputs.
    Normalisation is deferred to Sub-project 2.

    Returns: (HierarchicalOrchestrator, level1_agents, level2_agents)
    """
    # Lazy import from benchmarks/ — kept inside function body to avoid
    # polluting hpm/ module namespace with a benchmarks/ dependency at import time.
    import sys
    _repo_root = str(pathlib.Path(__file__).resolve().parent.parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from benchmarks.multi_agent_common import make_orchestrator  # noqa: E402

    l1_ids = l1_agent_ids or [f"l1_{i}" for i in range(n_l1_agents)]
    l2_ids = l2_agent_ids or [f"l2_{i}" for i in range(n_l2_agents)]

    l1_orch, l1_agents, _ = make_orchestrator(
        n_agents=n_l1_agents,
        feature_dim=l1_feature_dim,
        agent_ids=l1_ids,
        pattern_types=[l1_pattern_type] * n_l1_agents,
    )
    l2_orch, l2_agents, _ = make_orchestrator(
        n_agents=n_l2_agents,
        feature_dim=l1_feature_dim + 2,
        agent_ids=l2_ids,
        pattern_types=[l2_pattern_type] * n_l2_agents,
    )

    return (
        HierarchicalOrchestrator(l1_orch, l2_orch, l1_agents, l2_agents, K=K),
        l1_agents,
        l2_agents,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/agents/test_hierarchical.py -v
```
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm/agents/hierarchical.py tests/agents/test_hierarchical.py
git commit -m "feat: add HierarchicalOrchestrator and make_hierarchical_orchestrator"
```

---

### Task 4: __init__ re-export

**Files:**
- Modify: `hpm/agents/__init__.py`

- [ ] **Step 1: Check current exports**

```bash
cat hpm/agents/__init__.py
```

- [ ] **Step 2: Add re-exports**

Add to `hpm/agents/__init__.py`:

```python
from .hierarchical import (
    LevelBundle,
    encode_bundle,
    extract_bundle,
    HierarchicalOrchestrator,
    make_hierarchical_orchestrator,
)
```

- [ ] **Step 3: Verify imports work**

```bash
python -c "from hpm.agents import HierarchicalOrchestrator, LevelBundle, make_hierarchical_orchestrator; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Run full test suite to catch regressions**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: All existing tests still PASS, new hierarchical tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hpm/agents/__init__.py
git commit -m "feat: re-export hierarchical symbols from hpm.agents"
```

---

### Task 5: Smoke benchmark

**Files:**
- Create: `benchmarks/hierarchical_smoke.py`

- [ ] **Step 1: Write the benchmark**

```python
"""
Hierarchical Abstraction Stack — Smoke Benchmark
=================================================
Validates the 2-level HierarchicalOrchestrator end-to-end on a synthetic
Gaussian signal. Not ARC — just verifies that:
  1. Level 2 receives bundles at the correct K-cadence
  2. Level 2 accuracy is finite and non-NaN after training
  3. The _t counter increments correctly

Run:
    python benchmarks/hierarchical_smoke.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from hpm.agents.hierarchical import make_hierarchical_orchestrator
from benchmarks.common import print_results_table

L1_FEATURE_DIM = 16
N_L1_AGENTS = 2
N_L2_AGENTS = 1
N_STEPS = 100
K = 5
RNG_SEED = 42


def run() -> dict:
    rng = np.random.default_rng(RNG_SEED)
    mu = rng.standard_normal(L1_FEATURE_DIM)
    mu /= np.linalg.norm(mu)

    h_orch, l1_agents, l2_agents = make_hierarchical_orchestrator(
        n_l1_agents=N_L1_AGENTS,
        n_l2_agents=N_L2_AGENTS,
        l1_feature_dim=L1_FEATURE_DIM,
        K=K,
    )

    l2_cadence_ticks = 0   # how many times the cadence fired
    l2_accs = []

    for step in range(N_STEPS):
        obs = rng.normal(loc=mu, scale=0.1)
        result = h_orch.step(obs)

        if result["level2"]:
            l2_cadence_ticks += 1
            # Collect mean accuracy from Level 2 agents
            for aid, metrics in result["level2"].items():
                if "mean_accuracy" in metrics:
                    l2_accs.append(metrics["mean_accuracy"])

    # Each cadence tick delivers N_L1_AGENTS separate step() calls to level2_orch.
    # expected_l2_steps = ticks * N_L1_AGENTS (Witness Model, one bundle per L1 agent)
    expected_l2_ticks = N_STEPS // K
    expected_l2_steps = expected_l2_ticks * N_L1_AGENTS  # = 20 * 2 = 40
    expected_t = N_STEPS

    return {
        "n_steps": N_STEPS,
        "K": K,
        "l2_cadence_ticks": l2_cadence_ticks,
        "expected_l2_ticks": expected_l2_ticks,
        "expected_l2_steps": expected_l2_steps,
        "final_t": h_orch._t,
        "expected_t": expected_t,
        "l2_mean_acc": float(np.mean(l2_accs)) if l2_accs else float("nan"),
        "l2_acc_finite": bool(np.isfinite(l2_accs).all()) if l2_accs else False,
        "cadence_correct": l2_cadence_ticks == expected_l2_ticks,
        "t_correct": h_orch._t == expected_t,
    }


def main():
    print(f"Running Hierarchical Smoke Benchmark "
          f"({N_L1_AGENTS} L1 agents → {N_L2_AGENTS} L2 agent, K={K}, {N_STEPS} steps)...")
    m = run()

    cadence_ok = m["cadence_correct"]
    t_ok = m["t_correct"]
    acc_ok = m["l2_acc_finite"]
    passed = cadence_ok and t_ok and acc_ok

    print_results_table(
        title="Hierarchical Smoke Benchmark",
        cols=["Check", "Expected", "Actual", "Status"],
        rows=[
            {
                "Check": "L2 cadence ticks",
                "Expected": str(m["expected_l2_ticks"]),
                "Actual": str(m["l2_cadence_ticks"]),
                "Status": "✓" if cadence_ok else "✗",
            },
            {
                "Check": "Final _t",
                "Expected": str(m["expected_t"]),
                "Actual": str(m["final_t"]),
                "Status": "✓" if t_ok else "✗",
            },
            {
                "Check": "L2 accuracy finite",
                "Expected": "yes",
                "Actual": f"{m['l2_mean_acc']:.4f}" if acc_ok else "NaN",
                "Status": "✓" if acc_ok else "✗",
            },
            {
                "Check": "Overall",
                "Expected": "",
                "Actual": "",
                "Status": "✓ PASS" if passed else "✗ FAIL",
            },
        ],
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark**

```bash
python benchmarks/hierarchical_smoke.py
```
Expected output: all three checks ✓, result ✓ PASS.

- [ ] **Step 3: Run full test suite one final time**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/hierarchical_smoke.py
git commit -m "feat: add hierarchical_smoke benchmark"
```

- [ ] **Step 5: Push**

```bash
git push
```
