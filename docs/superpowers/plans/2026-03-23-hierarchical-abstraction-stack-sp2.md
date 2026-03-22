# Hierarchical Abstraction Stack — Sub-project 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a general N-level `StackedOrchestrator` that supports arbitrary abstraction depth, and add a separate hierarchical ARC benchmark to compare 3-level stacked performance against the flat single-agent baseline.

**Architecture:** `hpm/agents/stacked.py` introduces `LevelConfig` (per-level config dataclass), `StackedOrchestrator` (N-level stack with per-level cadence and the witness model from Sub-project 1), and `make_stacked_orchestrator` (factory that auto-computes dims). `benchmarks/hierarchical_arc.py` runs a 3-level stack (L1=64-dim, L2=66-dim, L3=68-dim) on the existing ARC encoding. **Scoring uses L1 agents** — L3 agents operate in 68-dim bundle space and cannot directly score 64-dim ARC vectors; L1 agents share the same feature space as the raw ARC encoding, so the benchmark tests whether the hierarchical stack shapes L1 representations better than a flat setup. Existing `arc_benchmark.py` and `multi_agent_arc.py` are NOT modified.

**Tech Stack:** Python 3.11+, NumPy, existing `hpm` package (`Agent`, `MultiAgentOrchestrator`, `extract_bundle`, `encode_bundle`), existing `benchmarks/multi_agent_common.py` (`make_orchestrator`), pytest.

---

## Key APIs to understand before starting

**`hpm/agents/hierarchical.py`** — already implemented. Provides `extract_bundle(agent)`, `encode_bundle(bundle)`. Study its `make_hierarchical_orchestrator` for the lazy-import pattern used to call `benchmarks/multi_agent_common.make_orchestrator` from inside `hpm/`.

**`benchmarks/multi_agent_arc.py`** — provides `encode_pair(input_grid, output_grid)`, `ensemble_score(agents, vec)`, `load_tasks()`, `task_fits(task)`. The hierarchical benchmark reuses all of these. Note: tasks are loaded from the full dataset and capped at `[:400]`; after applying `task_fits`, ~342 tasks remain — this is the comparison baseline.

**`benchmarks/multi_agent_common.py`** — `make_orchestrator(n_agents, feature_dim, agent_ids, pattern_types, ...)` returns `(orchestrator, agents, store)`.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `hpm/agents/stacked.py` | Create | `LevelConfig`, `StackedOrchestrator`, `make_stacked_orchestrator` |
| `hpm/agents/__init__.py` | Modify | Re-export new symbols |
| `tests/agents/test_stacked.py` | Create | All unit tests (written before implementation) |
| `benchmarks/hierarchical_arc.py` | Create | Hierarchical ARC benchmark |

---

### Task 1: All tests + full implementation of stacked.py

**Files:**
- Create: `tests/agents/test_stacked.py` (tests first)
- Create: `hpm/agents/stacked.py` (implementation after)

- [ ] **Step 1: Write ALL failing tests**

```python
# tests/agents/test_stacked.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pytest
from hpm.agents.stacked import LevelConfig, StackedOrchestrator, make_stacked_orchestrator


# ---------------------------------------------------------------------------
# LevelConfig defaults
# ---------------------------------------------------------------------------

def test_level_config_defaults():
    cfg = LevelConfig(n_agents=2)
    assert cfg.pattern_type == "gaussian"
    assert cfg.K == 1
    assert cfg.agent_ids is None


# ---------------------------------------------------------------------------
# make_stacked_orchestrator: dimension computation
# ---------------------------------------------------------------------------

def test_make_stacked_orchestrator_dims_3level():
    """L1=64, L2=66, L3=68 for a 3-level stack."""
    configs = [
        LevelConfig(n_agents=2),
        LevelConfig(n_agents=2, K=3),
        LevelConfig(n_agents=1, K=3),
    ]
    orch, all_agents = make_stacked_orchestrator(l1_feature_dim=64, level_configs=configs)
    assert all_agents[0][0].config.feature_dim == 64
    assert all_agents[1][0].config.feature_dim == 66
    assert all_agents[2][0].config.feature_dim == 68


def test_make_stacked_orchestrator_returns_correct_agent_counts():
    configs = [LevelConfig(n_agents=2), LevelConfig(n_agents=3, K=2)]
    orch, all_agents = make_stacked_orchestrator(l1_feature_dim=8, level_configs=configs)
    assert len(all_agents[0]) == 2
    assert len(all_agents[1]) == 3


# ---------------------------------------------------------------------------
# StackedOrchestrator.level_agents is public and matches factory output
# ---------------------------------------------------------------------------

def test_stacked_orchestrator_level_agents_public():
    configs = [LevelConfig(n_agents=2), LevelConfig(n_agents=1, K=2)]
    orch, all_agents = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    assert len(orch.level_agents) == 2
    assert orch.level_agents[0] is all_agents[0]
    assert orch.level_agents[1] is all_agents[1]


def test_stacked_orchestrator_level_agents_shape():
    configs = [LevelConfig(n_agents=2), LevelConfig(n_agents=2, K=1), LevelConfig(n_agents=1, K=1)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    assert len(orch.level_agents) == 3
    assert len(orch.level_agents[0]) == 2
    assert len(orch.level_agents[1]) == 2
    assert len(orch.level_agents[2]) == 1


# ---------------------------------------------------------------------------
# level_Ks: K[0]=1 always; K[1:] from configs[1:]
# ---------------------------------------------------------------------------

def test_stacked_orchestrator_level_ks():
    configs = [
        LevelConfig(n_agents=1, K=99),  # K=99 on L1 must be ignored
        LevelConfig(n_agents=1, K=3),
        LevelConfig(n_agents=1, K=5),
    ]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    assert orch.level_Ks[0] == 1   # L1 always steps
    assert orch.level_Ks[1] == 3
    assert orch.level_Ks[2] == 5


# ---------------------------------------------------------------------------
# _level_ticks initialised to zeros
# ---------------------------------------------------------------------------

def test_stacked_orchestrator_ticks_initial():
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=3)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    assert orch._level_ticks == [0, 0]


# ---------------------------------------------------------------------------
# Cadence: K=1 (synchronous)
# ---------------------------------------------------------------------------

def test_cadence_k1_all_levels_fire_every_step():
    """K=1: level2 fires on every step."""
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=1)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    rng = np.random.default_rng(0)
    for _ in range(10):
        result = orch.step(rng.standard_normal(4))
        assert result["level2"] != {}


def test_cadence_k1_ticks_equal():
    """K=1 2-level: _level_ticks[1] == _level_ticks[0] after N steps."""
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=1)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    rng = np.random.default_rng(0)
    for _ in range(15):
        orch.step(rng.standard_normal(4))
    assert orch._level_ticks[1] == orch._level_ticks[0]


def test_cadence_k1_l2_feature_dim():
    """2-level K=1: L2 agents have feature_dim = l1_feature_dim + 2."""
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=1)]
    orch, agents = make_stacked_orchestrator(l1_feature_dim=8, level_configs=configs)
    assert agents[1][0].config.feature_dim == 10


# ---------------------------------------------------------------------------
# Cadence: K=3
# ---------------------------------------------------------------------------

def test_cadence_k3_l2_fires_at_correct_steps():
    """K=3: level2 fires at t=3,6,9,12."""
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=3)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    rng = np.random.default_rng(0)
    fire_steps = []
    for _ in range(12):
        result = orch.step(rng.standard_normal(4))
        if result["level2"] != {}:
            fire_steps.append(result["t"])
    assert fire_steps == [3, 6, 9, 12]


def test_cadence_k3_tick_counters():
    """K=3: after 9 steps, _level_ticks[0]==9, _level_ticks[1]==3."""
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=3)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    rng = np.random.default_rng(0)
    for _ in range(9):
        orch.step(rng.standard_normal(4))
    assert orch._level_ticks[0] == 9
    assert orch._level_ticks[1] == 3


# ---------------------------------------------------------------------------
# Cadence: 3-level K=3
# ---------------------------------------------------------------------------

def test_cadence_3level_k3_l3_fires_at_t9_18_27():
    """3-level K=3: L3 fires at t=9,18,27 (every 9 L1 steps)."""
    configs = [
        LevelConfig(n_agents=2),
        LevelConfig(n_agents=2, K=3),
        LevelConfig(n_agents=1, K=3),
    ]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    rng = np.random.default_rng(0)
    l3_fire_steps = []
    for _ in range(30):
        result = orch.step(rng.standard_normal(4))
        if result["level3"] != {}:
            l3_fire_steps.append(result["t"])
    assert l3_fire_steps == [9, 18, 27]


# ---------------------------------------------------------------------------
# Non-cadence steps return {}
# ---------------------------------------------------------------------------

def test_non_cadence_step_returns_empty():
    """t=1 with K=5: level2 is {}."""
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=5)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    result = orch.step(np.zeros(4))  # t=1
    assert result["level2"] == {}


# ---------------------------------------------------------------------------
# t counter
# ---------------------------------------------------------------------------

def test_t_counter_increments():
    """result["t"] == step number (1-indexed)."""
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=1)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    for i in range(1, 6):
        result = orch.step(np.zeros(4))
        assert result["t"] == i


# ---------------------------------------------------------------------------
# K larger than n_steps
# ---------------------------------------------------------------------------

def test_k_larger_than_steps_no_error():
    """K=100, 10 steps: level2 never fires, no exception."""
    configs = [LevelConfig(n_agents=1), LevelConfig(n_agents=1, K=100)]
    orch, _ = make_stacked_orchestrator(l1_feature_dim=4, level_configs=configs)
    rng = np.random.default_rng(0)
    for _ in range(10):
        result = orch.step(rng.standard_normal(4))
    assert result["level2"] == {}
    assert orch._level_ticks[1] == 0
```

- [ ] **Step 2: Run tests to verify they all fail**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python -m pytest tests/agents/test_stacked.py -v
```
Expected: `ModuleNotFoundError` — `stacked.py` does not exist yet. All 17 tests fail.

- [ ] **Step 3: Implement `hpm/agents/stacked.py`**

```python
# hpm/agents/stacked.py
from __future__ import annotations
from dataclasses import dataclass, field
import pathlib
import numpy as np

from hpm.agents.agent import Agent
from hpm.agents.multi_agent import MultiAgentOrchestrator
from hpm.agents.hierarchical import extract_bundle, encode_bundle


@dataclass
class LevelConfig:
    """Configuration for one level in a StackedOrchestrator.

    n_agents: number of agents at this level
    pattern_type: pattern substrate ("gaussian", "laplace", etc.)
    K: cadence — how many times the level below must fire before this level fires.
       Ignored for level 0 (L1 always steps on every call).
    agent_ids: optional explicit IDs; auto-generated as "l{level+1}_{j}" if None.
    """
    n_agents: int
    pattern_type: str = "gaussian"
    K: int = 1
    agent_ids: list[str] | None = None


class StackedOrchestrator:
    """N-level abstraction stack.

    Level 0 (L1) processes raw observations every step.
    Level i fires every K[i] fires of level i-1 (cadence relative to level below).
    Each level receives N_prev separate step() calls per cadence tick (witness model).

    Attributes:
        level_orches: one MultiAgentOrchestrator per level
        level_agents: one list of Agent per level — public, for external inspection
        level_Ks: cadence values; level_Ks[0] = 1 (unused); level_Ks[i] = cadence for level i
        _level_ticks: how many times each level has fired; initialised to [0] * n_levels
    """

    def __init__(
        self,
        level_orches: list[MultiAgentOrchestrator],
        level_agents: list[list[Agent]],
        level_Ks: list[int],
    ):
        self.level_orches = level_orches
        self.level_agents = level_agents
        self.level_Ks = level_Ks
        self._level_ticks: list[int] = [0] * len(level_orches)

    def step(self, obs: np.ndarray) -> dict:
        """Step the hierarchy on one raw observation.

        Increment-then-check order (same as Sub-project 1 HierarchicalOrchestrator):
        each level's tick counter is incremented AFTER that level fires;
        the cadence check for level i+1 uses the updated _level_ticks[i]
        from the same step() call. First L2 cadence fires when _level_ticks[0] == K.

        Returns dict with keys "level1".."level{n}" and "t" (_level_ticks[0]).
        Deeper levels return {} on non-cadence steps.
        """
        n = len(self.level_orches)
        results: list[dict] = [{} for _ in range(n)]

        # Step level 0 (L1) — always fires
        l1_obs = {a.agent_id: obs for a in self.level_agents[0]}
        results[0] = self.level_orches[0].step(l1_obs)
        self._level_ticks[0] += 1

        # Step higher levels on cadence
        for i in range(1, n):
            if self._level_ticks[i - 1] % self.level_Ks[i] == 0:
                li_agent_ids = [a.agent_id for a in self.level_agents[i]]
                for prev_agent in self.level_agents[i - 1]:
                    bundle = extract_bundle(prev_agent)
                    encoded = encode_bundle(bundle)
                    # Witness model: all level-i agents receive the same bundle per call.
                    # N level-(i-1) agents → N separate step() calls to level_orches[i].
                    obs_dict = {aid: encoded for aid in li_agent_ids}
                    results[i] = self.level_orches[i].step(obs_dict)
                self._level_ticks[i] += 1

        out = {f"level{i + 1}": results[i] for i in range(n)}
        out["t"] = self._level_ticks[0]
        return out


def make_stacked_orchestrator(
    l1_feature_dim: int,
    level_configs: list[LevelConfig],
) -> tuple[StackedOrchestrator, list[list[Agent]]]:
    """Build an N-level StackedOrchestrator from a list of LevelConfig objects.

    Feature dimensions are computed automatically:
      level i has feature_dim = l1_feature_dim + 2 * i  (0-indexed)
      → L1=D, L2=D+2, L3=D+4, ...

    NOTE: level_configs[0].K is ignored. L1 always steps on every call.
    If level_configs[0].K != 1, it is silently treated as 1.

    Returns: (StackedOrchestrator, list_of_agent_lists_per_level)
    The second element is a convenience alias for orch.level_agents (same objects).
    """
    import sys
    _repo_root = str(pathlib.Path(__file__).resolve().parent.parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from benchmarks.multi_agent_common import make_orchestrator  # noqa: E402

    level_orches = []
    all_agents = []

    for i, cfg in enumerate(level_configs):
        feature_dim = l1_feature_dim + 2 * i
        agent_ids = cfg.agent_ids or [f"l{i + 1}_{j}" for j in range(cfg.n_agents)]
        orch, agents, _ = make_orchestrator(
            n_agents=cfg.n_agents,
            feature_dim=feature_dim,
            agent_ids=agent_ids,
            pattern_types=[cfg.pattern_type] * cfg.n_agents,
        )
        level_orches.append(orch)
        all_agents.append(agents)

    # level_Ks[0] = 1 (L1 always steps; config K is ignored for index 0)
    level_Ks = [1] + [cfg.K for cfg in level_configs[1:]]

    stacked = StackedOrchestrator(level_orches, all_agents, level_Ks)
    return stacked, all_agents
```

- [ ] **Step 4: Run all 17 tests to verify they pass**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python -m pytest tests/agents/test_stacked.py -v
```
Expected: all 17 tests PASS. If any cadence test fails, debug `StackedOrchestrator.step()`.

- [ ] **Step 5: Run full test suite to catch regressions**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python -m pytest tests/ --tb=short -q
```
Expected: all existing tests + 17 new tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && git add hpm/agents/stacked.py tests/agents/test_stacked.py && git commit -m "feat: add LevelConfig, StackedOrchestrator, make_stacked_orchestrator"
```

---

### Task 2: __init__ re-export

**Files:**
- Modify: `hpm/agents/__init__.py`

- [ ] **Step 1: Read current `hpm/agents/__init__.py`**

```bash
cat /home/mattthomson/workspace/HPM---Learning-Agent/hpm/agents/__init__.py
```

- [ ] **Step 2: Add re-exports**

Add to `hpm/agents/__init__.py` (after the existing hierarchical imports):

```python
from .stacked import (
    LevelConfig,
    StackedOrchestrator,
    make_stacked_orchestrator,
)
```

Also add to `__all__`:
```python
"LevelConfig",
"StackedOrchestrator",
"make_stacked_orchestrator",
```

- [ ] **Step 3: Verify imports work**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python -c "from hpm.agents import LevelConfig, StackedOrchestrator, make_stacked_orchestrator; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Run full test suite**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python -m pytest tests/ --tb=short -q
```

- [ ] **Step 5: Commit**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && git add hpm/agents/__init__.py && git commit -m "feat: re-export StackedOrchestrator symbols from hpm.agents"
```

---

### Task 3: Hierarchical ARC benchmark

**Files:**
- Create: `benchmarks/hierarchical_arc.py`

**Before starting:** Read `benchmarks/multi_agent_arc.py` to understand:
- `encode_pair(input_grid, output_grid)` — encodes transformation delta via fixed projection → 64-dim vec
- `ensemble_score(agents, vec)` — takes list of agents + vec, returns weighted NLL (lower = more probable)
- `load_tasks()`, `task_fits(task)`, `TRAIN_REPS = 10`, `N_DISTRACTORS = 4`

**Scoring design:** L3 agents operate in 68-dim bundle space and cannot directly score 64-dim raw ARC vectors. Scoring uses L1 agents (feature_dim=64) via `ensemble_score(l1_agents, encoded_vec)`. The hypothesis being tested: does training L1 agents inside a 3-level hierarchical stack produce better 64-dim representations than training a flat single agent on the same data?

**Task cap:** Take the first 400 tasks after loading (matching `multi_agent_arc.py`'s pattern). After applying `task_fits`, ~342 tasks remain — this is the comparison baseline.

- [ ] **Step 1: Write `benchmarks/hierarchical_arc.py`**

```python
"""
Hierarchical ARC Benchmark
===========================
Tests whether a 3-level StackedOrchestrator shapes L1 agent representations
better than a flat single-agent baseline on ARC discrimination tasks.

Stack config:
  L1: 2 agents, GaussianPattern, feature_dim=64, K=1 (always steps)
  L2: 2 agents, GaussianPattern, feature_dim=66, K=3 (fires every 3 L1 steps)
  L3: 1 agent,  GaussianPattern, feature_dim=68, K=3 (fires every 9 L1 steps)

Scoring: L1 agents (dim=64) score candidates via ensemble_score, same as the
flat baseline. L3 agents operate in 68-dim bundle space and cannot score 64-dim
raw ARC vectors directly. This benchmark tests whether hierarchical pressure
from L2/L3 improves L1 representation quality.

Both hierarchical and flat orchestrators are reset per task (same as multi_agent_arc.py).

Run:
    python benchmarks/hierarchical_arc.py

Note: Downloads ARC dataset from HuggingFace on first run (~5MB, cached).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from benchmarks.multi_agent_arc import (
    encode_pair, ensemble_score, load_tasks, task_fits, TRAIN_REPS, N_DISTRACTORS,
)
from benchmarks.multi_agent_common import make_orchestrator, print_results_table
from hpm.agents.stacked import LevelConfig, make_stacked_orchestrator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
L1_FEATURE_DIM = 64
STACK_CONFIGS = [
    LevelConfig(n_agents=2, K=1),   # L1: 64-dim
    LevelConfig(n_agents=2, K=3),   # L2: 66-dim, fires every 3 L1 steps
    LevelConfig(n_agents=1, K=3),   # L3: 68-dim, fires every 9 L1 steps
]
# With TRAIN_REPS=10 per pair and K_L2=3: L2 fires ~3 times per pair.
# With K_L3=3: L3 fires once per 9 L1 steps → fires at step 9 of training.
# For tasks with ≥1 training pair (all ARC tasks), L3 fires at least once.


def _make_flat_orchestrator():
    """Single flat agent baseline (comparable to arc_benchmark.py single agent)."""
    return make_orchestrator(
        n_agents=1,
        feature_dim=L1_FEATURE_DIM,
        agent_ids=["arc_flat"],
        pattern_types=["gaussian"],
    )


def run() -> dict:
    tasks = load_tasks()
    tasks = [t for t in tasks if task_fits(t)][:400]

    rng = np.random.default_rng(42)

    hierarchical_correct = 0
    flat_correct = 0
    n_evaluated = 0

    for i, task in enumerate(tasks):
        # Encode correct test output
        test_pair = task["test"][0]
        correct_vec = encode_pair(test_pair["input"], test_pair["output"])

        # Pick N_DISTRACTORS from other tasks
        distractor_idxs = [j for j in range(len(tasks)) if j != i]
        chosen = rng.choice(distractor_idxs, size=N_DISTRACTORS, replace=False)
        distractor_vecs = [
            encode_pair(tasks[di]["test"][0]["input"], tasks[di]["test"][0]["output"])
            for di in chosen
        ]
        candidates = [correct_vec] + distractor_vecs

        # --- Hierarchical ---
        stacked_orch, all_agents = make_stacked_orchestrator(
            l1_feature_dim=L1_FEATURE_DIM,
            level_configs=STACK_CONFIGS,
        )
        l1_agents = all_agents[0]

        for pair in task["train"]:
            vec = encode_pair(pair["input"], pair["output"])
            for _ in range(TRAIN_REPS):
                stacked_orch.step(vec)

        # Guard: L3 must have fired at least once for meaningful training
        if stacked_orch._level_ticks[-1] == 0:
            raise RuntimeError(
                f"L3 never fired on task {i} ({len(task['train'])} train pairs, "
                f"TRAIN_REPS={TRAIN_REPS}). Reduce K values or increase TRAIN_REPS."
            )

        # Score with L1 agents (dim=64 matches candidate vecs)
        scores_h = [ensemble_score(l1_agents, c) for c in candidates]
        if scores_h[0] == min(scores_h):  # lower NLL = more probable = correct
            hierarchical_correct += 1

        # --- Flat baseline ---
        flat_orch, flat_agents, _ = _make_flat_orchestrator()
        for pair in task["train"]:
            vec = encode_pair(pair["input"], pair["output"])
            for _ in range(TRAIN_REPS):
                flat_orch.step({flat_agents[0].agent_id: vec})

        scores_f = [ensemble_score(flat_agents, c) for c in candidates]
        if scores_f[0] == min(scores_f):
            flat_correct += 1

        n_evaluated += 1

    return {
        "n_tasks": n_evaluated,
        "hierarchical_correct": hierarchical_correct,
        "flat_correct": flat_correct,
        "hierarchical_acc": hierarchical_correct / n_evaluated if n_evaluated else 0.0,
        "flat_acc": flat_correct / n_evaluated if n_evaluated else 0.0,
        "chance": 1.0 / (N_DISTRACTORS + 1),
    }


def main():
    cfg_str = (
        f"3-level stack "
        f"(L1×2=64, L2×2=66 K={STACK_CONFIGS[1].K}, L3×1=68 K={STACK_CONFIGS[2].K})"
    )
    print(f"Running Hierarchical ARC Benchmark ({cfg_str})...")
    m = run()

    h_vs_chance = m["hierarchical_acc"] - m["chance"]
    f_vs_chance = m["flat_acc"] - m["chance"]
    h_vs_flat = m["hierarchical_acc"] - m["flat_acc"]

    print_results_table(
        title=f"Hierarchical ARC Benchmark ({m['n_tasks']} tasks, scored via L1 agents)",
        cols=["Setup", "Accuracy", "vs Chance", "vs Flat"],
        rows=[
            {
                "Setup": "Flat single-agent",
                "Accuracy": f"{m['flat_acc']:.1%}",
                "vs Chance": f"{f_vs_chance:+.1%}",
                "vs Flat": "—",
            },
            {
                "Setup": cfg_str,
                "Accuracy": f"{m['hierarchical_acc']:.1%}",
                "vs Chance": f"{h_vs_chance:+.1%}",
                "vs Flat": f"{h_vs_flat:+.1%}",
            },
        ],
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python benchmarks/hierarchical_arc.py
```
Expected: runs without error; L3 guard does not trigger; both accuracy values are finite; output table printed.

If the L3 guard triggers: reduce `STACK_CONFIGS[1].K` or `STACK_CONFIGS[2].K` to 2 and re-run.

- [ ] **Step 3: Run full test suite**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && python -m pytest tests/ --tb=short -q
```

- [ ] **Step 4: Commit and push**

```bash
cd /home/mattthomson/workspace/HPM---Learning-Agent && git add benchmarks/hierarchical_arc.py && git commit -m "feat: add hierarchical ARC benchmark (3-level stacked vs flat)"
git push
```

---

## Success criteria

- All 17 unit tests in `test_stacked.py` pass
- `from hpm.agents import LevelConfig, StackedOrchestrator, make_stacked_orchestrator` works
- `hierarchical_arc.py` runs without error; L3 guard does not trigger
- Both flat and hierarchical accuracy values are finite and above 0%
- Full test suite passes with no regressions
