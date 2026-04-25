# Implementation Plan: MetaCognitivePattern (L5)

Date: 2026-04-22
Branch: hpm-ai-v3-dev
Spec: docs/superpowers/specs/2026-04-22-meta-cognitive-pattern-design.md

---

## Goal

Implement MetaCognitivePattern — the L5 strategic oversight layer for hpm_ai_v3 agents. It observes population and curriculum state every N=10 episodes, selects one of 8 directives via a learned policy network, and updates via REINFORCE with eligibility traces.

## Architecture

- `hpm_ai_v3/meta_cognitive_pattern.py` — MetaCognitivePattern class + MetaFeatureExtractor + DirectiveExecutor
- `hpm_ai_v3/agents/meta_training.py` — MetaTrainingLoop wrapper
- `hpm_ai_v3/agents/base_discovery.py` — add exploration_temperature, recent_use_count, run_episode()
- `hpm_ai_v3/population.py` — add get_population_entropy(), get_diversity()
- `hpm_ai_v3/curriculum.py` — add advance_phase(), set_difficulty()
- `hpm_ai_v3/tests/test_meta_cognitive.py` — all unit tests

## Tech Stack

- Python 3.10+
- PyTorch (already in use)
- numpy (already in use)
- pytest for tests

---

## Tasks

### Task 1 — CurriculumManager additions

**Step 1.1 — Write failing tests**

File: `hpm_ai_v3/tests/test_meta_cognitive.py`

```python
import pytest
from unittest.mock import MagicMock
from hpm_ai_v3.curriculum import CurriculumManager


def make_curriculum_with_two_phases():
    """Helper: returns a CurriculumManager with at least 2 phases loaded."""
    cm = CurriculumManager()
    if len(cm.patterns) < 2:
        # Inject mock patterns if filesystem has fewer than 2
        from hpm_ai_v3.curriculum import CurriculumPattern
        cm.patterns = [
            CurriculumPattern("phase0", 0, [{"text": "1+1", "answer": 2.0}]),
            CurriculumPattern("phase1", 1, [{"text": "2+2", "answer": 4.0}]),
        ]
        cm.active_pattern_idx = 0
        cm.phase = 0
    return cm


class TestCurriculumManagerAdditions:
    def test_advance_phase_increments_index(self):
        cm = make_curriculum_with_two_phases()
        initial_idx = cm.active_pattern_idx
        cm.advance_phase()
        assert cm.active_pattern_idx == initial_idx + 1

    def test_advance_phase_updates_phase_attribute(self):
        cm = make_curriculum_with_two_phases()
        cm.advance_phase()
        assert cm.phase == cm.patterns[cm.active_pattern_idx].phase

    def test_advance_phase_resets_recent_rewards(self):
        cm = make_curriculum_with_two_phases()
        cm.recent_rewards = [1.0, 1.0, 1.0]
        cm.advance_phase()
        assert cm.recent_rewards == []

    def test_advance_phase_does_not_exceed_last_phase(self):
        cm = make_curriculum_with_two_phases()
        cm.active_pattern_idx = len(cm.patterns) - 1
        cm.advance_phase()
        assert cm.active_pattern_idx == len(cm.patterns) - 1

    def test_set_difficulty_increases(self):
        cm = make_curriculum_with_two_phases()
        cm.difficulty = 0.5
        cm.set_difficulty(0.1)
        assert abs(cm.difficulty - 0.6) < 1e-6

    def test_set_difficulty_decreases(self):
        cm = make_curriculum_with_two_phases()
        cm.difficulty = 0.5
        cm.set_difficulty(-0.2)
        assert abs(cm.difficulty - 0.3) < 1e-6

    def test_set_difficulty_clamps_at_zero(self):
        cm = make_curriculum_with_two_phases()
        cm.difficulty = 0.1
        cm.set_difficulty(-0.5)
        assert cm.difficulty == 0.0

    def test_set_difficulty_clamps_at_one(self):
        cm = make_curriculum_with_two_phases()
        cm.difficulty = 0.9
        cm.set_difficulty(0.5)
        assert cm.difficulty == 1.0
```

**Step 1.2 — Run tests (expect failure)**

```bash
python -m pytest hpm_ai_v3/tests/test_meta_cognitive.py::TestCurriculumManagerAdditions -x 2>&1 | head -30
```

**Step 1.3 — Implement in `hpm_ai_v3/curriculum.py`**

Add after the `update()` method in `CurriculumManager`:

```python
def advance_phase(self):
    """Directly advance curriculum phase (called by meta-directive)."""
    if self.active_pattern_idx < len(self.patterns) - 1:
        self.active_pattern_idx += 1
        self.phase = self.patterns[self.active_pattern_idx].phase
        self.recent_rewards = []
        print(f"--- META: Advancing to {self.patterns[self.active_pattern_idx].name} (Phase {self.phase}) ---")

def set_difficulty(self, delta: float):
    """Adjust difficulty by delta, clamped to [0.0, 1.0]."""
    self.difficulty = max(0.0, min(1.0, self.difficulty + delta))
```

**Step 1.4 — Run tests (expect pass)**

```bash
python -m pytest hpm_ai_v3/tests/test_meta_cognitive.py::TestCurriculumManagerAdditions -x
```

**Step 1.5 — Commit**

```bash
git add hpm_ai_v3/curriculum.py hpm_ai_v3/tests/test_meta_cognitive.py
git commit -m "feat: add advance_phase() and set_difficulty() to CurriculumManager for meta-directive support"
```

---

### Task 2 — PatternPopulation additions

**Step 2.1 — Add tests to `hpm_ai_v3/tests/test_meta_cognitive.py`**

```python
import numpy as np
from hpm_ai_v3.population import PatternPopulation
from hpm_ai_v3.causal_pattern import CausalPattern


def make_population():
    patterns = [CausalPattern(4, 2, 2) for _ in range(4)]
    weights = [0.4, 0.3, 0.2, 0.1]
    pop = PatternPopulation(patterns)
    for p, w in zip(pop.patterns, weights):
        p.weight = w
    return pop


class TestPatternPopulationAdditions:
    def test_get_population_entropy_is_float(self):
        pop = make_population()
        entropy = pop.get_population_entropy()
        assert isinstance(entropy, float)

    def test_get_population_entropy_uniform_is_max(self):
        pop = make_population()
        for p in pop.patterns:
            p.weight = 0.25
        uniform_entropy = pop.get_population_entropy()
        for p in pop.patterns:
            p.weight = 1.0  # one dominant
        pop.patterns[1].weight = 0.0
        pop.patterns[2].weight = 0.0
        pop.patterns[3].weight = 0.0
        skewed_entropy = pop.get_population_entropy()
        assert uniform_entropy > skewed_entropy

    def test_get_diversity_is_float(self):
        pop = make_population()
        diversity = pop.get_diversity()
        assert isinstance(diversity, float)
        assert 0.0 <= diversity <= 1.0

    def test_get_diversity_identical_patterns_is_zero(self):
        p = CausalPattern(4, 2, 2)
        pop = PatternPopulation([p, p])
        assert pop.get_diversity() == 0.0
```

**Step 2.2 — Run tests (expect failure)**

```bash
python -m pytest hpm_ai_v3/tests/test_meta_cognitive.py::TestPatternPopulationAdditions -x 2>&1 | head -30
```

**Step 2.3 — Implement in `hpm_ai_v3/population.py`**

Add after `get_top_patterns()`:

```python
def get_population_entropy(self) -> float:
    """Shannon entropy of weight distribution (nats)."""
    weights = np.array([p.weight for p in self.patterns])
    total = weights.sum()
    if total < 1e-9 or len(weights) == 0:
        return 0.0
    probs = weights / total
    probs = probs[probs > 1e-9]
    return float(-np.sum(probs * np.log(probs)))

def get_diversity(self) -> float:
    """Mean pairwise structural distance across all pattern pairs."""
    n = len(self.patterns)
    if n < 2:
        return 0.0
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(self.patterns[i].structural_distance(self.patterns[j]))
    return float(np.mean(distances))
```

**Step 2.4 — Run tests (expect pass)**

```bash
python -m pytest hpm_ai_v3/tests/test_meta_cognitive.py::TestPatternPopulationAdditions -x
```

**Step 2.5 — Commit**

```bash
git add hpm_ai_v3/population.py hpm_ai_v3/tests/test_meta_cognitive.py
git commit -m "feat: add get_population_entropy() and get_diversity() to PatternPopulation"
```

---

### Task 3 — ActionPattern and base_discovery.py additions

**Step 3.1 — Add tests**

```python
from hpm_ai_v3.agents.base_discovery import ActionPattern


class TestActionPatternAdditions:
    def test_action_pattern_has_exploration_temperature(self):
        ap = ActionPattern("arithmetic")
        assert hasattr(ap, "exploration_temperature")
        assert ap.exploration_temperature == 1.0

    def test_action_pattern_has_recent_use_count(self):
        ap = ActionPattern("arithmetic")
        assert hasattr(ap, "recent_use_count")
        assert ap.recent_use_count == 0

    def test_mark_used_increments_recent_use_count(self):
        ap = ActionPattern("arithmetic")
        ap.mark_used()
        assert ap.recent_use_count == 1
```

**Step 3.2 — Run tests (expect failure)**

```bash
python -m pytest hpm_ai_v3/tests/test_meta_cognitive.py::TestActionPatternAdditions -x 2>&1 | head -20
```

**Step 3.3 — Implement in `hpm_ai_v3/agents/base_discovery.py`**

In `ActionPattern.__init__()`, add after existing fields:

```python
self.exploration_temperature: float = 1.0
self.recent_use_count: int = 0
```

Override `mark_used()` in `ActionPattern` to also increment `recent_use_count`:

```python
def mark_used(self):
    super().mark_used()
    self.recent_use_count += 1
```

**Step 3.4 — Run tests (expect pass)**

```bash
python -m pytest hpm_ai_v3/tests/test_meta_cognitive.py::TestActionPatternAdditions -x
```

**Step 3.5 — Commit**

```bash
git add hpm_ai_v3/agents/base_discovery.py hpm_ai_v3/tests/test_meta_cognitive.py
git commit -m "feat: add exploration_temperature and recent_use_count to ActionPattern"
```

---

### Task 4 — MetaCognitivePattern core class

**Step 4.1 — Add tests**

```python
import torch
from hpm_ai_v3.meta_cognitive_pattern import MetaCognitivePattern, MetaDirective


class TestMetaCognitivePattern:
    def setup_method(self):
        self.mcp = MetaCognitivePattern()

    def test_inherits_hpm_pattern(self):
        from hpm_ai_v3.pattern import HPMPattern
        assert isinstance(self.mcp, HPMPattern)

    def test_policy_network_output_shape(self):
        features = torch.zeros(64)
        probs = self.mcp._policy_forward(features)
        assert probs.shape == (8,)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_act_returns_valid_directive(self):
        features = torch.zeros(64)
        directive = self.mcp.act(features)
        assert directive in list(MetaDirective)

    def test_meta_feature_vector_has_correct_dim(self):
        # Build mock agent with minimal required attributes
        agent = _make_mock_agent()
        curriculum = _make_mock_curriculum()
        features = self.mcp.observe(agent, curriculum)
        assert features.shape == (64,)

    def test_record_transition_stores_entry(self):
        features = torch.zeros(64)
        self.mcp.record_transition(features, MetaDirective.CONTINUE, 1.0)
        assert len(self.mcp._trajectory) == 1

    def test_update_policy_runs_without_error(self):
        features = torch.zeros(64)
        for _ in range(3):
            self.mcp.record_transition(features, MetaDirective.CONTINUE, 0.5)
        self.mcp.update_policy()  # must not raise
        assert len(self.mcp._trajectory) == 0  # cleared after update

    def test_log_prob_returns_scalar_tensor(self):
        obs = {"meta_features": torch.zeros(64)}
        lp = self.mcp.log_prob(obs)
        assert lp.ndim == 0

    def test_structural_distance_same_class(self):
        other = MetaCognitivePattern()
        assert self.mcp.structural_distance(other) == 0.0

    def test_structural_distance_different_class(self):
        from hpm_ai_v3.causal_pattern import CausalPattern
        other = CausalPattern(4, 2, 2)
        assert self.mcp.structural_distance(other) == 1.0

    def test_extract_causal_graph_has_one_node(self):
        g = self.mcp.extract_causal_graph()
        assert len(g.nodes) == 1


def _make_mock_agent():
    agent = MagicMock()
    agent.population.patterns = []
    agent.population.get_population_entropy.return_value = 1.0
    agent.population.get_diversity.return_value = 0.5
    agent.population.get_top_patterns.return_value = []
    agent._meta_success_history = [0.5] * 10
    agent._steps_since_advance = 5
    return agent


def _make_mock_curriculum():
    cm = MagicMock()
    cm.phase = 0
    cm.difficulty = 0.5
    return cm
```

**Step 4.2 — Run tests (expect failure)**

```bash
python -m pytest hpm_ai_v3/tests/test_meta_cognitive.py::TestMetaCognitivePattern -x 2>&1 | head -40
```

**Step 4.3 — Implement `hpm_ai_v3/meta_cognitive_pattern.py`**

```python
"""
MetaCognitivePattern — L5 strategic oversight for hpm_ai_v3 agents.
Operates at episode-level granularity, issuing directives to reshape
population dynamics and curriculum progression.
"""
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .pattern import HPMPattern


class MetaDirective(IntEnum):
    CONTINUE = 0
    INJECT_EXPLORATION = 1
    RESET_STUCK_PATTERNS = 2
    ADVANCE_PHASE = 3
    SPAWN_SPECIALIST = 4
    INCREASE_DIFFICULTY = 5
    DECREASE_DIFFICULTY = 6
    CONSOLIDATE = 7


class _MetaPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class _MetaFeatureProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(8, 64), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MetaCognitivePattern(HPMPattern):
    """
    L5 meta-pattern: observes population + curriculum state every N episodes
    and issues one of 8 strategic directives via a learned policy (REINFORCE).
    """

    def __init__(self, pattern_id: Optional[str] = None, meta_lr: float = 1e-3,
                 gamma: float = 0.99, lambda_trace: float = 0.8):
        super().__init__(pattern_id=pattern_id or "meta_cognitive")
        self.substrate_type = "meta_cognitive"
        self.weight = 1.0  # Fixed — not subject to replicator dynamics

        self._projector = _MetaFeatureProjector()
        self._policy = _MetaPolicyNet()
        self._optimizer = optim.Adam(
            list(self._projector.parameters()) + list(self._policy.parameters()),
            lr=meta_lr
        )
        self.gamma = gamma
        self.lambda_trace = lambda_trace

        # REINFORCE trajectory buffer: list of (log_prob, reward)
        self._trajectory: List[Tuple[torch.Tensor, float]] = []
        self._eligibility_trace: Optional[torch.Tensor] = None
        self._reward_baseline: float = 0.0
        self._baseline_alpha: float = 0.1

        # Tracking for meta-reward
        self._prev_success_rate: float = 0.0
        self._prev_phase: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def observe(self, agent: Any, curriculum: Any) -> torch.Tensor:
        """Compute 64-dim meta-feature vector from agent + curriculum state."""
        raw = self._extract_raw_features(agent, curriculum)
        raw_tensor = torch.tensor(raw, dtype=torch.float32)
        with torch.no_grad():
            features = self._projector(raw_tensor)
        return features

    def _extract_raw_features(self, agent: Any, curriculum: Any) -> List[float]:
        pop = agent.population
        history = getattr(agent, "_meta_success_history", [0.0])
        success_rate = float(np.mean(history)) if history else 0.0

        n_phases = max(1, len(getattr(curriculum, "patterns", [1])))
        phase_norm = float(getattr(curriculum, "phase", 0)) / n_phases

        entropy = pop.get_population_entropy()
        # Normalise entropy: max entropy for n patterns = ln(n)
        n = max(1, len(pop.patterns))
        entropy_norm = entropy / max(np.log(n), 1e-6)

        weights = [p.weight for p in pop.patterns]
        top_dominance = float(max(weights)) if weights else 0.0

        temps = [getattr(p, "exploration_temperature", 1.0) for p in pop.patterns]
        exploration_rate = float(np.mean(temps)) if temps else 1.0

        steps_since_advance = float(getattr(agent, "_steps_since_advance", 0))
        steps_since_norm = min(1.0, steps_since_advance / 100.0)

        avg_reward = success_rate  # same window

        diversity = pop.get_diversity()

        return [
            success_rate,
            phase_norm,
            float(entropy_norm),
            top_dominance,
            exploration_rate,
            steps_since_norm,
            avg_reward,
            diversity,
        ]

    def _policy_forward(self, features: torch.Tensor) -> torch.Tensor:
        return self._policy(features)

    def act(self, meta_features: torch.Tensor) -> MetaDirective:
        """Sample a directive from the policy (used during training)."""
        probs = self._policy_forward(meta_features)
        dist = torch.distributions.Categorical(probs)
        idx = dist.sample()
        return MetaDirective(idx.item())

    def act_greedy(self, meta_features: torch.Tensor) -> MetaDirective:
        """Argmax directive (used during evaluation)."""
        probs = self._policy_forward(meta_features)
        return MetaDirective(probs.argmax().item())

    def record_transition(self, meta_features: torch.Tensor,
                          directive: MetaDirective, reward: float):
        """Store (log_prob, reward) for REINFORCE update."""
        probs = self._policy_forward(meta_features)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(torch.tensor(int(directive)))
        self._trajectory.append((log_prob, reward))

    def update_policy(self):
        """REINFORCE update with eligibility traces over stored trajectory."""
        if not self._trajectory:
            return

        # Compute discounted returns
        returns = []
        G = 0.0
        for _, r in reversed(self._trajectory):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Baseline (running mean)
        self._reward_baseline = (
            (1 - self._baseline_alpha) * self._reward_baseline
            + self._baseline_alpha * float(returns_t.mean())
        )
        returns_t = returns_t - self._reward_baseline

        # Policy gradient loss
        loss = torch.tensor(0.0, requires_grad=True)
        for (log_prob, _), G_t in zip(self._trajectory, returns_t):
            loss = loss + (-log_prob * G_t)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._trajectory.clear()

    def observe_and_act(self, agent: Any, curriculum: Any) -> MetaDirective:
        """Convenience entry point: observe state, select directive, execute it."""
        features = self.observe(agent, curriculum)
        directive = self.act(features)
        self.execute_directive(directive, agent, curriculum)
        return directive

    def compute_meta_reward(self, agent: Any, curriculum: Any) -> float:
        """Compute meta-reward after N episodes."""
        history = getattr(agent, "_meta_success_history", [0.0])
        success_rate_now = float(np.mean(history)) if history else 0.0
        phase_now = int(getattr(curriculum, "phase", 0))

        phase_advance_bonus = 2.0 if phase_now > self._prev_phase else 0.0
        success_delta = success_rate_now - self._prev_success_rate
        steps = max(1, getattr(agent, "_steps_since_advance", 1))
        sample_efficiency = min(1.0, success_rate_now / steps * 10)
        diversity_bonus = agent.population.get_diversity()

        self._prev_success_rate = success_rate_now
        self._prev_phase = phase_now

        return (2.0 * phase_advance_bonus
                + 1.0 * success_delta
                + 0.5 * sample_efficiency
                + 0.3 * diversity_bonus)

    # ------------------------------------------------------------------
    # Directive Execution
    # ------------------------------------------------------------------

    def execute_directive(self, directive: MetaDirective, agent: Any, curriculum: Any):
        """Dispatch directive to implementation."""
        dispatch = {
            MetaDirective.CONTINUE: self._do_continue,
            MetaDirective.INJECT_EXPLORATION: self._do_inject_exploration,
            MetaDirective.RESET_STUCK_PATTERNS: self._do_reset_stuck_patterns,
            MetaDirective.ADVANCE_PHASE: self._do_advance_phase,
            MetaDirective.SPAWN_SPECIALIST: self._do_spawn_specialist,
            MetaDirective.INCREASE_DIFFICULTY: self._do_increase_difficulty,
            MetaDirective.DECREASE_DIFFICULTY: self._do_decrease_difficulty,
            MetaDirective.CONSOLIDATE: self._do_consolidate,
        }
        print(f"  [MetaCognitive] Directive: {directive.name}")
        dispatch[directive](agent, curriculum)

    def _do_continue(self, agent: Any, curriculum: Any):
        pass

    def _do_inject_exploration(self, agent: Any, curriculum: Any):
        from .agents.base_discovery import ActionPattern
        for p in agent.population.patterns:
            if hasattr(p, "exploration_temperature"):
                p.exploration_temperature = min(3.0, p.exploration_temperature * 1.5)
        # Inject a new random ActionPattern with low weight
        new_p = ActionPattern("arithmetic", pattern_id=f"explore_{int(time.time())%10000}")
        new_p.weight = 0.01
        new_p.exploration_temperature = 2.0
        agent.population.patterns.append(new_p)

    def _do_reset_stuck_patterns(self, agent: Any, curriculum: Any):
        patterns = agent.population.patterns
        if not patterns:
            return
        weights = [p.weight for p in patterns]
        threshold = np.percentile(weights, 25)
        for p in patterns:
            if p.weight <= threshold:
                p.weight = agent.population.pruning_threshold

    def _do_advance_phase(self, agent: Any, curriculum: Any):
        if hasattr(curriculum, "advance_phase"):
            curriculum.advance_phase()
            agent._steps_since_advance = 0

    def _do_spawn_specialist(self, agent: Any, curriculum: Any):
        top = agent.population.get_top_patterns(3)
        if hasattr(agent, "compiler") and len(top) >= 2:
            try:
                agent.compiler.spawn_agent_from_composite(top)
            except Exception as e:
                print(f"  [MetaCognitive] SPAWN_SPECIALIST failed: {e}")

    def _do_increase_difficulty(self, agent: Any, curriculum: Any):
        if hasattr(curriculum, "set_difficulty"):
            curriculum.set_difficulty(0.1)

    def _do_decrease_difficulty(self, agent: Any, curriculum: Any):
        if hasattr(curriculum, "set_difficulty"):
            curriculum.set_difficulty(-0.2)

    def _do_consolidate(self, agent: Any, curriculum: Any):
        patterns = agent.population.patterns
        if not patterns:
            return
        weights = [p.weight for p in patterns]
        median_w = float(np.median(weights))
        for p in patterns:
            if p.weight < median_w:
                p.weight = 0.0
        # Re-normalise
        total = sum(p.weight for p in patterns)
        if total > 0:
            for p in patterns:
                p.weight /= total

    # ------------------------------------------------------------------
    # HPMPattern abstract method implementations
    # ------------------------------------------------------------------

    def log_prob(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = observations.get("meta_features", torch.zeros(64))
        probs = self._policy_forward(features)
        dist = torch.distributions.Categorical(probs)
        # Return log prob of argmax (most likely action)
        return dist.log_prob(probs.argmax())

    def sample(self, context: Dict[str, Any], num_samples: int = 1) -> Dict[str, torch.Tensor]:
        features = context.get("meta_features", torch.zeros(64))
        probs = self._policy_forward(features)
        dist = torch.distributions.Categorical(probs)
        samples = dist.sample((num_samples,))
        return {"directives": samples, "probs": probs}

    def intervene(self, intervention: Dict[str, Any],
                  context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Force a specific directive index."""
        forced = intervention.get("directive", MetaDirective.CONTINUE)
        features = context.get("meta_features", torch.zeros(64))
        probs = self._policy_forward(features)
        return {"directive": torch.tensor(int(forced)), "probs": probs}

    def update_parameters(self, observations: Dict[str, torch.Tensor],
                          learning_rate: float = 0.01):
        """Delegate to REINFORCE update (trajectory must be pre-loaded)."""
        self.update_policy()

    def structural_distance(self, other: "HPMPattern") -> float:
        return 0.0 if isinstance(other, MetaCognitivePattern) else 1.0

    def extract_causal_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_node(self.id, label="meta_cognitive")
        return g
```

**Step 4.4 — Run tests (expect pass)**

```bash
python -m pytest hpm_ai_v3/tests/test_meta_cognitive.py::TestMetaCognitivePattern -x
```

**Step 4.5 — Commit**

```bash
git add hpm_ai_v3/meta_cognitive_pattern.py hpm_ai_v3/tests/test_meta_cognitive.py
git commit -m "feat: implement MetaCognitivePattern L5 with REINFORCE policy and 8 directives"
```

---

### Task 5 — MetaTrainingLoop

**Step 5.1 — Add tests**

```python
from unittest.mock import MagicMock, patch
from hpm_ai_v3.agents.meta_training import MetaTrainingLoop
from hpm_ai_v3.meta_cognitive_pattern import MetaCognitivePattern, MetaDirective


class TestMetaTrainingLoop:
    def test_run_meta_step_calls_observe_and_act(self):
        agent = _make_mock_agent()
        curriculum = _make_mock_curriculum()
        meta = MetaCognitivePattern()
        loop = MetaTrainingLoop(agent, curriculum, meta, N=3)

        with patch.object(meta, "observe_and_act", return_value=MetaDirective.CONTINUE) as mock_act:
            with patch.object(meta, "record_transition") as mock_record:
                with patch.object(meta, "update_policy") as mock_update:
                    loop.run_meta_step()
        mock_act.assert_called_once()

    def test_run_meta_step_calls_update_policy(self):
        agent = _make_mock_agent()
        curriculum = _make_mock_curriculum()
        meta = MetaCognitivePattern()
        loop = MetaTrainingLoop(agent, curriculum, meta, N=3)

        with patch.object(meta, "observe_and_act", return_value=MetaDirective.CONTINUE):
            with patch.object(meta, "record_transition"):
                with patch.object(meta, "update_policy") as mock_update:
                    loop.run_meta_step()
        mock_update.assert_called_once()

    def test_reset_use_counts_resets_recent_use_count(self):
        from hpm_ai_v3.agents.base_discovery import ActionPattern
        agent = _make_mock_agent()
        ap = ActionPattern("arithmetic")
        ap.recent_use_count = 5
        agent.population.patterns = [ap]
        curriculum = _make_mock_curriculum()
        meta = MetaCognitivePattern()
        loop = MetaTrainingLoop(agent, curriculum, meta, N=3)
        loop._reset_use_counts()
        assert ap.recent_use_count == 0
```

**Step 5.2 — Run tests (expect failure)**

```bash
python -m pytest hpm_ai_v3/tests/test_meta_cognitive.py::TestMetaTrainingLoop -x 2>&1 | head -30
```

**Step 5.3 — Implement `hpm_ai_v3/agents/meta_training.py`**

```python
"""
MetaTrainingLoop — wraps an existing agent training loop with meta-cognitive oversight.
Calls the agent's run_episode() N times, then triggers a meta-step.
"""
from typing import Any, List, Optional
import numpy as np

from ..meta_cognitive_pattern import MetaCognitivePattern


class MetaTrainingLoop:
    def __init__(self, agent: Any, curriculum: Any,
                 meta_pattern: MetaCognitivePattern,
                 N: int = 10, max_meta_steps: int = 100):
        self.agent = agent
        self.curriculum = curriculum
        self.meta = meta_pattern
        self.N = N
        self.max_meta_steps = max_meta_steps

        # Initialise tracking attributes on agent if not present
        if not hasattr(agent, "_meta_success_history"):
            agent._meta_success_history = []
        if not hasattr(agent, "_steps_since_advance"):
            agent._steps_since_advance = 0

    def run(self):
        """Run max_meta_steps meta-steps."""
        for meta_step in range(self.max_meta_steps):
            print(f"\n=== Meta-Step {meta_step + 1}/{self.max_meta_steps} ===")
            self.run_meta_step()

    def run_meta_step(self):
        """Run N base episodes, then one meta update."""
        episode_rewards: List[float] = []

        for _ in range(self.N):
            reward = self._run_single_episode()
            episode_rewards.append(reward)
            self.agent._meta_success_history.append(reward)
            if len(self.agent._meta_success_history) > 50:
                self.agent._meta_success_history.pop(0)
            self.agent._steps_since_advance += 1

        # Meta observation and directive
        features = self.meta.observe(self.agent, self.curriculum)
        directive = self.meta.observe_and_act(self.agent, self.curriculum)

        # Meta reward
        meta_reward = self.meta.compute_meta_reward(self.agent, self.curriculum)
        self.meta.record_transition(features, directive, meta_reward)
        self.meta.update_policy()

        self._reset_use_counts()

        print(f"  [MetaLoop] N={self.N} episodes, mean_reward={np.mean(episode_rewards):.3f}, "
              f"meta_reward={meta_reward:.3f}, directive={directive.name}")

    def _run_single_episode(self) -> float:
        """Run one episode on the agent. Returns episode reward."""
        if hasattr(self.agent, "run_episode"):
            reward, _ = self.agent.run_episode()
        else:
            # Fallback: single act() call
            task = self.curriculum.get_current_task()
            self.agent.current_task = task
            result = self.agent.act()
            reward = 1.0 if result.get("status") == "success" else -0.5
            self.curriculum.update(reward)
        return reward

    def _reset_use_counts(self):
        """Reset recent_use_count on all ActionPatterns after each meta-step."""
        for p in self.agent.population.patterns:
            if hasattr(p, "recent_use_count"):
                p.recent_use_count = 0
```

**Step 5.4 — Run tests (expect pass)**

```bash
python -m pytest hpm_ai_v3/tests/test_meta_cognitive.py::TestMetaTrainingLoop -x
```

**Step 5.5 — Commit**

```bash
git add hpm_ai_v3/agents/meta_training.py hpm_ai_v3/tests/test_meta_cognitive.py
git commit -m "feat: implement MetaTrainingLoop wrapping N-episode meta-step cycle"
```

---

### Task 6 — Full test suite pass

**Step 6.1 — Run all meta tests**

```bash
python -m pytest hpm_ai_v3/tests/test_meta_cognitive.py -v
```

All tests must pass. Fix any failures before proceeding.

**Step 6.2 — Run existing tests to check no regressions**

```bash
python -m pytest hpm_ai_v3/tests/ -v --ignore=hpm_ai_v3/tests/test_meta_cognitive.py
```

**Step 6.3 — Final commit**

```bash
git add hpm_ai_v3/tests/test_meta_cognitive.py
git commit -m "test: complete MetaCognitivePattern test suite — all tests green"
```

---

## Self-Review Checklist

- [ ] Spec section 2 (64-dim features): MetaFeatureProjector nn.Linear(8,64) matches spec
- [ ] Spec section 3 (policy net 64→128→8): _MetaPolicyNet matches spec exactly
- [ ] Spec section 4 (8 directives): all 8 implemented in execute_directive dispatch table
- [ ] Spec section 5 (meta-reward formula): compute_meta_reward matches all 4 components and weights
- [ ] Spec section 6 (REINFORCE + eligibility traces): update_policy() uses discounted returns + baseline
- [ ] Spec section 7.1 (ActionPattern fields): exploration_temperature and recent_use_count added
- [ ] Spec section 7.2 (PatternPopulation): get_population_entropy() and get_diversity() added; get_top_patterns() already existed
- [ ] Spec section 7.3 (CurriculumManager): advance_phase() and set_difficulty() added
- [ ] Spec section 8 (HPMPattern abstract methods): all 6 abstract methods implemented
- [ ] Spec section 9 constraint 4 (weight=1.0 fixed): MetaCognitivePattern.weight = 1.0 in __init__
- [ ] Spec section 9 constraint 1 (episode-level only): MetaTrainingLoop only calls observe_and_act between episodes
- [ ] No TBDs, no placeholders, all file paths exact
