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
        
        # Internal ToolSelector binding (used by LM for cache clearing)
        self._tool_selector = None

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
        # Curriculum pattern index is a better proxy for phase progress
        active_idx = getattr(curriculum, "active_pattern_idx", 0)
        phase_norm = float(active_idx) / n_phases

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
        phase_now = int(getattr(curriculum, "active_pattern_idx", 0))

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
