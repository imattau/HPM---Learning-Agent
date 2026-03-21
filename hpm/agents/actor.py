from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _field_score(field_quality: dict) -> float:
    """Scalar summarising field health: high when many deep patterns, low redundancy."""
    count = field_quality.get("level4plus_count", 0) or 0
    redundancy = field_quality.get("redundancy") or 0.0
    return float(count) * (1.0 - float(redundancy))


_INTERNAL_ACTION_NAMES = ["EXPLOIT", "EXPLORE", "REGROUND"]


# ---------------------------------------------------------------------------
# ExternalHead
# ---------------------------------------------------------------------------

class ExternalHead:
    """
    Selects a discrete external action every step.

    Uses the top pattern's log_prob scores as a prior over actions,
    combined with learned Q-values. Updated by external reward via TD(0).
    """

    def __init__(
        self,
        n_actions: int,
        alpha_ext: float,
        beta: float,
        temperature: float,
    ):
        self.q_values = np.zeros(n_actions)
        self.alpha_ext = alpha_ext
        self.beta = beta
        self.temperature = temperature
        self._last_action: int | None = None

    def select(
        self,
        action_vectors: np.ndarray,
        prediction,
        forecaster,
        top_pattern_id: str | None,
    ) -> int | None:
        """Return action index, or None if prediction is None."""
        if prediction is None:
            return None

        n_actions = len(action_vectors)

        # Look up top pattern from forecaster's store
        top_pattern = None
        if forecaster is not None and top_pattern_id is not None:
            for p, _w, _aid in forecaster._store.query_all():
                if p.id == top_pattern_id:
                    top_pattern = p
                    break

        if top_pattern is not None:
            # log_prob returns negative log-prob (negated logpdf); negate again
            # so higher logit = action more likely under the pattern
            log_prob_scores = np.array([
                -float(top_pattern.log_prob(action_vectors[i]))
                for i in range(n_actions)
            ])
            logits = self.beta * log_prob_scores + self.q_values
        else:
            # Fallback: Q-values only (no pattern available)
            logits = self.q_values.copy()

        probs = _softmax(logits / self.temperature)
        action = int(np.random.choice(n_actions, p=probs))
        self._last_action = action
        return action

    def update(self, external_reward: float) -> None:
        """TD(0) Q-update for the last selected action."""
        if self._last_action is not None:
            q = self.q_values[self._last_action]
            self.q_values[self._last_action] = q + self.alpha_ext * (external_reward - q)


# ---------------------------------------------------------------------------
# InternalHead
# ---------------------------------------------------------------------------

class InternalHead:
    """
    Selects a meta-action (EXPLOIT / EXPLORE / REGROUND) when triggered.

    Trigger: high redundancy OR fragility_flag.
    Reward: delta_field_score at the *next* trigger (delayed reward).
    """

    def __init__(
        self,
        alpha_int: float,
        temperature: float,
        redundancy_threshold: float,
        min_bridge_level_bounds: tuple,
    ):
        self.q_values = np.zeros(3)
        self.alpha_int = alpha_int
        self.temperature = temperature
        self.redundancy_threshold = redundancy_threshold
        self.min_bridge_level_bounds = min_bridge_level_bounds
        self._last_action: int | None = None
        self._baseline_score: float | None = None

    def _triggered(self, field_quality: dict, forecast_report: dict) -> bool:
        redundancy = field_quality.get("redundancy", 0.0) or 0.0
        fragility_flag = forecast_report.get("fragility_flag", False)
        return redundancy > self.redundancy_threshold or bool(fragility_flag)

    def step(
        self,
        field_quality: dict,
        forecast_report: dict,
        forecaster,
        bridge,
    ) -> str | None:
        """
        Check trigger, apply delayed reward, select and execute meta-action.
        Returns action name or None if not triggered.
        """
        if not self._triggered(field_quality, forecast_report):
            return None

        current_score = _field_score(field_quality)

        # Apply delayed reward from the previous trigger
        if self._last_action is not None and self._baseline_score is not None:
            reward = current_score - self._baseline_score
            q = self.q_values[self._last_action]
            self.q_values[self._last_action] = q + self.alpha_int * (reward - q)

        # Select new action
        probs = _softmax(self.q_values / self.temperature)
        action = int(np.random.choice(3, p=probs))
        self._last_action = action
        self._baseline_score = current_score

        # Execute
        self._execute(action, forecaster, bridge)
        return _INTERNAL_ACTION_NAMES[action]

    def _execute(self, action: int, forecaster, bridge) -> None:
        lo, hi = self.min_bridge_level_bounds
        if action == 0:  # EXPLOIT
            if forecaster is not None:
                forecaster.min_bridge_level = min(forecaster.min_bridge_level + 1, hi)
        elif action == 1:  # EXPLORE
            if forecaster is not None:
                forecaster.min_bridge_level = max(forecaster.min_bridge_level - 1, lo)
        elif action == 2:  # REGROUND
            if bridge is not None:
                bridge._t = bridge.T_substrate - 1


# ---------------------------------------------------------------------------
# DecisionalActor
# ---------------------------------------------------------------------------

class DecisionalActor:
    """
    HPM Phase 5 — Decisional RL Actor ("The Actor").

    Sits at the end of MultiAgentOrchestrator.step(). Makes two learned
    decisions each step:

    - ExternalHead: discrete action selection using top-pattern log-prob prior
      + learned Q-values; updated by environment reward each step.
    - InternalHead: meta-action dispatch (EXPLOIT/EXPLORE/REGROUND) triggered
      by field_quality; updated by delta_field_quality reward on next trigger.

    Composed into MultiAgentOrchestrator as optional actor=None kwarg.
    """

    def __init__(
        self,
        action_vectors: np.ndarray,       # shape (n_actions, feature_dim)
        forecaster=None,                   # ref to PredictiveSynthesisAgent
        bridge=None,                       # ref to SubstrateBridgeAgent
        alpha_ext: float = 0.1,
        alpha_int: float = 0.1,
        beta: float = 1.0,
        temperature: float = 1.0,
        redundancy_threshold: float = 0.3,
        min_bridge_level_bounds: tuple = (2, 6),
    ):
        self._action_vectors = action_vectors
        self._forecaster = forecaster
        self._bridge = bridge
        self._external_head = ExternalHead(
            n_actions=len(action_vectors),
            alpha_ext=alpha_ext,
            beta=beta,
            temperature=temperature,
        )
        self._internal_head = InternalHead(
            alpha_int=alpha_int,
            temperature=temperature,
            redundancy_threshold=redundancy_threshold,
            min_bridge_level_bounds=min_bridge_level_bounds,
        )

    def step(
        self,
        step_t: int,
        field_quality: dict,
        forecast_report: dict,
        external_reward: float = 0.0,
    ) -> dict:
        """
        Called by MultiAgentOrchestrator every step.

        Step flow:
        1. ExternalHead: select action using top-pattern prior + Q; update Q on reward
        2. InternalHead: check trigger; apply delayed reward; select + execute meta-action
        3. Return actor_report with all 4 keys

        Args:
            step_t:          Current orchestrator step counter.
            field_quality:   Dict from StructuralLawMonitor (or {}).
            forecast_report: Dict from PredictiveSynthesisAgent (or {}).
            external_reward: Reward from environment this step (default 0.0).

        Returns:
            actor_report dict -- all 4 keys always present.
        """
        prediction = forecast_report.get("prediction")
        top_pattern_id = forecast_report.get("top_pattern_id")

        # 1. ExternalHead
        external_action = self._external_head.select(
            self._action_vectors, prediction, self._forecaster, top_pattern_id
        )
        if external_action is not None:
            self._external_head.update(external_reward)

        # 2. InternalHead
        internal_action = self._internal_head.step(
            field_quality, forecast_report, self._forecaster, self._bridge
        )

        # 3. Return
        return {
            "external_action": external_action,
            "internal_action": internal_action,
            "external_q_values": self._external_head.q_values.tolist(),
            "internal_q_values": self._internal_head.q_values.tolist(),
        }
