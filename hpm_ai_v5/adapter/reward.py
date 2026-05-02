"""RewardAdapter — computes earned utility from prediction accuracy."""

from __future__ import annotations

from collections import deque
from typing import Any

from ..core.state import State
from .packet import AdapterPacket


class RewardAdapter:
    """Inject an earned utility signal into the packet based on prediction accuracy.

    The caller communicates the previous prediction by setting
    ``packet.context["predicted_next"]`` before running the registry.
    This adapter reads that key, compares it to the current observation,
    and injects a utility signal into the last State on the packet.
    """

    name = "reward"
    requires: list[str] = []
    provides: list[str] = ["state"]

    def __init__(
        self,
        correct_reward: float = 1.0,
        incorrect_penalty: float = 0.0,
        decay: float = 0.9,
        window_size: int = 8,
    ) -> None:
        self.correct_reward = correct_reward
        self.incorrect_penalty = incorrect_penalty
        self.decay = decay
        self.window_size = window_size

        self._previous_prediction: Any = None
        self._running_reward: float = 0.0
        self._correct: int = 0
        self._total: int = 0
        self._history: deque[bool] = deque(maxlen=window_size)

    def reset(self) -> None:
        """Clear all internal state."""
        self._previous_prediction = None
        self._running_reward = 0.0
        self._correct = 0
        self._total = 0
        self._history.clear()

    @property
    def accuracy(self) -> float:
        """Rolling accuracy over all steps."""
        return self._correct / self._total if self._total else 0.0

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        # Determine actual value from last state or raw
        actual: Any = packet.states[-1].value if packet.states else packet.raw

        # Check if we have a previous prediction to evaluate
        predicted = packet.context.get("predicted_next")
        correct: bool | None = None

        if predicted is not None:
            correct = predicted == actual
            self._total += 1
            if correct:
                self._correct += 1
            self._history.append(correct)
            step_reward = self.correct_reward if correct else self.incorrect_penalty
            self._running_reward = (
                self.decay * self._running_reward + (1 - self.decay) * step_reward
            )

        rolling_acc = sum(self._history) / len(self._history) if self._history else 0.0

        goal = {
            "utility": self._running_reward,
            "rolling_accuracy": rolling_acc,
            "correct": correct,
        }

        # Replace the last state with goal injected
        if packet.states:
            last = packet.states[-1]
            packet.states[-1] = State(
                value=last.value,
                step=last.step,
                goal=goal,
                context=last.context,
            )

        # Update packet context
        packet.context["reward_running"] = self._running_reward
        packet.context["reward_rolling_accuracy"] = rolling_acc
        packet.context["reward_correct"] = correct

        # Append a State representing the running reward
        packet.states.append(State(value=self._running_reward))

        packet.log(
            self.name,
            {
                "running_reward": self._running_reward,
                "rolling_accuracy": rolling_acc,
                "correct": correct,
            },
        )

        return packet
