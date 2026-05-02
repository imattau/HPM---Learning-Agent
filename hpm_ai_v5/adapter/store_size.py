"""Adapter that recommends a dynamic PatternStore size based on context signals."""

from __future__ import annotations

from ..core.state import State
from .packet import AdapterPacket


class PatternStoreSizeAdapter:
    """Emit a recommended_max_patterns value into packet.context."""

    name = "store_size"
    requires: list[str] = []
    provides: list[str] = ["store_config"]

    def __init__(
        self,
        min_patterns: int = 8,
        max_patterns: int = 128,
        base_patterns: int = 32,
        entropy_scale: float = 2.0,
        stability_window: int = 8,
        decay_rate: float = 0.1,
    ) -> None:
        self.min_patterns = min_patterns
        self.max_patterns = max_patterns
        self.base_patterns = base_patterns
        self.entropy_scale = entropy_scale
        self.stability_window = stability_window
        self.decay_rate = decay_rate

        self.current_recommendation: int = base_patterns
        self._stable_steps: int = 0

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        entropy: float = float(packet.context.get("entropy", 0.0))
        surprise: float = float(packet.context.get("surprise", 0.0))
        dominant_period: int = int(packet.context.get("dominant_period", 0))

        period_floor = max(0, dominant_period * 2)
        entropy_target = int(self.base_patterns * (1.0 + entropy * self.entropy_scale))
        target = max(period_floor, entropy_target)
        target = max(self.min_patterns, min(self.max_patterns, target))

        if surprise < 0.3:
            self._stable_steps += 1
            if self._stable_steps >= self.stability_window:
                # decay current_recommendation toward target by decay_rate
                self.current_recommendation = int(
                    self.current_recommendation
                    + self.decay_rate * (target - self.current_recommendation)
                )
            else:
                # still grow to meet target floor before window is reached
                self.current_recommendation = max(self.current_recommendation, target)
        else:
            self._stable_steps = 0
            self.current_recommendation = max(self.current_recommendation, target)

        # clamp again after update
        self.current_recommendation = max(
            self.min_patterns, min(self.max_patterns, self.current_recommendation)
        )

        packet.context["recommended_max_patterns"] = self.current_recommendation
        packet.context["store_size_entropy_target"] = entropy_target
        packet.context["store_size_period_floor"] = period_floor
        packet.context["store_size_stable_steps"] = self._stable_steps

        packet.states.append(State(value=self.current_recommendation))
        packet.log(self.name, {"recommended_max_patterns": self.current_recommendation})

        return packet
