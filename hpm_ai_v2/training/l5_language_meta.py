"""Prequential L5 mixture over next-word order experts.

The mixer scores each event with its current strategy weights, then updates
those weights from expert log losses. It never observes a target before scoring
that event. State is serializable so later data rounds can continue calibration.
"""
from __future__ import annotations

import math
from typing import Mapping


class PrequentialL5Mixer:
    """Exponentially weighted online selector for normalized expert forecasts."""

    def __init__(self, experts: tuple[str, ...] | list[str], eta: float = 0.1,
                 fixed_share: float = 0.02):
        names = tuple(str(name) for name in experts)
        if not names or len(set(names)) != len(names):
            raise ValueError("experts must be a nonempty sequence of unique names")
        if not math.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive")
        if not math.isfinite(fixed_share) or not 0.0 <= fixed_share < 1.0:
            raise ValueError("fixed_share must be finite and in [0, 1)")
        self.experts = names
        self.eta = float(eta)
        self.fixed_share = float(fixed_share)
        self.log_weights = {name: 0.0 for name in names}
        self.updates = 0
        self.cumulative_log_loss = {name: 0.0 for name in names}

    def weights(self) -> dict[str, float]:
        peak = max(self.log_weights.values())
        masses = {name: math.exp(value - peak) for name, value in self.log_weights.items()}
        total = sum(masses.values())
        return {name: mass / total for name, mass in masses.items()}

    def probability(self, expert_probabilities: Mapping[str, float]) -> float:
        self._validate(expert_probabilities)
        return sum(self.weights()[name] * float(expert_probabilities[name])
                   for name in self.experts)

    def update_after_score(self, expert_probabilities: Mapping[str, float]) -> float:
        """Return pre-update mixture probability, then incorporate this event."""
        mixture_probability = self.probability(expert_probabilities)
        for name in self.experts:
            loss = -math.log(max(float(expert_probabilities[name]), 1e-300))
            self.cumulative_log_loss[name] += loss
            self.log_weights[name] -= self.eta * loss
        peak = max(self.log_weights.values())
        # A relative floor keeps every strategy available after long streams.
        self.log_weights = {name: max(value, peak - 40.0)
                            for name, value in self.log_weights.items()}
        if self.fixed_share:
            masses = self.weights()
            uniform = self.fixed_share / len(self.experts)
            self.log_weights = {
                name: math.log((1.0 - self.fixed_share) * masses[name] + uniform)
                for name in self.experts
            }
        self.updates += 1
        return mixture_probability

    def _validate(self, probabilities: Mapping[str, float]) -> None:
        if set(probabilities) != set(self.experts):
            raise ValueError("each event must provide a probability for every configured expert")
        if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
               for value in probabilities.values()):
            raise ValueError("expert probabilities must be finite and in [0, 1]")

    def state_dict(self) -> dict:
        return {"format": "hpm-l5-prequential-mixture-v1", "experts": list(self.experts),
                "eta": self.eta, "fixed_share": self.fixed_share,
                "log_weights": dict(self.log_weights),
                "updates": self.updates, "cumulative_log_loss": dict(self.cumulative_log_loss)}

    @classmethod
    def from_state_dict(cls, state: dict) -> "PrequentialL5Mixer":
        if state.get("format") != "hpm-l5-prequential-mixture-v1":
            raise ValueError("unsupported L5 language state format")
        mixer = cls(state["experts"], float(state["eta"]),
                    float(state.get("fixed_share", 0.0)))
        log_weights = {name: float(value) for name, value in state["log_weights"].items()}
        losses = {name: float(value) for name, value in state["cumulative_log_loss"].items()}
        if set(log_weights) != set(mixer.experts) or set(losses) != set(mixer.experts):
            raise ValueError("L5 state expert keys do not match configured experts")
        if any(not math.isfinite(v) for v in log_weights.values()) or any(
                not math.isfinite(v) or v < 0 for v in losses.values()):
            raise ValueError("L5 state contains invalid weights or losses")
        updates = int(state["updates"])
        if updates < 0:
            raise ValueError("L5 update count must be nonnegative")
        mixer.log_weights = log_weights
        mixer.cumulative_log_loss = losses
        mixer.updates = updates
        return mixer
