"""Learned meta-policy for selecting decoder adapters."""
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from hpm_ai_v4.agents.agent import HPMAgent


@dataclass(frozen=True)
class DecoderSpec:
    """Concrete decoder choice: family plus generation mode."""

    family: str
    mode: str = "decode"
    include_seed: bool = True

    def key(self) -> str:
        return f"{self.family}:{self.mode}:{int(self.include_seed)}"


class MetaDecoderPolicy:
    """HPM-style meta layer that learns which decoder spec works best."""

    def __init__(self, num_workers: int = 1):
        # 16 context buckets + 16 action slots + 16 reward buckets + slack.
        self.agent = HPMAgent(num_initial_patterns=4, obs_dim=64, num_workers=num_workers)
        self._spec_codes: Dict[str, int] = {}
        self._code_specs: Dict[int, DecoderSpec] = {}
        self._next_spec_code = 16
        self._selection_counts: Dict[str, int] = {}
        self._reward_ema: Dict[str, float] = {}
        self._metacognitive_reward_ema: Dict[str, float] = {}
        self._context_reward_ema: Dict[int, float] = {}
        self._context_reliability: Dict[int, float] = {}
        self._age = 0
        self.bootstrap_half_life = 48.0
        self.bootstrap_floor = 0.0
        self.bootstrap_warmup = 64
        self.exploration_weight = 0.10
        self.exploration_start_weight = 0.25
        self.exploration_decay_half_life = 2000.0

    def select(
        self,
        features: Dict[str, Any],
        candidates: Sequence[DecoderSpec],
        learn: bool = True,
    ) -> DecoderSpec:
        if not candidates:
            raise ValueError("MetaDecoderPolicy.select requires at least one candidate")

        context_code = self._context_code(features)
        dist = self._distribution([context_code])
        bootstrap = self.bootstrap_strength() if self._age < self.bootstrap_warmup else 0.0

        scored: List[Tuple[float, DecoderSpec]] = []
        for spec in candidates:
            code = self._spec_code(spec)
            score = float(dist[code])
            score += self._control_bonus(features, spec)
            score += bootstrap * self._heuristic_bonus(features, spec)
            score += self._exploration_bonus(spec)
            score += 0.02 * self._reward_ema.get(spec.key(), 0.0)
            score += self._metacognitive_bonus(context_code, spec)
            scored.append((score, spec))

        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = scored[0][1]
        if learn:
            chosen_key = chosen.key()
            self._selection_counts[chosen_key] = self._selection_counts.get(chosen_key, 0) + 1
            self.agent.obs_buffer.append(context_code)
            if len(self.agent.obs_buffer) > 100:
                self.agent.obs_buffer = self.agent.obs_buffer[-100:]
        return chosen

    def observe(self, features: Dict[str, Any], spec: DecoderSpec, outcome: Dict[str, float]) -> None:
        context_code = self._context_code(features)
        spec_code = self._spec_code(spec)
        reward_code = self._reward_code(outcome)
        self._age += 1
        reward = self._outcome_reward(outcome)
        key = spec.key()
        self._reward_ema[key] = 0.85 * self._reward_ema.get(key, 0.0) + 0.15 * reward
        self._metacognitive_reward_ema[key] = 0.90 * self._metacognitive_reward_ema.get(key, 0.0) + 0.10 * reward
        self._context_reward_ema[context_code] = 0.88 * self._context_reward_ema.get(context_code, 0.0) + 0.12 * reward
        self._context_reliability[context_code] = 0.90 * self._context_reliability.get(context_code, 0.0) + 0.10 * max(
            0.0,
            reward - 0.25,
        )
        for obs in (context_code, spec_code, reward_code):
            self.agent.perceive_and_learn(obs)

    def bootstrap_strength(self) -> float:
        """Return the current strength of the bootstrap prior."""
        strength = float(np.exp(-self._age / max(1.0, self.bootstrap_half_life)))
        return max(self.bootstrap_floor, strength)

    def _distribution(self, context_obs: List[int]) -> np.ndarray:
        relevant = self.agent.reasoner.get_relevant_patterns(context_obs, top_k=3)
        if not relevant:
            return np.ones(self.agent.obs_dim, dtype=np.float32) / float(self.agent.obs_dim)
        dist = self.agent.reasoner.compose_predictions(relevant, context_obs)
        dist = dist / (dist.sum() + 1e-12)
        return dist.astype(np.float32)

    def _context_code(self, features: Dict[str, Any]) -> int:
        code = 0
        if float(features.get("recent_agreement", 0.0)) >= 0.55:
            code |= 1
        if float(features.get("recent_plausibility", 0.0)) >= 0.45:
            code |= 2
        if float(features.get("structural_score", 0.0)) >= 0.12:
            code |= 4
        if int(features.get("stage_idx", 0)) >= 2:
            code |= 8
        return code

    def _reward_code(self, outcome: Dict[str, float]) -> int:
        agreement = float(outcome.get("token_agreement", 0.0))
        plausibility = float(outcome.get("plausibility", 0.0))
        structural = float(outcome.get("structural_score", 0.0))
        score = 0.55 * agreement + 0.30 * plausibility + 0.15 * structural
        if score < 0.25:
            return 32
        if score < 0.50:
            return 33
        if score < 0.75:
            return 34
        return 35

    def _spec_code(self, spec: DecoderSpec) -> int:
        key = spec.key()
        if key not in self._spec_codes:
            code = self._next_spec_code
            self._next_spec_code += 1
            if code >= 32:
                raise ValueError("MetaDecoderPolicy exhausted its action space")
            self._spec_codes[key] = code
            self._code_specs[code] = spec
        return self._spec_codes[key]

    def _heuristic_bonus(self, features: Dict[str, Any], spec: DecoderSpec) -> float:
        bonus = 0.0
        target_present = bool(features.get("target_present", False))
        requested_mode = str(features.get("requested_mode", "decode"))
        dialogue_act = str(features.get("dialogue_act", "default"))
        validators_present = bool(features.get("validators_present", False))
        recent_plausibility = float(features.get("recent_plausibility", 0.0))
        recent_agreement = float(features.get("recent_agreement", 0.0))
        stage_idx = int(features.get("stage_idx", 0))

        if spec.mode == requested_mode:
            bonus += 0.12
        if requested_mode == "target":
            if spec.family == "target":
                bonus += 0.40
            if spec.mode == "target":
                bonus += 0.15
            elif spec.family != "target":
                bonus -= 0.08
        if spec.family == "target" and target_present:
            bonus += 0.10
        if spec.family == "constrained" and validators_present:
            bonus += 0.10
        if spec.mode == "hybrid" and (recent_plausibility < 0.5 or recent_agreement < 0.5):
            bonus += 0.04
        if spec.mode == "decode" and recent_plausibility >= 0.5:
            bonus += 0.02
        if stage_idx >= 3 and spec.mode == "target":
            bonus += 0.02

        if dialogue_act in {"greeting", "closing"}:
            if spec.family == "word" and spec.mode == "decode":
                bonus += 0.10
            if spec.family == "constrained":
                bonus += 0.04
            if spec.family == "char":
                bonus -= 0.03
        elif dialogue_act in {"question", "request"}:
            if spec.family == "word":
                bonus += 0.08
            if spec.family == "target":
                bonus += 0.04
            if spec.family == "constrained":
                bonus += 0.03
            if spec.family == "char":
                bonus -= 0.02
        elif dialogue_act == "clarification":
            if spec.family in {"word", "target", "constrained"}:
                bonus += 0.05
            if spec.family == "char":
                bonus -= 0.02
        return bonus

    def _control_bonus(self, features: Dict[str, Any], spec: DecoderSpec) -> float:
        """Soft prior from the multi-polygraph control context."""
        family_prior = self._decode_prior(features.get("control_family_prior", {}))
        mode_prior = self._decode_prior(features.get("control_mode_prior", {}))
        control_strength = min(1.0, max(0.0, float(features.get("control_strength", 0.0))))
        dominant_family = features.get("control_dominant_family")
        dominant_mode = features.get("control_dominant_mode")

        bonus = 0.0
        if family_prior:
            bonus += (0.04 + 0.08 * control_strength) * family_prior.get(spec.family, 0.0)
        if mode_prior:
            bonus += (0.02 + 0.05 * control_strength) * mode_prior.get(spec.mode, 0.0)
        if dominant_family and spec.family == dominant_family:
            bonus += 0.03 + 0.04 * control_strength
        if dominant_mode and spec.mode == dominant_mode:
            bonus += 0.015 + 0.02 * control_strength
        return bonus

    def _metacognitive_bonus(self, context_code: int, spec: DecoderSpec) -> float:
        """L5 self-monitoring prior over which strategies have been reliable in this context."""
        context_signal = self._context_reward_ema.get(context_code, 0.0)
        context_reliability = self._context_reliability.get(context_code, 0.0)
        spec_signal = self._metacognitive_reward_ema.get(spec.key(), 0.0)
        bonus = 0.0
        bonus += 0.03 * context_signal
        bonus += 0.04 * context_reliability
        bonus += 0.02 * spec_signal
        if spec.family == "target" and context_reliability > 0.35:
            bonus += 0.02
        if spec.family == "word" and context_signal > 0.35:
            bonus += 0.015
        return bonus

    def _exploration_bonus(self, spec: DecoderSpec) -> float:
        """Small novelty bonus so unseen or underused specs are not crowded out."""
        count = self._selection_counts.get(spec.key(), 0)
        early_phase = float(np.exp(-self._age / max(1.0, self.exploration_decay_half_life)))
        effective_weight = self.exploration_weight + (self.exploration_start_weight - self.exploration_weight) * early_phase
        return effective_weight / float(np.sqrt(count + 1.0))

    def _outcome_reward(self, outcome: Dict[str, float]) -> float:
        agreement = float(outcome.get("token_agreement", 0.0))
        plausibility = float(outcome.get("plausibility", 0.0))
        structural = float(outcome.get("structural_score", 0.0))
        return 0.55 * agreement + 0.30 * plausibility + 0.15 * structural

    def _decode_prior(self, prior: Any) -> Dict[str, float]:
        if not isinstance(prior, dict) or not prior:
            return {}
        total = float(sum(max(0.0, float(v)) for v in prior.values()))
        if total <= 0.0:
            return {}
        return {str(k): max(0.0, float(v)) / total for k, v in prior.items() if float(v) > 0.0}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "age": self._age,
            "spec_codes": dict(self._spec_codes),
            "selection_counts": dict(self._selection_counts),
            "reward_ema": dict(self._reward_ema),
            "metacognitive_reward_ema": dict(self._metacognitive_reward_ema),
            "context_reward_ema": dict(self._context_reward_ema),
            "context_reliability": dict(self._context_reliability),
            "obs_buffer": list(self.agent.obs_buffer),
            "next_spec_code": self._next_spec_code,
            "bootstrap_half_life": self.bootstrap_half_life,
            "bootstrap_floor": self.bootstrap_floor,
            "bootstrap_warmup": self.bootstrap_warmup,
            "exploration_weight": self.exploration_weight,
            "exploration_start_weight": self.exploration_start_weight,
            "exploration_decay_half_life": self.exploration_decay_half_life,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._age = int(state.get("age", 0))
        self._spec_codes = dict(state.get("spec_codes", {}))
        self._code_specs = {
            int(code): self._decode_spec_key(key)
            for key, code in self._spec_codes.items()
        }
        self._selection_counts = dict(state.get("selection_counts", {}))
        self._reward_ema = dict(state.get("reward_ema", {}))
        self._metacognitive_reward_ema = dict(state.get("metacognitive_reward_ema", {}))
        self._context_reward_ema = dict(state.get("context_reward_ema", {}))
        self._context_reliability = dict(state.get("context_reliability", {}))
        self.agent.obs_buffer = list(state.get("obs_buffer", []))
        self._next_spec_code = int(state.get("next_spec_code", self._next_spec_code))
        self.bootstrap_half_life = float(state.get("bootstrap_half_life", self.bootstrap_half_life))
        self.bootstrap_floor = float(state.get("bootstrap_floor", self.bootstrap_floor))
        self.bootstrap_warmup = int(state.get("bootstrap_warmup", self.bootstrap_warmup))
        self.exploration_weight = float(state.get("exploration_weight", self.exploration_weight))
        self.exploration_start_weight = float(state.get("exploration_start_weight", self.exploration_start_weight))
        self.exploration_decay_half_life = float(state.get("exploration_decay_half_life", self.exploration_decay_half_life))

    @staticmethod
    def _decode_spec_key(key: str) -> DecoderSpec:
        family, mode, include_seed = key.split(":")
        return DecoderSpec(family, mode, bool(int(include_seed)))
