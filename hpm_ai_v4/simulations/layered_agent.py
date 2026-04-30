"""LayeredAgent: stacked L1 -> L2 -> L3 hierarchy over character streams."""
from collections import Counter, defaultdict
import json
import os
import re
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from hpm_ai_v4.agents.agent import HPMAgent
from hpm_ai_v4.agents.decoders import CharDecoder, ConstrainedDecoder, ExplanationDecoder, TargetDecoder, WordDecoder
from hpm_ai_v4.agents.meta_decoder_policy import DecoderSpec, MetaDecoderPolicy
from hpm_ai_v4.io.adapters import AsciiCharAdapter, CharClassAdapter, WordAdapter
from hpm_ai_v4.pattern import HierarchicalPattern, FlatPattern
from hpm_ai_v4.tools.dictionary import DictionaryValidator
from hpm_ai_v4.tools.grammar import GrammarValidator
from hpm_ai_v4.tools.text_signals import TextSignalExtractor


def _init_equal_weights(agent: HPMAgent, hier_k: int, obs_dim: int) -> None:
    """Replace agent.patterns with equal-weight hier+flat population."""
    agent.patterns = []
    for i in range(4):
        p = HierarchicalPattern(i, latent_dim=hier_k, obs_dim=obs_dim)
        p.weight = 0.15
        agent.patterns.append(p)
    for i in range(4, 6):
        p = FlatPattern.flat(i, obs_dim=obs_dim)
        p.weight = 0.10
        agent.patterns.append(p)


class LayeredAgent:
    """Three-level HPM stack where each level consumes the lower level's latent state."""

    SOFT_STATE_BINS = 5
    SOFT_STATE_OBS_DIM = 10
    DEFAULT_LAYER_LATENT_DIMS = {
        "l1": 2,
        "l2": 2,
        "l3": 8,
        "l4": 2,
    }

    def __init__(self, num_workers: int = 1,
                 dictionary: Optional[DictionaryValidator] = None,
                 grammar: Optional[GrammarValidator] = None,
                 surface_mode: str = "ascii",
                 layer_latent_dims: Optional[Dict[str, int]] = None):
        self._adapter = None
        self.surface_mode = ""
        self.layer_latent_dims = dict(self.DEFAULT_LAYER_LATENT_DIMS)
        if layer_latent_dims:
            for key, value in layer_latent_dims.items():
                if key in self.layer_latent_dims:
                    self.layer_latent_dims[key] = max(2, int(value))
        self._set_surface_mode(surface_mode)
        self.dictionary = dictionary
        self.grammar = grammar
        self.l1 = HPMAgent(obs_dim=self._adapter.obs_dim, num_initial_patterns=4, num_workers=num_workers,
                           dictionary=dictionary, grammar=grammar)
        self.l2 = HPMAgent(obs_dim=self.SOFT_STATE_OBS_DIM, num_initial_patterns=4, num_workers=num_workers,
                           dictionary=dictionary, grammar=grammar)
        self.l3 = HPMAgent(obs_dim=self.SOFT_STATE_OBS_DIM, num_initial_patterns=4, num_workers=num_workers,
                           dictionary=dictionary, grammar=grammar)
        self.l4 = HPMAgent(obs_dim=32, num_initial_patterns=4, num_workers=num_workers,
                           dictionary=dictionary, grammar=grammar)
        _init_equal_weights(self.l1, hier_k=self.layer_latent_dims["l1"], obs_dim=self._adapter.obs_dim)
        _init_equal_weights(self.l2, hier_k=self.layer_latent_dims["l2"], obs_dim=self.SOFT_STATE_OBS_DIM)
        _init_equal_weights(self.l3, hier_k=self.layer_latent_dims["l3"], obs_dim=self.SOFT_STATE_OBS_DIM)
        _init_equal_weights(self.l4, hier_k=self.layer_latent_dims["l4"], obs_dim=32)
        self._raw_history: List[int] = []
        self._l1_state_history: List[int] = []
        self._l2_state_history: List[int] = []
        self._class_char_counts = {i: Counter() for i in range(self._adapter.obs_dim)}
        self._char_counts = Counter()
        self._transition_counts = Counter()
        self._last_decoder_choice = "word"
        self._last_decoder_mode = "decode"
        self._last_decoder_stats: Dict[str, float] = {
            "token_agreement": 0.0,
            "plausibility": 0.0,
            "structural_score": 0.0,
        }
        self._pending_feedback: Dict[str, Any] = {}
        self.text_signals = TextSignalExtractor(use_spacy=False)
        self.decoder_policy = MetaDecoderPolicy(num_workers=num_workers)
        self.l5 = self.decoder_policy.agent
        self.decoders = {
            "char": CharDecoder(),
            "word": WordDecoder(),
            "target": TargetDecoder(),
            "constrained": ConstrainedDecoder(),
            "explain": ExplanationDecoder(),
        }

    def _set_surface_mode(self, surface_mode: str, surface_state: Optional[Dict[str, Any]] = None) -> None:
        mode = str(surface_mode or "coarse")
        if mode == "ascii":
            adapter = AsciiCharAdapter()
        elif mode == "word":
            adapter = WordAdapter(
                max_vocab_size=int((surface_state or {}).get("max_vocab_size", 5000)),
                lowercase=bool((surface_state or {}).get("lowercase", True)),
                vocab=dict((surface_state or {}).get("word_vocab", {}) or {}) or None,
            )
        else:
            adapter = CharClassAdapter()
            mode = "coarse"
        self.surface_mode = mode
        self._adapter = adapter
        if hasattr(self, "l1"):
            self.l1.obs_dim = adapter.obs_dim
            self.l1.reasoner.agent = self.l1
        self._class_char_counts = defaultdict(Counter)

    def perceive(self, raw_char_id: int, feedback: Optional[Dict[str, Any]] = None) -> None:
        """Feed one character through the hierarchy."""
        if feedback is None:
            feedback = {}
        if hasattr(self, "_pending_feedback") and self._pending_feedback:
            feedback.update(self._pending_feedback)
            self._pending_feedback = {}

        class_id = self._adapter.encode(raw_char_id)
        self._raw_history.append(int(class_id))
        surface_label = self._surface_label_for_obs(class_id)
        self._class_char_counts[class_id][surface_label] += 1
        self._char_counts[surface_label] += 1
        if len(self._raw_history) > 1:
            prev_label = self._surface_label_for_obs(self._raw_history[-2])
            self._transition_counts[(prev_label, surface_label)] += 1
        stacked_feedback = self._stack_feedback_signal(feedback)
        self.l1.perceive_and_learn(class_id, feedback=stacked_feedback)

        l1_state = self.l1_top_state()
        self._l1_state_history.append(l1_state)
        self.l2.perceive_and_learn(l1_state, feedback=stacked_feedback)

        l2_state = self.l2_soft_state()
        self._l2_state_history.append(l2_state)
        self.l3.perceive_and_learn(l2_state, feedback=stacked_feedback)

    def _top_state(self, agent: HPMAgent) -> int:
        if not agent.patterns:
            return 0
        best = max(agent.patterns, key=lambda p: p.weight)
        return int(best.get_top_state(agent.obs_buffer[-20:]))

    def l1_top_state(self) -> int:
        return self._top_state(self.l1)

    def l2_top_state(self) -> int:
        return self._top_state(self.l2)

    def _best_pattern_posterior(self, agent: HPMAgent, obs_seq: Optional[List[int]] = None) -> np.ndarray:
        if not agent.patterns:
            return np.array([0.5, 0.5], dtype=np.float32)

        context = list(obs_seq or agent.obs_buffer[-20:])
        best = max(agent.patterns, key=lambda p: p.weight)
        if best.latent_dim <= 1:
            return np.ones(1, dtype=np.float32)
        alpha, _ = best._forward(context)
        if alpha.size == 0:
            return np.ones(best.latent_dim, dtype=np.float32) / float(best.latent_dim)
        posterior = alpha[-1].astype(np.float32)
        posterior = np.nan_to_num(
            posterior,
            nan=1.0 / max(1, best.latent_dim),
            posinf=1.0 / max(1, best.latent_dim),
            neginf=1.0 / max(1, best.latent_dim),
        )
        posterior /= posterior.sum() + 1e-12
        return posterior

    def _soft_state_code(self, posterior: np.ndarray) -> int:
        if posterior.size == 0:
            return 0
        if posterior.size == 1:
            return self.SOFT_STATE_BINS - 1
        top = int(np.argmax(posterior))
        confidence = float(posterior[top])
        confidence_bucket = min(
            self.SOFT_STATE_BINS - 1,
            max(0, int(round(confidence * (self.SOFT_STATE_BINS - 1)))),
        )
        return int(top * self.SOFT_STATE_BINS + confidence_bucket)

    def l1_state_distribution(self) -> np.ndarray:
        """Posterior over the best L1 latent state, preserved as a distribution."""
        return self._best_pattern_posterior(self.l1)

    def l1_soft_state(self) -> int:
        """Encode L1 posterior state + confidence into a discrete symbol for L2."""
        return self._soft_state_code(self.l1_state_distribution())

    def l2_state_distribution(self) -> np.ndarray:
        """Soft summary of L2 over the current population."""
        if not self.l2.patterns:
            return np.ones(self.l2.obs_dim, dtype=np.float32) / float(self.l2.obs_dim)

        obs_seq = list(self.l2.obs_buffer[-20:]) if self.l2.obs_buffer else []
        dist = np.zeros(self.l2.obs_dim, dtype=np.float32)
        total_w = 0.0
        for p in self.l2.patterns:
            w = float(max(0.0, p.weight))
            if w <= 0.0:
                continue
            dist += w * p.predict_next_distribution(obs_seq)
            total_w += w
        if total_w <= 0.0:
            return np.ones(self.l2.obs_dim, dtype=np.float32) / float(self.l2.obs_dim)
        dist /= total_w
        dist /= dist.sum() + 1e-12
        return dist.astype(np.float32)

    def l2_soft_state(self) -> int:
        """Encode L2 posterior state + confidence into a discrete symbol for L3."""
        posterior = self._best_pattern_posterior(self.l2)
        return self._soft_state_code(posterior)

    def l3_soft_state(self) -> int:
        """Encode L3 posterior state + confidence into a discrete symbol for L4/Binding."""
        posterior = self._best_pattern_posterior(self.l3)
        return self._soft_state_code(posterior)

    def generate(self, steps: int = 20) -> str:
        """Generate a readable sequence of level-3 state labels."""
        return self.decoders["explain"].decode(self, steps=steps)

    def _decoder_policy_features(
        self,
        target_text: str | None = None,
        requested_mode: str = "decode",
        context_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        l1 = self.l1_metrics()
        l2 = self.l2_metrics(list(self._l1_state_history[-200:]))
        l3 = self.l3_metrics(list(self._l2_state_history[-200:]))
        l4 = self.l4_metrics()
        l5 = self.l5_metrics()
        control = self.l1.reasoner.control_context(self._raw_history)
        structural_score = float((l1["mi"] + l2["mi"] + l3["mi"]) / 3.0)
        selection_counts = self.decoder_policy.state_dict().get("selection_counts", {})
        selection_total = sum(int(v) for v in selection_counts.values())
        selection_entropy = 0.0
        if selection_total > 0:
            probs = np.array([float(v) / float(selection_total) for v in selection_counts.values()], dtype=np.float32)
            selection_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        reasoner_memory = int(self.l1.reasoner.memory_size + self.l2.reasoner.memory_size + self.l3.reasoner.memory_size)
        features = {
            "recent_agreement": self._last_decoder_stats.get("token_agreement", 0.0),
            "recent_plausibility": self._last_decoder_stats.get("plausibility", 0.0),
            "structural_score": structural_score,
            "meta_structural_score": float((l4["mi"] + l5["mi"]) / 2.0),
            "policy_age": self.decoder_policy._age,
            "policy_entropy": selection_entropy,
            "reasoner_memory": reasoner_memory,
            "control_family_prior": control.get("family_prior", {}),
            "control_mode_prior": control.get("mode_prior", {}),
            "control_stage_prior": control.get("stage_prior", {}),
            "control_strength": float(control.get("community_strength", 0.0)),
            "control_dominant_family": control.get("dominant_family"),
            "control_dominant_mode": control.get("dominant_mode"),
            "control_dominant_stage": control.get("dominant_stage"),
            "control_top_community": control.get("top_community"),
            "control_summary_count": int(control.get("summary_count", 0)),
            "stage_idx": self._stack_stage_index(),
            "validators_present": bool(self.dictionary or self.grammar),
            "target_present": bool(target_text),
            "requested_mode": requested_mode,
        }
        if context_features:
            features.update({str(k): v for k, v in context_features.items()})
        return features

    def _stack_feedback_signal(self, feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Blend external feedback with current meta-state for lower-level learning."""
        signal = dict(feedback or {})
        control = self.l1.reasoner.control_context(self._raw_history, feature_pack=signal)
        meta_structural_score = float((self.l4_metrics()["mi"] + self.l5_metrics()["mi"]) / 2.0)
        control_strength = float(control.get("community_strength", 0.0))
        topdown_gate = float(np.clip(0.5 * control_strength + 0.5 * meta_structural_score, 0.0, 1.0))
        signal.setdefault("control_mode_prior", control.get("mode_prior", {}))
        signal.setdefault("control_family_prior", control.get("family_prior", {}))
        signal.setdefault("control_stage_prior", control.get("stage_prior", {}))
        signal.setdefault("control_strength", control_strength)
        signal.setdefault("control_dominant_mode", control.get("dominant_mode"))
        signal.setdefault("control_dominant_family", control.get("dominant_family"))
        signal.setdefault("control_dominant_stage", control.get("dominant_stage"))
        signal.setdefault("control_summary_count", int(control.get("summary_count", 0)))
        signal.setdefault("meta_structural_score", meta_structural_score)
        signal.setdefault("topdown_gate", topdown_gate)
        signal.setdefault("topdown_confidence", topdown_gate)
        if signal.get("mode") is None and signal.get("desired_mode") is None:
            signal.setdefault("mode", control.get("dominant_mode"))
        return signal

    def _stack_stage_index(self) -> int:
        stage = self.l1.development.level if hasattr(self.l1, "development") else "surface"
        return {
            "surface": 0,
            "local": 1,
            "relational": 2,
            "abstract": 3,
            "generative": 4,
        }.get(stage, 0)

    def _learn_decoder_policy(
        self,
        decoder_spec: DecoderSpec,
        generated_text: str,
        target_text: str | None = None,
    ) -> Dict[str, float]:
        if target_text:
            stats = self.evaluate_generated_text(generated_text, target_text)
        else:
            stats = {
                "token_agreement": 0.0,
                "plausibility": self._text_plausibility(generated_text),
            }
        stats["structural_score"] = float((self.l1_metrics()["mi"] + self.l2_metrics(list(self._l1_state_history[-200:]))["mi"] + self.l3_metrics(list(self._l2_state_history[-200:]))["mi"]) / 3.0)
        self._last_decoder_choice = decoder_spec.family
        self._last_decoder_mode = decoder_spec.mode
        self._last_decoder_stats = dict(stats)
        self._update_meta_levels(decoder_spec, stats, target_text)
        self.decoder_policy.observe(self._decoder_policy_features(target_text), decoder_spec, stats)
        return stats

    def _update_meta_levels(
        self,
        decoder_spec: DecoderSpec,
        stats: Dict[str, float],
        target_text: str | None,
    ) -> None:
        family_code = {
            "word": 0,
            "char": 1,
            "target": 2,
            "constrained": 3,
        }.get(decoder_spec.family, 4)
        mode_code = {
            "decode": 0,
            "hybrid": 1,
            "target": 2,
        }.get(decoder_spec.mode, 3)
        agreement = float(stats.get("token_agreement", 0.0))
        plausibility = float(stats.get("plausibility", 0.0))
        structural = float(stats.get("structural_score", 0.0))
        stage_idx = self._stack_stage_index()
        requested_target = 1 if target_text else 0
        validators = 1 if (self.dictionary or self.grammar) else 0

        quality_code = 16 + min(3, int(agreement * 4))
        plausibility_code = 20 + min(3, int(plausibility * 4))
        structural_code = 24 + min(3, int(structural * 4))
        stage_code = 28 + min(3, stage_idx)
        target_code = 31 if requested_target else 30
        validator_code = 29 if validators else 28

        for obs in [
            family_code + 4 * mode_code,
            quality_code,
            plausibility_code,
            structural_code,
            stage_code,
            target_code,
            validator_code,
        ]:
            self.l4.perceive_and_learn(int(obs % self.l4.obs_dim))

    def _choose_decoder_spec(
        self,
        candidates: List[DecoderSpec],
        target_text: str | None = None,
        requested_mode: str = "decode",
        context_features: Optional[Dict[str, Any]] = None,
        learn_policy: bool = True,
    ) -> DecoderSpec:
        if not candidates:
            raise ValueError("No decoder candidates available")
        return self.decoder_policy.select(
            self._decoder_policy_features(
                target_text,
                requested_mode=requested_mode,
                context_features=context_features,
            ),
            candidates,
            learn=learn_policy,
        )

    def generate_text(
        self,
        steps: int = 80,
        seed_text: str | None = None,
        target_text: str | None = None,
        mode: str = "decode",
        include_seed: bool = True,
        feedback: bool = False,
        update_policy: bool = True,
        context_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate readable text using a learned decoder policy."""
        if self.surface_mode == "word":
            candidates = [
                DecoderSpec("word", "decode", include_seed),
                DecoderSpec("word", "hybrid", include_seed),
            ]
            if target_text:
                candidates.append(DecoderSpec("word", "target", include_seed))
        elif target_text and mode == "target":
            candidates = [DecoderSpec("target", "target", False)]
            if self.dictionary or self.grammar:
                candidates.append(DecoderSpec("constrained", "target", False))
        else:
            task_family = str((context_features or {}).get("task_family", ""))
            allow_char = task_family not in {"repair"}
            candidates = [
                DecoderSpec("word", "decode", include_seed),
                DecoderSpec("word", "hybrid", include_seed),
            ]
            if allow_char:
                candidates.extend([
                    DecoderSpec("char", "decode", include_seed),
                    DecoderSpec("char", "hybrid", include_seed),
                ])
            if target_text:
                candidates.append(DecoderSpec("target", "target", False))
                candidates.append(DecoderSpec("word", "target", include_seed))
            if task_family == "repair":
                candidates.append(DecoderSpec("word", "target", include_seed))
        chosen = self._choose_decoder_spec(
            candidates,
            target_text=target_text,
            requested_mode=mode,
            context_features=context_features,
            learn_policy=update_policy,
        )
        decoder = self.decoders[chosen.family]
        if self.surface_mode == "word":
            text = decoder.decode(
                self,
                steps=steps,
                seed_text=seed_text,
                target_text=target_text,
                mode=chosen.mode,
                include_seed=chosen.include_seed,
                feedback=feedback,
            )
        elif chosen.family == "target":
            text = decoder.decode(
                self,
                target_text=target_text or "",
                seed_text=seed_text,
                horizon=max(steps, len(target_text) if target_text else steps),
                strategy="beam",
                lookback=self.l1.reasoner.context_window,
            )
        else:
            text = decoder.decode(
                self,
                steps=steps,
                seed_text=seed_text,
                target_text=target_text,
                mode=chosen.mode,
                include_seed=chosen.include_seed,
                feedback=feedback,
            )
        if update_policy:
            self._learn_decoder_policy(chosen, text, target_text=target_text)
        return text

    def generate_chars(
        self,
        steps: int = 80,
        seed_text: str | None = None,
        target_text: str | None = None,
        mode: str = "decode",
        include_seed: bool = True,
        feedback: bool = False,
        update_policy: bool = True,
        context_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate printable character continuation through a learned policy."""
        if self.surface_mode == "word":
            return self.generate_text(
                steps=steps,
                seed_text=seed_text,
                target_text=target_text,
                mode=mode,
                include_seed=include_seed,
                feedback=feedback,
                update_policy=update_policy,
                context_features=context_features,
            )
        candidates = [
            DecoderSpec("char", "decode", include_seed),
            DecoderSpec("char", "hybrid", include_seed),
        ]
        if target_text:
            candidates.append(DecoderSpec("char", "target", include_seed))
        chosen = self._choose_decoder_spec(
            candidates,
            target_text=target_text,
            requested_mode=mode,
            context_features=context_features,
            learn_policy=update_policy,
        )
        text = self.decoders["char"].decode(
            self,
            steps=steps,
            seed_text=seed_text,
            target_text=target_text,
            mode=chosen.mode,
            include_seed=chosen.include_seed,
            feedback=feedback,
        )
        if update_policy:
            self._learn_decoder_policy(chosen, text, target_text=target_text)
        return text

    def l4_metrics(self) -> Dict[str, Any]:
        top3 = sorted(self.l4.patterns, key=lambda p: -p.weight)[:3]
        total_w = sum(p.weight for p in top3) + 1e-12
        mi = float(sum(p.weight * p.compression() for p in top3) / total_w) if top3 else 0.0
        return {
            'pop_size': len(self.l4.patterns),
            'mi': mi,
            'stage': self.l4.development.level,
            'best_weight': max(p.weight for p in self.l4.patterns) if self.l4.patterns else 0.0,
        }

    def l5_metrics(self) -> Dict[str, Any]:
        top3 = sorted(self.l5.patterns, key=lambda p: -p.weight)[:3]
        total_w = sum(p.weight for p in top3) + 1e-12
        mi = float(sum(p.weight * p.compression() for p in top3) / total_w) if top3 else 0.0
        return {
            'pop_size': len(self.l5.patterns),
            'mi': mi,
            'stage': self.l5.development.level,
            'best_weight': max(p.weight for p in self.l5.patterns) if self.l5.patterns else 0.0,
        }

    @classmethod
    def from_library(cls, path: str, frozen: bool = True,
                     num_workers: int = 1,
                     dictionary=None, grammar=None) -> "LayeredAgent":
        """Create a LayeredAgent with l1 patterns pre-loaded from a saved library.

        Parameters
        ----------
        path:
            Path to a .pkl file produced by build_large_nlp_library.
        frozen:
            If True, the loaded l1 patterns' weights are fixed (no weight
            updates during further training) so the library acts as a stable
            prior.  Higher levels train freely on top.
        """
        from hpm_ai_v4.tools.serializer import PatternSerializer
        agent = cls(num_workers=num_workers, dictionary=dictionary, grammar=grammar)
        surface_path = path + ".surface.json"
        if os.path.exists(surface_path):
            with open(surface_path, "r", encoding="utf-8") as f:
                surface_state = json.load(f)
            agent._set_surface_mode(surface_state.get("surface_mode", agent.surface_mode), surface_state=surface_state)
        patterns = PatternSerializer.load(path)
        if patterns:
            obs_dim = getattr(patterns[0], "obs_dim", agent._adapter.obs_dim)
            if not os.path.exists(surface_path):
                inferred_surface = "word" if obs_dim > 95 else "ascii" if obs_dim == 95 else "coarse"
                agent._set_surface_mode(inferred_surface)
        agent.l1.patterns = patterns
        if frozen:
            for p in agent.l1.patterns:
                p._frozen = True
        return agent

    def save_bundle(self, base_path: str) -> None:
        from hpm_ai_v4.tools.serializer import PatternSerializer
        PatternSerializer.save(self.l1.patterns, base_path + ".l1.pkl")
        PatternSerializer.save(self.l2.patterns, base_path + ".l2.pkl")
        PatternSerializer.save(self.l3.patterns, base_path + ".l3.pkl")
        PatternSerializer.save(self.l4.patterns, base_path + ".l4.pkl")
        PatternSerializer.save(self.l5.patterns, base_path + ".l5.pkl")
        self._save_reasoner_state(base_path, "l1", self.l1.reasoner)
        self._save_reasoner_state(base_path, "l2", self.l2.reasoner)
        self._save_reasoner_state(base_path, "l3", self.l3.reasoner)
        self._save_reasoner_state(base_path, "l4", self.l4.reasoner)
        self._save_reasoner_state(base_path, "l5", self.l5.reasoner)
        with open(base_path + ".surface.json", "w", encoding="utf-8") as f:
            surface_state = {
                "surface_mode": self.surface_mode,
                "layer_latent_dims": dict(self.layer_latent_dims),
            }
            if self.surface_mode == "word" and hasattr(self._adapter, "_word_to_id"):
                surface_state["max_vocab_size"] = int(getattr(self._adapter, "_max_vocab_size", 5000))
                surface_state["lowercase"] = bool(getattr(self._adapter, "lowercase", True))
                surface_state["word_vocab"] = dict(getattr(self._adapter, "_word_to_id", {}))
            json.dump(surface_state, f)
        with open(base_path + ".policy.json", "w", encoding="utf-8") as f:
            json.dump(self.decoder_policy.state_dict(), f)

    def load_bundle(self, base_path: str) -> int:
        from hpm_ai_v4.tools.serializer import PatternSerializer
        loaded = 0
        if os.path.exists(base_path + ".l1.pkl"):
            self.l1.patterns = PatternSerializer.load(base_path + ".l1.pkl")
            loaded += 1
        if os.path.exists(base_path + ".l2.pkl"):
            self.l2.patterns = PatternSerializer.load(base_path + ".l2.pkl")
            loaded += 1
        if os.path.exists(base_path + ".l3.pkl"):
            self.l3.patterns = PatternSerializer.load(base_path + ".l3.pkl")
            loaded += 1
        if os.path.exists(base_path + ".l4.pkl"):
            self.l4.patterns = PatternSerializer.load(base_path + ".l4.pkl")
            loaded += 1
        if os.path.exists(base_path + ".l5.pkl"):
            self.l5.patterns = PatternSerializer.load(base_path + ".l5.pkl")
            loaded += 1
        loaded += self._load_reasoner_state(base_path, "l1", self.l1.reasoner)
        loaded += self._load_reasoner_state(base_path, "l2", self.l2.reasoner)
        loaded += self._load_reasoner_state(base_path, "l3", self.l3.reasoner)
        loaded += self._load_reasoner_state(base_path, "l4", self.l4.reasoner)
        loaded += self._load_reasoner_state(base_path, "l5", self.l5.reasoner)
        policy_path = base_path + ".policy.json"
        if os.path.exists(policy_path):
            with open(policy_path, "r", encoding="utf-8") as f:
                self.decoder_policy.load_state_dict(json.load(f))
        surface_path = base_path + ".surface.json"
        if os.path.exists(surface_path):
            with open(surface_path, "r", encoding="utf-8") as f:
                surface_state = json.load(f)
            layer_latent_dims = surface_state.get("layer_latent_dims")
            if isinstance(layer_latent_dims, dict):
                for key, value in layer_latent_dims.items():
                    if key in self.layer_latent_dims:
                        self.layer_latent_dims[key] = max(2, int(value))
            self._set_surface_mode(surface_state.get("surface_mode", self.surface_mode), surface_state=surface_state)
        elif self.l1.patterns:
            obs_dim = getattr(self.l1.patterns[0], "obs_dim", self._adapter.obs_dim)
            inferred_surface = "word" if obs_dim > 95 else "ascii" if obs_dim == 95 else "coarse"
            self._set_surface_mode(inferred_surface)
        return loaded

    def _save_reasoner_state(self, base_path: str, level: str, reasoner) -> None:
        path = f"{base_path}.reasoner.{level}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(reasoner.state_dict(), f)

    def _load_reasoner_state(self, base_path: str, level: str, reasoner) -> int:
        path = f"{base_path}.reasoner.{level}.json"
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            reasoner.load_state_dict(json.load(f))
        return 1

    def plan_text_continuation(
        self,
        target_text: str,
        seed_text: str | None = None,
        horizon: Optional[int] = None,
        strategy: str = "beam",
        lookback: Optional[int] = None,
        feature_pack: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Plan a printable continuation directly through the reasoner."""
        if self.surface_mode == "word":
            return self.decoders["word"].decode(
                self,
                steps=horizon if horizon is not None else len(self._tokenize_words(target_text)),
                seed_text=seed_text,
                target_text=target_text,
                mode="target",
                include_seed=False,
            )
        return self.decoders["target"].decode(
            self,
            target_text=target_text,
            seed_text=seed_text,
            horizon=horizon,
            strategy=strategy,
            lookback=lookback,
            feature_pack=feature_pack,
        )

    def generate_constrained_text(
        self,
        steps: int = 80,
        seed_text: str | None = None,
        target_text: str | None = None,
        mode: str = "decode",
        include_seed: bool = True,
        feedback: bool = False,
        allowed_words: Optional[set[str]] = None,
        strict_dictionary: bool = True,
        strict_grammar: bool = True,
        update_policy: bool = True,
        context_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text with additional lexical constraints."""
        if self.surface_mode == "word":
            return self.generate_text(
                steps=steps,
                seed_text=seed_text,
                target_text=target_text,
                mode=mode,
                include_seed=include_seed,
                feedback=feedback,
                update_policy=update_policy,
                context_features=context_features,
            )
        candidates = [
            DecoderSpec("constrained", "decode", include_seed),
            DecoderSpec("constrained", "hybrid", include_seed),
        ]
        if target_text:
            candidates.append(DecoderSpec("constrained", "target", include_seed))
        chosen = self._choose_decoder_spec(
            candidates,
            target_text=target_text,
            requested_mode=mode,
            context_features=context_features,
            learn_policy=update_policy,
        )
        text = self.decoders["constrained"].decode(
            self,
            steps=steps,
            seed_text=seed_text,
            target_text=target_text,
            mode=chosen.mode,
            include_seed=chosen.include_seed,
            feedback=feedback,
            allowed_words=allowed_words,
            strict_dictionary=strict_dictionary,
            strict_grammar=strict_grammar,
        )
        if update_policy:
            self._learn_decoder_policy(chosen, text, target_text=target_text)
        return text

    def repair_text(
        self,
        corrupted_text: str,
        target_text: str,
        steps: Optional[int] = None,
        use_constraints: Optional[bool] = None,
        mode: str = "target",
        update_policy: bool = True,
    ) -> str:
        """Repair a noisy fragment by target-conditioned continuation.

        This is a thin convenience wrapper over the existing target-aware
        generation paths. It keeps the core learning loop unchanged while
        making the text-repair use case explicit.
        """
        if steps is None:
            steps = max(8, min(20, len(target_text) // 2 if target_text else 20))
        if use_constraints is None:
            use_constraints = bool(self.dictionary or self.grammar)

        if use_constraints:
            if self.surface_mode == "word":
                return self.generate_text(
                    steps=steps,
                    seed_text=corrupted_text,
                    target_text=target_text,
                    mode=mode,
                    include_seed=False,
                    update_policy=update_policy,
                    context_features={"task_family": "repair", "repair_mode": True},
                )
            return self.generate_constrained_text(
                steps=steps,
                seed_text=corrupted_text,
                target_text=target_text,
                mode=mode,
                include_seed=False,
                allowed_words=set(self.dictionary.words) if self.dictionary else None,
                strict_dictionary=True,
                strict_grammar=True,
                update_policy=update_policy,
            )

        return self.generate_text(
            steps=steps,
            seed_text=corrupted_text,
            target_text=target_text,
            mode=mode,
            include_seed=False,
            update_policy=update_policy,
            context_features={"task_family": "repair", "repair_mode": True},
        )

    def observe_text(
        self,
        text: str,
        feedback_mode: str = "target",
        generated_text: str | None = None,
        self_feedback_weight: float = 0.05,
        feedback_signal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train on external text and optionally apply a small self-feedback pass.

        feedback_mode:
            target  - only the supplied text is learned
            self    - only generated_text is fed back (if provided)
            hybrid  - learn target text and optionally add a gated self-feedback pass
            none    - do not learn, just return stats (useful for discourse-only updates)
        """
        if feedback_mode not in {"target", "self", "hybrid", "none"}:
            raise ValueError(f"Unsupported feedback_mode: {feedback_mode!r}")

        stats: Dict[str, Any] = {
            "target_chars": 0,
            "self_chars": 0,
            "token_agreement": 0.0,
            "plausibility": 0.0,
        }

        if text and feedback_mode != "none":
            for raw_id in self._surface_ids_from_text(text):
                self.perceive(raw_id, feedback=feedback_signal)
                stats["target_chars"] += 1

        if feedback_mode in {"self", "hybrid"} and generated_text:
            stats["token_agreement"] = self._text_agreement(generated_text, text)
            stats["plausibility"] = self._text_plausibility(generated_text)

            should_feedback = feedback_mode == "self"
            if feedback_mode == "hybrid":
                should_feedback = (
                    stats["token_agreement"] >= 0.25 and
                    stats["plausibility"] >= 0.35
                )

            if should_feedback and self_feedback_weight > 0:
                sampled = self._sample_feedback_text(generated_text, self_feedback_weight)
                stats["self_chars"] = len(sampled)
                self._feed_text_back(sampled, feedback=feedback_signal)

        return stats

    def observe_code_dsl(
        self,
        target_program: str,
        generated_program: str | None = None,
        adapter: Any = None,
        feedback_mode: str = "target",
        self_feedback_weight: float = 0.05,
        feedback_signal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Observe a code/DSL program and feed execution feedback back into the stack."""
        from hpm_ai_v4.io.adapters import CodeDSLAdapter

        adapter = adapter or CodeDSLAdapter()
        target_value = adapter.execute(target_program)
        generated_value = adapter.execute(generated_program) if generated_program else None
        code_signal = {
            "kind": "code_dsl",
            "parseable": bool(adapter.from_text(target_program)),
            "canonical_match": bool(generated_program) and adapter.to_text(generated_program) == adapter.to_text(target_program),
            "target_value": target_value,
        }
        if generated_program is not None:
            code_signal["generated_parseable"] = bool(adapter.from_text(generated_program))
            code_signal["execution_match"] = generated_value == target_value and generated_value is not None
            code_signal["semantic_mismatch"] = generated_value is not None and generated_value != target_value
            code_text_signal = self.text_signals.analyze(
                generated_program,
                context_texts=[target_program],
                target_text=target_program,
                dictionary=self.dictionary,
                grammar=self.grammar,
            )
            code_signal.update(code_text_signal.to_dict())
            code_signal["text_signal_score"] = code_text_signal.combined_score()
        if feedback_signal:
            code_signal.update(feedback_signal)

        stats = self.observe_text(
            target_program,
            feedback_mode=feedback_mode,
            generated_text=generated_program,
            self_feedback_weight=self_feedback_weight,
            feedback_signal=code_signal,
        )
        if generated_program is not None:
            stats.update(code_signal)
            self._learn_decoder_policy(
                DecoderSpec(self._last_decoder_choice, self._last_decoder_mode, True),
                generated_program,
                target_text=target_program,
            )
        return stats

    def evaluate_generated_text(self, generated_text: str, target_text: str) -> Dict[str, float]:
        """Compare generated text against a target for reporting and gating."""
        generated_tokens = self._tokenize_words(generated_text.lower())
        target_tokens = self._tokenize_words(target_text.lower())
        if not target_tokens:
            return {"token_agreement": 0.0, "plausibility": self._text_plausibility(generated_text)}

        matches = sum(1 for a, b in zip(generated_tokens, target_tokens) if a == b)
        agreement = matches / len(target_tokens)
        return {
            "token_agreement": agreement,
            "plausibility": self._text_plausibility(generated_text),
        }

    def predict_next_chars(self, context_raw: List[int], top_k: int = 5) -> List[Tuple[str, float]]:
        """Top-k next surface-bucket predictions from L1."""
        return self.predict_next_surface(context_raw, top_k=top_k)

    def predict_next_surface(self, context_raw: List[int], top_k: int = 5) -> List[Tuple[str, float]]:
        """Top-k next surface-token predictions from L1."""
        context_cls = [self._adapter.encode(v) for v in context_raw]
        relevant = self.l1.reasoner.get_relevant_patterns(context_cls, top_k=top_k)
        if not relevant:
            return []
        dist = self.l1.reasoner.compose_predictions(relevant, context_cls)
        bucketed: Dict[str, float] = {}
        for idx, prob in enumerate(dist):
            if self.surface_mode == "word" and hasattr(self._adapter, "decode_token"):
                bucket = self._adapter.decode_token(int(idx))
            elif hasattr(self._adapter, "bucket_for_token"):
                bucket = self._adapter.bucket_for_token(int(idx))
            else:
                bucket = self._adapter.decode_class(int(idx))
            bucketed[bucket] = bucketed.get(bucket, 0.0) + float(prob)
        top = sorted(bucketed.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [(name, float(prob)) for name, prob in top]

    def _class_name_to_id(self, class_name: str) -> int:
        for i in range(self._adapter.obs_dim):
            if self._adapter.decode_class(i) == class_name:
                return i
        return 3

    def _tokenize_words(self, text: str) -> List[str]:
        if self.surface_mode == "word" and hasattr(self._adapter, "tokenize"):
            return [tok for tok in self._adapter.tokenize(text) if tok not in {self._adapter.NEWLINE_TOKEN, getattr(self._adapter, "PARA_TOKEN", "<PARA>")}]
        return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", text)

    def _text_to_raw_ids(self, text: str) -> List[int]:
        raw_ids: List[int] = []
        for ch in text:
            if ch == '\n':
                raw_ids.append(94)
            elif 32 <= ord(ch) <= 126:
                raw_ids.append(ord(ch) - 32)
        return raw_ids

    def _surface_ids_from_text(self, text: str) -> List[int]:
        if self.surface_mode == "word" and hasattr(self._adapter, "to_observations"):
            return [int(tok) for tok in self._adapter.to_observations(text, max_length=max(1000, len(text) * 2))]
        return self._text_to_raw_ids(text)

    def _surface_history_text(self) -> str:
        if self.surface_mode == "word" and hasattr(self._adapter, "from_observations"):
            try:
                return self._adapter.from_observations(self._raw_history)
            except Exception:
                pass
        return "".join(chr(v + 32) for v in self._raw_history if 0 <= int(v) <= 94)

    def _surface_label_for_obs(self, obs: int) -> str:
        if self.surface_mode == "word" and hasattr(self._adapter, "decode_token"):
            return str(self._adapter.decode_token(int(obs)))
        if int(obs) == 94:
            return "\n"
        if 0 <= int(obs) <= 94:
            return chr(int(obs) + 32)
        return str(int(obs))

    def _text_agreement(self, generated_text: str, target_text: str) -> float:
        generated_tokens = self._tokenize_words(generated_text.lower())
        target_tokens = self._tokenize_words(target_text.lower())
        if not target_tokens:
            return 0.0
        matches = sum(1 for a, b in zip(generated_tokens, target_tokens) if a == b)
        return matches / len(target_tokens)

    def _text_plausibility(self, text: str) -> float:
        tokens = [tok.lower() for tok in self._tokenize_words(text) if tok.strip()]
        if not tokens:
            return 0.0

        scores: List[float] = []
        word_tokens = [tok for tok in tokens if tok.isalpha()]
        if self.dictionary:
            for tok in word_tokens:
                scores.append(self.dictionary.score_word(tok))
        if self.grammar and word_tokens:
            prev = None
            for tok in word_tokens:
                if prev is not None:
                    scores.append(1.0 if self.grammar.is_valid_transition(prev, tok) else 0.0)
                prev = tok

        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def _sample_feedback_text(self, text: str, weight: float) -> str:
        if weight >= 1.0:
            return text
        if weight <= 0.0:
            return ""
        stride = max(1, int(round(1.0 / weight)))
        sampled = [ch for idx, ch in enumerate(text) if idx % stride == 0]
        return "".join(sampled)

    def _token_class_name(self, token: str) -> str:
        if token.isdigit():
            return 'digit'
        if token.isalpha():
            return 'letter'
        if token == '\n':
            return 'newline'
        if token == ' ':
            return 'space'
        return 'punctuation'

    def _detokenize_words(self, tokens: List[str]) -> str:
        pieces: List[str] = []
        for tok in tokens:
            if not pieces:
                pieces.append(tok)
                continue
            if tok in {'.', ',', ';', ':', '!', '?', ')', ']', '}'}:
                pieces[-1] = pieces[-1].rstrip() + tok
            elif pieces[-1] in {'(', '[', '{'}:
                pieces.append(tok)
            else:
                pieces.append(' ' + tok)
        return ''.join(pieces)

    def _feed_text_back(self, text: str, feedback: Optional[Dict[str, Any]] = None) -> None:
        """Optionally close the loop by letting generated text update the stack."""
        for raw_id in self._surface_ids_from_text(text):
            self.perceive(raw_id, feedback=feedback)

    def _fallback_chars_for_class(self, class_id: int) -> List[str]:
        chars = []
        for raw_id in range(95):
            if self._adapter.encode(raw_id) == class_id:
                chars.append(chr(raw_id + 32))
        return chars

    def _choose_char_for_class(
        self,
        class_id: int,
        prev_char: str | None,
        preferred_char: str | None = None,
    ) -> str:
        candidates = self._class_char_counts.get(class_id)
        if not candidates:
            candidates = Counter()
        if not candidates:
            for ch in self._fallback_chars_for_class(class_id):
                candidates[ch] += 1

        if preferred_char is not None:
            preferred_class = self._adapter.encode_char(preferred_char)
            if preferred_class == class_id:
                candidates = Counter(candidates)
                candidates[preferred_char] += max(1, sum(candidates.values()))

        best_ch = None
        best_score = -np.inf
        total = sum(candidates.values()) + 1e-12
        for ch, count in candidates.items():
            score = np.log(count / total)
            score += 0.3 * np.log((self._char_counts[ch] + 1.0) / (sum(self._char_counts.values()) + 1.0))
            if prev_char is not None:
                score += 0.8 * np.log((self._transition_counts[(prev_char, ch)] + 1.0) /
                                       (self._char_counts[prev_char] + len(candidates) + 1.0))
            if preferred_char is not None and ch == preferred_char:
                score += 1.0
            if score > best_score:
                best_score = score
                best_ch = ch
        return best_ch if best_ch is not None else ' '

    def l1_metrics(self) -> Dict[str, Any]:
        top3 = sorted(self.l1.patterns, key=lambda p: -p.weight)[:3]
        total_w = sum(p.weight for p in top3) + 1e-12
        mi = float(sum(p.weight * p.compression() for p in top3) / total_w) if top3 else 0.0
        return {
            'pop_size': len(self.l1.patterns),
            'mi': mi,
            'stage': self.l1.development.level,
            'best_weight': max(p.weight for p in self.l1.patterns) if self.l1.patterns else 0.0,
        }

    def l2_metrics(self, recent_l1_states: List[int]) -> Dict[str, Any]:
        correct = 0
        total = max(1, len(recent_l1_states) - 1)
        for i in range(len(recent_l1_states) - 1):
            ctx = recent_l1_states[max(0, i - 20):i]
            actual = recent_l1_states[i + 1]
            relevant = self.l2.reasoner.get_relevant_patterns(ctx, top_k=3)
            if relevant:
                dist = self.l2.reasoner.compose_predictions(relevant, ctx)
                if int(np.argmax(dist)) == actual:
                    correct += 1
        top3 = sorted(self.l2.patterns, key=lambda p: -p.weight)[:3]
        total_w = sum(p.weight for p in top3) + 1e-12
        mi = float(sum(p.weight * p.compression() for p in top3) / total_w) if top3 else 0.0
        return {
            'accuracy': correct / total,
            'pop_size': len(self.l2.patterns),
            'mi': mi,
        }

    def l3_metrics(self, recent_l2_states: List[int]) -> Dict[str, Any]:
        correct = 0
        total = max(1, len(recent_l2_states) - 1)
        for i in range(len(recent_l2_states) - 1):
            ctx = recent_l2_states[max(0, i - 20):i]
            actual = recent_l2_states[i + 1]
            relevant = self.l3.reasoner.get_relevant_patterns(ctx, top_k=3)
            if relevant:
                dist = self.l3.reasoner.compose_predictions(relevant, ctx)
                if int(np.argmax(dist)) == actual:
                    correct += 1
        top3 = sorted(self.l3.patterns, key=lambda p: -p.weight)[:3]
        total_w = sum(p.weight for p in top3) + 1e-12
        mi = float(sum(p.weight * p.compression() for p in top3) / total_w) if top3 else 0.0
        return {
            'accuracy': correct / total,
            'pop_size': len(self.l3.patterns),
            'mi': mi,
        }
