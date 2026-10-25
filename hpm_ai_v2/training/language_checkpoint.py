"""Checkpoint serialization for the HFN language prediction model."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hpm_ai_v2.training.l5_language_meta import PrequentialL5Mixer

if TYPE_CHECKING:
    from hpm_ai_v2.training.language_prediction import HierarchicalNextUnitModel


class LanguageCheckpointMixin:
    def save(self, path: str | Path) -> None:
        payload = {"format": ("hpm-hierarchical-next-unit-v3" if self._raw_counts_complete
                              else "hpm-hierarchical-next-unit-v2"),
                   "max_order": self.max_order, "alpha": self.alpha,
                   "character_word_mix": self.character_word_mix,
                   "char_alpha": self.char_alpha, "min_word_count": self.min_word_count, "vocab": self.vocab,
                   "contexts": [{"context": list(k), "counts": dict(v)} for k, v in sorted(self.counts.items())],
                   "word_counts": dict(self.word_counts), "char_vocab": self.char_vocab,
                   "char_contexts": [{"context": list(k), "counts": dict(v)} for k, v in sorted(self.char_counts.items())],
                   "sentence_counts": dict(self.sentence_counts),
                   "observer_weights": {
                       "word": {str(order): self.observer.get_weight(node.id)
                                for order, node in self.word_level_nodes.items()} if self.observer else {},
                       "character": {str(order): self.observer.get_weight(node.id)
                                     for order, node in self.char_level_nodes.items()} if self.observer else {},
                       "classes": {class_id: self.observer.get_weight(class_id)
                                   for class_id in self.class_node_registry} if self.observer else {},
                   },
                   "training_manifest": self.training_manifest}
        if self.class_nodes_enabled:
            payload["class_nodes_enabled"] = True
            payload["class_tau"] = self.class_tau
            payload["class_lambda"] = self.class_lambda
            payload["class_lambdas"] = self.class_lambdas
            payload["class_min_observations"] = self.class_min_observations
            payload["class_profile_limit"] = self.class_profile_limit
            payload["class_follower_limit"] = self.class_follower_limit
            payload["class_labels"] = self.class_labels
            payload["class_profile_features"] = self.class_profile_features
            payload["class_profiles"] = {key: value.tolist() for key, value in self.class_profiles.items()}
            payload["class_profile_variances"] = {
                key: value.tolist() for key, value in self.class_profile_variances.items()
            }
            payload["class_selection"] = self.class_selection
        if self.l5_language_state is not None:
            payload["l5_language_state"] = self.l5_language_state
            payload["l5_calibration_document_hashes"] = sorted(self.l5_calibration_document_hashes)
        if self._raw_counts_complete:
            payload["raw_contexts"] = [
                {"context": list(k), "counts": dict(v)}
                for k, v in sorted(self.raw_word_context_counts.items())
            ]
        payload["training_document_hashes"] = sorted(self._training_document_hashes)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HierarchicalNextUnitModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        checkpoint_format = payload.get("format")
        if checkpoint_format not in {"hpm-hierarchical-next-unit-v1", "hpm-hierarchical-next-unit-v2",
                                     "hpm-hierarchical-next-unit-v3"}:
            raise ValueError("unsupported HPM language checkpoint format")
        if checkpoint_format == "hpm-hierarchical-next-unit-v3" and "raw_contexts" not in payload:
            raise ValueError("version-3 checkpoint is missing raw lexical context counts")
        model = cls(payload["max_order"], payload["alpha"], payload["min_word_count"],
                    class_nodes=bool(payload.get("class_nodes_enabled", False)),
                    class_tau=payload.get("class_tau", 0.55),
                    class_lambda=float(payload.get("class_lambda", 0.5)),
                    class_min_observations=int(payload.get("class_min_observations", 3)),
                    class_profile_limit=int(payload.get("class_profile_limit", 1500)),
                    class_follower_limit=int(payload.get("class_follower_limit", 2048)))
        model.class_labels = {str(key): str(value) for key, value in payload.get("class_labels", {}).items()}
        model.class_lambdas = {str(key): float(value)
                               for key, value in payload.get("class_lambdas", {}).items()}
        if not model.class_lambdas and model.class_lambda > 0.0:
            # Older opt-in checkpoints used one global mixture coefficient.
            model.class_lambdas = {
                class_id: model.class_lambda
                for class_id in set(model.class_labels.values())
            }
        model.class_profile_features = [str(value) for value in payload.get("class_profile_features", [])]
        model.class_profiles = {str(key): np.asarray(value, dtype=float)
                                for key, value in payload.get("class_profiles", {}).items()}
        model.class_profile_variances = {str(key): np.asarray(value, dtype=float)
                                         for key, value in payload.get("class_profile_variances", {}).items()}
        model.class_selection = payload.get("class_selection")
        model._class_membership_frozen = bool(model.class_labels) or model.class_nodes_enabled
        model.char_alpha = payload.get("char_alpha", model.alpha)
        model.character_word_mix = float(payload.get("character_word_mix", 0.0))
        if not 0.0 <= model.character_word_mix <= 1.0:
            raise ValueError("checkpoint character_word_mix must be between 0 and 1")
        model.vocab = payload["vocab"]
        model.word_to_index = {word: i for i, word in enumerate(model.vocab)}
        model.word_counts = Counter(payload["word_counts"])
        model.counts = defaultdict(Counter, {
            tuple(item["context"]): Counter(item["counts"]) for item in payload["contexts"]
        })
        if "raw_contexts" in payload:
            model.raw_word_context_counts = defaultdict(Counter, {
                tuple(item["context"]): Counter(item["counts"])
                for item in payload["raw_contexts"]
            })
            model._raw_counts_complete = True
        else:
            model.raw_word_context_counts = defaultdict(Counter, {
                context: Counter(counts) for context, counts in model.counts.items()
            })
            model._raw_counts_complete = model.min_word_count == 1
        model.char_vocab = payload["char_vocab"]
        model.char_counts = defaultdict(Counter, {
            tuple(item["context"]): Counter(item["counts"]) for item in payload["char_contexts"]
        })
        model.sentence_counts = Counter(payload.get("sentence_counts", {}))
        model.training_manifest = payload.get("training_manifest")
        if payload.get("l5_language_state") is not None:
            restored_l5 = PrequentialL5Mixer.from_state_dict(payload["l5_language_state"])
            expected_experts = ("observer", *(f"word_order_{order}" for order in range(model.max_order)))
            if restored_l5.experts != expected_experts:
                raise ValueError("checkpoint L5 language experts do not match max_order")
            order_alpha = float(payload["l5_language_state"].get("order_alpha", 0.1))
            if not math.isfinite(order_alpha) or order_alpha <= 0:
                raise ValueError("checkpoint L5 order_alpha must be finite and positive")
            model.l5_language_state = {**restored_l5.state_dict(), "order_alpha": order_alpha}
            hashes = payload.get("l5_calibration_document_hashes", [])
            if any(not isinstance(value, str) or len(value) != 64 for value in hashes):
                raise ValueError("checkpoint L5 calibration document hashes are invalid")
            model.l5_calibration_document_hashes = set(hashes)
        model._training_document_hashes = set(payload.get("training_document_hashes", []))
        model._rebuild_spelling_char_model()
        if model.class_nodes_enabled:
            if not model.class_profile_features and model.class_profiles:
                model._profiles_from_counts(model.raw_word_context_counts)
                if all(len(profile) == len(model._current_profile_features)
                       for profile in model.class_profiles.values()):
                    model.class_profile_features = list(model._current_profile_features)
            model._fit_class_nodes_from_counts(refit=False)
        model._build_hfn_nodes()
        model._initialize_pattern_observer()
        for layer, registry in (("word", model.word_level_nodes),
                                ("character", model.char_level_nodes)):
            for order, weight in payload.get("observer_weights", {}).get(layer, {}).items():
                node = registry.get(int(order))
                if node is not None:
                    model.observer.set_weight(node.id, weight)
        for class_id, weight in payload.get("observer_weights", {}).get("classes", {}).items():
            if class_id in model.class_node_registry:
                model.observer.set_weight(class_id, float(weight))
        return model
