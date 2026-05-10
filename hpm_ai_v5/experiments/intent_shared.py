"""Shared intent benchmark scaffolding for v5 runners."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable
import math

from hpm_ai_v5.adapter import AdapterPacket
from hpm_ai_v5.adapter.nlp import (
    NLPTokenizer,
    CanonicalPhraser,
    NamedEntityCanonicaliser,
    ContentWordExtractor,
    SkeletonExtractor,
    SkeletonNgramAdapter,
    StartOfEpisodeAdapter,
)
from hpm_ai_v5.adapter.comprehension_feedback import (
    ComprehensionCarryBridgeAdapter,
    ComprehensionFeedbackAdapter,
)
from hpm_ai_v5.adapter.validation_only import ValidationOnlyAdapter
from hpm_ai_v5.core import PatternEngine, PatternManager, PatternStore, PatternStoreProjector
from hpm_ai_v5.pipeline import HPMPipeline
from hpm_ai_v5.polygraphs.nlp import StructuralNLPPolygraphGenerator, InterconnectedNLPPolygraphGenerator
from hpm_ai_v5.agents.atis import _match_target_name


def build_intent_pipeline(
    engine: PatternEngine,
    *,
    label_adapter,
    bridge_adapter=None,
    canonical_override_adapter=None,
    interconnected: bool = False,
    include_bridge_anchors: bool = False,
    polygraph_confidence_skip: float = 1.1,
    view_configs: dict[str, dict[str, float]] | None = None,
) -> tuple[HPMPipeline, object]:
    pipeline = HPMPipeline(
        preprocessor=NLPTokenizer(),
        engine=engine,
        postprocessor=ValidationOnlyAdapter(),
        polygraph_generator=InterconnectedNLPPolygraphGenerator() if interconnected else StructuralNLPPolygraphGenerator(),
        polygraph_confidence_skip=polygraph_confidence_skip,
        view_configs=view_configs or {},
    )
    pipeline.register_preprocessor(StartOfEpisodeAdapter())
    pipeline.register_preprocessor(CanonicalPhraser())
    if canonical_override_adapter is not None:
        pipeline.register_preprocessor(canonical_override_adapter)
    pipeline.register_preprocessor(ComprehensionFeedbackAdapter())
    pipeline.register_preprocessor(NamedEntityCanonicaliser())
    pipeline.register_preprocessor(ContentWordExtractor())
    pipeline.register_preprocessor(SkeletonExtractor())
    pipeline.register_preprocessor(SkeletonNgramAdapter())
    if include_bridge_anchors and bridge_adapter is not None:
        pipeline.register_preprocessor(bridge_adapter)
        pipeline.register_preprocessor(ComprehensionCarryBridgeAdapter())
    pipeline.register_preprocessor(label_adapter)
    return pipeline, label_adapter


def top_k_projection_candidates(
    engine: PatternEngine,
    view,
    *,
    top_k: int = 3,
) -> list[tuple[object, float]]:
    """Build selective weighted projection candidates for a view."""

    ranked = engine.store.top_k(view.state.value, k=top_k)
    candidates: list[tuple[object, float]] = []
    for index, pattern in enumerate(ranked):
        distance = pattern.distance(
            view.state.value,
            canonicalization_mode=engine.store.canonicalization_mode or engine.store.config.canonicalization_mode,
            distance_scale=engine.store.distance_scale or engine.store.config.distance_scale,
        )
        exact_threshold = engine.store.exact_threshold or engine.store.config.exact_threshold
        near_threshold = engine.store.near_threshold or engine.store.config.near_threshold
        if distance <= exact_threshold:
            proximity = 1.0
            base_weight = 1.0
        elif distance <= near_threshold:
            proximity = 1.0 - ((distance - exact_threshold) / max(near_threshold - exact_threshold, 1e-6))
            if proximity < 0.72:
                continue
            base_weight = 0.45 * (proximity ** 2)
        else:
            continue

        if index == 0:
            rank_discount = 1.0
        elif index == 1:
            if proximity < 0.8:
                continue
            rank_discount = 0.2
        else:
            if proximity < 0.92:
                continue
            rank_discount = 0.05

        weight = base_weight * rank_discount
        if weight < 0.03:
            continue
        candidates.append((pattern, weight))
    return candidates


@dataclass
class IntentBenchmarkSupport:
    """Shared reset, training, and vote-finalization helpers for intent benchmarks."""

    engine: PatternEngine
    manager: PatternManager
    pipeline: HPMPipeline
    label_adapter: object
    consolidation: bool = True
    projector: PatternStoreProjector | None = None
    inference_source_weights: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.pattern_intent: dict[str, str] = {}
        self._intent_votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        if self.inference_source_weights is None:
            self.inference_source_weights = {}

    def reset(self) -> None:
        self.engine.current_state = None
        self.engine.history = []
        self.pipeline.view_engines.clear()
        self.pipeline.view_matches.clear()
        self.engine.store = PatternStore(config=self.engine.config)
        self.pattern_intent.clear()
        self._intent_votes.clear()
        for adapter in self.pipeline.preprocessing_pipeline.adapters.values():
            if hasattr(adapter, "reset"):
                adapter.reset()

    def _reset_utterance(self) -> None:
        self.engine.current_state = None
        self.engine.history = []
        for adapter in self.pipeline.preprocessing_pipeline.adapters.values():
            if hasattr(adapter, "reset"):
                adapter.reset()

    @staticmethod
    def _match_weight(status: str) -> float:
        if status == "exact":
            return 1.0
        if status == "near":
            return 0.5
        if status == "variant":
            return 0.5
        return 0.0

    @staticmethod
    def _l2_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        limit = min(len(left), len(right))
        if limit == 0:
            return math.inf
        return math.sqrt(sum((left[i] - right[i]) ** 2 for i in range(limit)))

    def calibrate_content_view(self, train: list[dict]) -> None:
        adapters = self.pipeline.preprocessing_pipeline.adapters
        if "content_word_extractor" not in adapters:
            return

        vectors_by_intent: dict[str, list[tuple[float, ...]]] = defaultdict(list)
        for item in train:
            packet = AdapterPacket(raw=item["text"], context={})
            for adapter in adapters.values():
                if hasattr(adapter, "reset"):
                    adapter.reset()
            packet = self.pipeline.preprocessing_pipeline.run(packet, target_outputs=list(adapters.keys()))
            vector = tuple(packet.context.get("content_vector", ()))
            if vector:
                vectors_by_intent[item["intent"]].append(vector)

        if not vectors_by_intent:
            return

        centroids: dict[str, tuple[float, ...]] = {}
        for intent, vectors in vectors_by_intent.items():
            dim = min(len(v) for v in vectors)
            if dim == 0:
                continue
            centroid = tuple(sum(v[i] for v in vectors) / len(vectors) for i in range(dim))
            centroids[intent] = centroid

        intra_distances: list[float] = []
        inter_distances: list[float] = []
        intents = list(centroids.keys())
        for intent, vectors in vectors_by_intent.items():
            centroid = centroids.get(intent)
            if centroid is None:
                continue
            for vector in vectors:
                intra_distances.append(self._l2_distance(vector, centroid))
        for idx, left_intent in enumerate(intents):
            for right_intent in intents[idx + 1 :]:
                inter_distances.append(self._l2_distance(centroids[left_intent], centroids[right_intent]))

        if not intra_distances:
            return

        intra_distances.sort()
        inter_distances.sort()
        exact_idx = min(len(intra_distances) - 1, max(0, int(0.60 * (len(intra_distances) - 1))))
        near_idx = min(len(intra_distances) - 1, max(0, int(0.90 * (len(intra_distances) - 1))))
        exact_threshold = min(max(intra_distances[exact_idx], 0.015), 0.12)
        near_candidate = max(intra_distances[near_idx] * 1.15, exact_threshold + 0.01)
        if inter_distances:
            nearest_inter = inter_distances[0]
            near_threshold = min(near_candidate, max(exact_threshold + 0.01, nearest_inter * 0.8))
        else:
            near_threshold = near_candidate
        near_threshold = min(max(near_threshold, exact_threshold + 0.01), 0.35)

        self.pipeline.view_configs["content_view"] = {
            "exact_threshold": exact_threshold,
            "near_threshold": near_threshold,
        }

    def train_utterance(self, text: str, intent: str) -> None:
        self._reset_utterance()
        self.label_adapter.label = intent
        result = self.pipeline.step(text)
        self.record_votes(intent)
        if self.projector is not None and self.pipeline.polygraph_generator is not None:
            views = self.pipeline.polygraph_generator.generate(text, context=result.input.context)
            for view in views:
                winner = self.pipeline.view_matches.get(view.name)
                weighted = top_k_projection_candidates(self.engine, view, top_k=3)
                if winner and winner.pattern:
                    weighted = [(winner.pattern, 1.0), *[(pattern, weight) for pattern, weight in weighted if pattern.name != winner.pattern.name]]
                self.projector.observe_candidates(view, weighted[:3], pattern_intent=self.pattern_intent)

    def record_votes(self, intent: str) -> None:
        primary_match = self.engine.last_match
        if primary_match:
            target_name, utility = _match_target_name(primary_match)
            if target_name:
                self._intent_votes[target_name][intent] += self._match_weight(primary_match.status) * max(utility, 0.1)

        for source, match in self.pipeline.view_matches.items():
            if not match:
                continue
            target_name, utility = _match_target_name(match)
            if not target_name:
                continue
            source_weight = self.inference_source_weights.get(source, 1.0)
            contribution = source_weight * self._match_weight(match.status) * max(utility, 0.1)
            self._intent_votes[target_name][intent] += contribution

    def finalize_pattern_intent(self) -> dict[str, str]:
        self.pattern_intent = {
            pname: max(votes, key=votes.__getitem__)
            for pname, votes in self._intent_votes.items()
        }
        return self.pattern_intent

    def run_training(self, train: list[dict], *, shuffle: bool = True) -> None:
        self.reset()
        self.calibrate_content_view(train)
        if shuffle:
            random.shuffle(train)
        self.manager.start_episode(self.engine)
        for item in train:
            self.train_utterance(item["text"], item["intent"])
        if self.consolidation:
            self.manager.end_episode(self.engine)
        self.finalize_pattern_intent()

    def evaluate_predictions(
        self,
        test: list[dict],
        predictor: Callable[[str], str | None],
    ) -> float:
        correct = 0
        for item in test:
            if predictor(item["text"]) == item["intent"]:
                correct += 1
        return correct / max(len(test), 1)
