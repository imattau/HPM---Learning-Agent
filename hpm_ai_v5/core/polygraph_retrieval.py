"""Polygraph-backed pattern retrieval and store mapping for v5."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .pattern import Pattern
from .store import MatchResult, PatternStore
from ..polygraphs.base import PolygraphView


@dataclass(frozen=True, slots=True)
class PolygraphCandidate:
    """Ranked retrieval candidate backed by multi-view support."""

    pattern_name: str
    total_score: float
    agreement_count: int
    view_scores: dict[str, float] = field(default_factory=dict)
    anchor_support: dict[str, float] = field(default_factory=dict)
    concept_support: dict[str, float] = field(default_factory=dict)
    concept_overlap_count: int = 0
    specific_concept_count: int = 0
    pattern: Pattern | None = None


@dataclass(frozen=True, slots=True)
class ProjectedPattern:
    """Per-pattern multi-view metadata accumulated during training."""

    pattern_name: str
    view_names: tuple[str, ...] = field(default_factory=tuple)
    leaf_keys: tuple[str, ...] = field(default_factory=tuple)
    anchor_ids: tuple[str, ...] = field(default_factory=tuple)
    concept_ids: tuple[str, ...] = field(default_factory=tuple)
    total_support: float = 0.0
    view_support: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PatternStoreMap:
    """Simple diagnostic map of projected pattern-store structure."""

    projected_patterns: dict[str, ProjectedPattern]
    redundancy_clusters: list[tuple[str, ...]] = field(default_factory=list)
    ambiguity_clusters: list[tuple[str, ...]] = field(default_factory=list)
    isolated_patterns: tuple[str, ...] = field(default_factory=tuple)
    bridge_hubs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    concept_hubs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    view_coverage: dict[str, int] = field(default_factory=dict)
    avg_patterns_per_anchor: float = 0.0
    avg_patterns_per_concept: float = 0.0


@dataclass
class PatternStoreProjector:
    """Build lightweight multi-view projections over the canonical pattern store."""

    projected_patterns: dict[str, ProjectedPattern] = field(default_factory=dict)
    _pattern_views: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set), init=False)
    _pattern_leaf_keys: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set), init=False)
    _pattern_anchor_ids: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set), init=False)
    _pattern_concept_ids: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set), init=False)
    _pattern_total_support: dict[str, float] = field(default_factory=lambda: defaultdict(float), init=False)
    _pattern_view_support: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)), init=False)
    _view_patterns: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set), init=False)
    _anchor_patterns: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set), init=False)
    _concept_patterns: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set), init=False)
    _intent_patterns: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set), init=False)

    def observe(self, views: Iterable[PolygraphView], view_matches: dict[str, MatchResult], pattern_intent: dict[str, str] | None = None) -> None:
        pattern_intent = pattern_intent or {}
        for view in views:
            match = view_matches.get(view.name)
            if not match or not match.pattern:
                continue
            self.observe_candidates(
                view,
                [(match.pattern, 1.0)],
                pattern_intent=pattern_intent,
            )

    def observe_candidates(
        self,
        view: PolygraphView,
        candidates: Iterable[tuple[Pattern, float]],
        *,
        pattern_intent: dict[str, str] | None = None,
    ) -> None:
        pattern_intent = pattern_intent or {}
        for pattern, weight in candidates:
            if pattern is None or weight <= 0.0:
                continue
            pattern_name = pattern.name
            self._pattern_views[pattern_name].add(view.name)
            self._pattern_leaf_keys[pattern_name].update(view.leaf_keys)
            self._pattern_anchor_ids[pattern_name].update(view.anchor_ids)
            self._pattern_concept_ids[pattern_name].update(view.concept_ids)
            self._pattern_total_support[pattern_name] += weight
            self._pattern_view_support[pattern_name][view.name] += weight
            self._view_patterns[view.name].add(pattern_name)
            for anchor_id in view.anchor_ids:
                self._anchor_patterns[anchor_id].add(pattern_name)
            for concept_id in view.concept_ids:
                self._concept_patterns[concept_id].add(pattern_name)
            intent = pattern_intent.get(pattern_name)
            if intent:
                self._intent_patterns[intent].add(pattern_name)
            self.projected_patterns[pattern_name] = ProjectedPattern(
                pattern_name=pattern_name,
                view_names=tuple(sorted(self._pattern_views[pattern_name])),
                leaf_keys=tuple(sorted(self._pattern_leaf_keys[pattern_name])),
                anchor_ids=tuple(sorted(self._pattern_anchor_ids[pattern_name])),
                concept_ids=tuple(sorted(self._pattern_concept_ids[pattern_name])),
                total_support=self._pattern_total_support[pattern_name],
                view_support=dict(self._pattern_view_support[pattern_name]),
            )

    def view_patterns(self, view_name: str) -> set[str]:
        return set(self._view_patterns.get(view_name, ()))

    def build_map(self) -> PatternStoreMap:
        redundancy_clusters = [
            tuple(sorted(patterns))
            for patterns in self._anchor_patterns.values()
            if len(patterns) >= 2
        ]
        ambiguity_clusters = [
            tuple(sorted(patterns))
            for patterns in self._anchor_patterns.values()
            if len({intent for intent, owned in self._intent_patterns.items() if owned.intersection(patterns)}) >= 2
        ]
        isolated = tuple(sorted(
            pattern_name
            for pattern_name, projection in self.projected_patterns.items()
            if len(projection.view_names) <= 1
        ))
        bridge_hubs = {
            anchor_id: tuple(sorted(patterns))
            for anchor_id, patterns in self._anchor_patterns.items()
            if len(patterns) >= 2
        }
        concept_hubs = {
            concept_id: tuple(sorted(patterns))
            for concept_id, patterns in self._concept_patterns.items()
            if len(patterns) >= 2
        }
        view_coverage = {view_name: len(patterns) for view_name, patterns in self._view_patterns.items()}
        avg_patterns_per_anchor = (
            sum(len(patterns) for patterns in self._anchor_patterns.values()) / max(len(self._anchor_patterns), 1)
        )
        avg_patterns_per_concept = (
            sum(len(patterns) for patterns in self._concept_patterns.values()) / max(len(self._concept_patterns), 1)
        )
        return PatternStoreMap(
            projected_patterns=dict(self.projected_patterns),
            redundancy_clusters=sorted(set(redundancy_clusters)),
            ambiguity_clusters=sorted(set(ambiguity_clusters)),
            isolated_patterns=isolated,
            bridge_hubs=bridge_hubs,
            concept_hubs=concept_hubs,
            view_coverage=view_coverage,
            avg_patterns_per_anchor=avg_patterns_per_anchor,
            avg_patterns_per_concept=avg_patterns_per_concept,
        )


@dataclass(slots=True)
class PolygraphPatternRetriever:
    """Retrieve candidate patterns via multi-view support rather than a single distance."""

    store: PatternStore
    projector: PatternStoreProjector
    use_concept_scoring: bool = False
    pattern_intent: dict[str, str] = field(default_factory=dict)
    use_dialogue_priors: bool = False
    use_expected_intent_injection: bool = False
    broad_concepts: frozenset[str] = field(default_factory=lambda: frozenset({
        "concept::listing_request",
        "concept::availability_request",
        "concept::entity_constraint",
    }))

    def _projection_view_prior(self, pattern_name: str, view_name: str) -> float:
        projection = self.projector.projected_patterns.get(pattern_name)
        if projection is None:
            return 0.0
        support = projection.view_support.get(view_name, 0.0)
        if support <= 0.0:
            return 0.0
        return min(0.03, 0.01 * math.log1p(support))

    def _projection_total_prior(self, pattern_name: str) -> float:
        projection = self.projector.projected_patterns.get(pattern_name)
        if projection is None or projection.total_support <= 0.0:
            return 0.0
        return min(0.05, 0.015 * math.log1p(projection.total_support))

    def _anchor_weight(self, anchor_id: str) -> float:
        pattern_count = len(self.projector._anchor_patterns.get(anchor_id, ()))
        if pattern_count <= 1:
            rarity = 1.0
        else:
            rarity = 1.0 / math.sqrt(pattern_count)
        if anchor_id == "intent::utterance":
            return 0.05 * rarity
        return min(1.0, 0.35 + (0.65 * rarity))

    def _concept_weight(self, concept_id: str) -> float:
        pattern_count = len(self.projector._concept_patterns.get(concept_id, ()))
        rarity = 1.0 if pattern_count <= 1 else 1.0 / math.sqrt(pattern_count)
        if concept_id in self.broad_concepts:
            return 0.03 * rarity
        return min(0.45, 0.18 + (0.27 * rarity))

    def _dialogue_intent_prior(self, pattern_name: str, view: PolygraphView) -> float:
        if not self.use_dialogue_priors:
            return 0.0
        intent = self.pattern_intent.get(pattern_name)
        if not intent:
            return 0.0
        bias = 0.0
        for anchor_id in view.anchor_ids:
            if anchor_id.startswith("dialogue::kb_expected::"):
                expected_intent = anchor_id.rsplit("::", 1)[-1]
                bias += 0.22 if intent == expected_intent else -0.08
            elif anchor_id.startswith("dialogue::route::"):
                route_intent = anchor_id.rsplit("::", 1)[-1]
                bias += 0.10 if intent == route_intent else -0.03
            elif anchor_id.startswith("dialogue::focus::"):
                focus_intent = anchor_id.rsplit("::", 1)[-1]
                bias += 0.08 if intent == focus_intent else -0.02
        return bias

    def _expected_intents(self, view: PolygraphView) -> tuple[str, ...]:
        if not self.use_expected_intent_injection:
            return ()
        expected: list[str] = []
        for anchor_id in view.anchor_ids:
            if anchor_id.startswith("dialogue::kb_expected::"):
                expected.append(anchor_id.rsplit("::", 1)[-1])
            elif anchor_id.startswith("dialogue::route::"):
                expected.append(anchor_id.rsplit("::", 1)[-1])
            elif anchor_id.startswith("dialogue::focus::"):
                expected.append(anchor_id.rsplit("::", 1)[-1])
        return tuple(dict.fromkeys(expected))

    def _inject_expected_intent_candidates(
        self,
        view: PolygraphView,
        candidate_view_scores: dict[str, dict[str, float]],
        candidate_anchor_support: dict[str, dict[str, float]],
    ) -> None:
        expected_intents = self._expected_intents(view)
        if not expected_intents:
            return
        for intent in expected_intents:
            pattern_names = sorted(
                self.projector._intent_patterns.get(intent, ()),
                key=lambda pattern_name: self.projector._pattern_total_support.get(pattern_name, 0.0),
                reverse=True,
            )[:3]
            for pattern_name in pattern_names:
                pattern = self.store.get(pattern_name)
                if not pattern:
                    continue
                injected_score = (
                    0.12
                    + self._projection_view_prior(pattern_name, view.name)
                    + self._projection_total_prior(pattern_name)
                    + max(0.0, self._dialogue_intent_prior(pattern_name, view))
                )
                candidate_view_scores[pattern_name][view.name] = max(
                    candidate_view_scores[pattern_name].get(view.name, 0.0),
                    injected_score,
                )
                for anchor_id in view.anchor_ids:
                    if anchor_id.startswith("dialogue::"):
                        candidate_anchor_support[pattern_name][anchor_id] += 0.05 * self._anchor_weight(anchor_id)

    def _view_score(self, view: PolygraphView, pattern: Pattern) -> float:
        distance = pattern.distance(
            view.state.value,
            canonicalization_mode=self.store.canonicalization_mode or self.store.config.canonicalization_mode,
            distance_scale=self.store.distance_scale or self.store.config.distance_scale,
        )
        exact_threshold = self.store.exact_threshold or self.store.config.exact_threshold
        near_threshold = self.store.near_threshold or self.store.config.near_threshold
        if distance <= exact_threshold:
            return 1.0
        if distance <= near_threshold:
            return max(0.1, 1.0 - (distance / max(near_threshold, 1e-6)))
        return 0.0

    def retrieve(self, views: Iterable[PolygraphView], *, top_k: int = 3) -> list[PolygraphCandidate]:
        candidate_view_scores: dict[str, dict[str, float]] = defaultdict(dict)
        candidate_anchor_support: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        candidate_concept_support: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        candidate_concept_overlap_count: dict[str, int] = defaultdict(int)
        candidate_specific_concept_count: dict[str, int] = defaultdict(int)

        for view in views:
            pattern_names = self.projector.view_patterns(view.name)
            for pattern_name in pattern_names:
                pattern = self.store.get(pattern_name)
                if not pattern:
                    continue
                score = self._view_score(view, pattern)
                if score <= 0.0:
                    continue
                candidate_view_scores[pattern_name][view.name] = max(
                    candidate_view_scores[pattern_name].get(view.name, 0.0),
                    score
                    + self._projection_view_prior(pattern_name, view.name)
                    + self._dialogue_intent_prior(pattern_name, view),
                )
                for anchor_id in view.anchor_ids:
                    candidate_anchor_support[pattern_name][anchor_id] += score * self._anchor_weight(anchor_id)
                if self.use_concept_scoring:
                    projection = self.projector.projected_patterns.get(pattern_name)
                    if projection is not None:
                        shared_concepts = set(view.concept_ids).intersection(projection.concept_ids)
                        if shared_concepts:
                            candidate_concept_overlap_count[pattern_name] += len(shared_concepts)
                        specific_shared = [concept_id for concept_id in shared_concepts if concept_id not in self.broad_concepts]
                        if specific_shared:
                            candidate_specific_concept_count[pattern_name] += len(specific_shared)
                            for concept_id in specific_shared:
                                candidate_concept_support[pattern_name][concept_id] += score * self._concept_weight(concept_id)
                            for concept_id in shared_concepts.intersection(self.broad_concepts):
                                candidate_concept_support[pattern_name][concept_id] += score * (0.35 * self._concept_weight(concept_id))
                        elif shared_concepts and len(view.anchor_ids) >= 2:
                            for concept_id in shared_concepts:
                                candidate_concept_support[pattern_name][concept_id] += score * (0.2 * self._concept_weight(concept_id))
            self._inject_expected_intent_candidates(
                view,
                candidate_view_scores,
                candidate_anchor_support,
            )

        ranked: list[PolygraphCandidate] = []
        for pattern_name, view_scores in candidate_view_scores.items():
            agreement_count = len(view_scores)
            total_score = sum(view_scores.values())
            if agreement_count >= 2:
                total_score += 0.25 * agreement_count
            if candidate_anchor_support.get(pattern_name):
                total_score += 0.08 * sum(candidate_anchor_support[pattern_name].values())
            if self.use_concept_scoring and candidate_concept_support.get(pattern_name):
                concept_total = sum(candidate_concept_support[pattern_name].values())
                if len(candidate_concept_support[pattern_name]) >= 2:
                    concept_total *= 1.10
                if candidate_specific_concept_count.get(pattern_name, 0) == 0 and agreement_count < 2:
                    concept_total *= 0.25
                total_score += concept_total
            total_score += self._projection_total_prior(pattern_name)
            ranked.append(PolygraphCandidate(
                pattern_name=pattern_name,
                total_score=total_score,
                agreement_count=agreement_count,
                view_scores=dict(view_scores),
                anchor_support=dict(candidate_anchor_support.get(pattern_name, {})),
                concept_support=dict(candidate_concept_support.get(pattern_name, {})),
                concept_overlap_count=candidate_concept_overlap_count.get(pattern_name, 0),
                specific_concept_count=candidate_specific_concept_count.get(pattern_name, 0),
                pattern=self.store.get(pattern_name),
            ))

        ranked.sort(key=lambda candidate: (-candidate.total_score, -candidate.agreement_count, candidate.pattern_name))
        return ranked[:max(0, top_k)]
