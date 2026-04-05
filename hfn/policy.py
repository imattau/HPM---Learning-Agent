"""
Decision policy layer for HFN Observer.

This module centralizes scoring and gating logic so Observer can orchestrate
without embedding multiple ad hoc heuristics.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchPolicyConfig:
    tau: float = 1.0
    budget: int = 10


@dataclass(frozen=True)
class LearningPolicyConfig:
    alpha_gain: float = 0.1
    beta_loss: float = 0.05
    lambda_complexity: float = 0.1
    weight_decay_rate: float = 0.0
    explainer_relative_accuracy_floor: float = 0.8


@dataclass(frozen=True)
class StructurePolicyConfig:
    absorption_miss_threshold: int = 5
    compression_cooccurrence_threshold: int = 3


@dataclass(frozen=True)
class NoveltyPolicyConfig:
    residual_surprise_threshold: float = 2.0
    gap_query_threshold: float = 0.7
    lacunarity_enabled: bool = False
    lacunarity_creation_factor: float = 2.0


@dataclass(frozen=True)
class CreateContext:
    forest_size: int
    residual_surprise: float
    residual_threshold: float
    lacunarity_enabled: bool
    density_ratio: float
    density_factor: float


@dataclass(frozen=True)
class AbsorptionContext:
    miss_count: int
    base_miss_threshold: int
    persistence_guided: bool
    node_persistence: float
    persistence_max: float
    crowding_hotspot: bool
    coherence: float


@dataclass(frozen=True)
class StructureArbitrationContext:
    create_allowed: bool
    create_score: float
    compress_allowed: bool
    compress_score: float
    gap_allowed: bool


@dataclass(frozen=True)
class StructureActions:
    create_first: bool
    compress_first: bool
    run_gap_query: bool


class DecisionPolicy:
    """Pure decision scoring and gating helpers."""

    def __init__(
        self,
        search: SearchPolicyConfig | None = None,
        learning: LearningPolicyConfig | None = None,
        structure: StructurePolicyConfig | None = None,
        novelty: NoveltyPolicyConfig | None = None,
    ) -> None:
        self.search = search if search is not None else SearchPolicyConfig()
        self.learning = learning if learning is not None else LearningPolicyConfig()
        self.structure = structure if structure is not None else StructurePolicyConfig()
        self.novelty = novelty if novelty is not None else NoveltyPolicyConfig()

    def expand_score(self, surprise: float, weight: float) -> float:
        return surprise - weight

    def node_utility(self, accuracy: float, complexity: float, lambda_complexity: float) -> float:
        return accuracy - (lambda_complexity * complexity)

    def active_explaining_ids(self, accuracy_scores: dict[str, float]) -> set[str]:
        if not accuracy_scores:
            return set()
        best = max(accuracy_scores.values())
        floor = best * self.learning.explainer_relative_accuracy_floor
        return {nid for nid, acc in accuracy_scores.items() if acc >= floor}

    def should_create(self, ctx: CreateContext) -> bool:
        if ctx.forest_size == 0:
            return True
        if ctx.residual_surprise < ctx.residual_threshold:
            return False
        if ctx.lacunarity_enabled and ctx.density_ratio > ctx.density_factor:
            return False
        return True

    def create_score(self, residual_surprise: float, density_ratio: float, lacunarity_enabled: bool) -> float:
        density_penalty = density_ratio if lacunarity_enabled else 0.0
        return residual_surprise - density_penalty

    def effective_miss_threshold(self, ctx: AbsorptionContext) -> int:
        threshold = ctx.base_miss_threshold
        if ctx.persistence_guided and ctx.persistence_max > 0:
            norm_p = min(ctx.node_persistence, ctx.persistence_max) / ctx.persistence_max
            threshold = int(round(threshold * (1.0 + norm_p)))
        if ctx.crowding_hotspot:
            threshold = max(1, threshold // 2)
        coherence_scaled = max(1, int(threshold * (0.5 + 0.5 * ctx.coherence)))
        return coherence_scaled

    def absorb_score(self, miss_count: int, effective_threshold: int, overlap: float) -> float:
        # Positive score means "absorb now" pressure.
        return (miss_count - effective_threshold) + overlap

    def should_absorb(self, absorb_score: float) -> bool:
        return absorb_score >= 0.0

    def weight_update(
        self,
        current_weight: float,
        explaining: bool,
        accuracy: float,
        overlap_sum: float,
    ) -> float:
        if explaining:
            updated = current_weight + self.learning.alpha_gain * (1.0 - current_weight) * accuracy
            return min(updated, 1.0)
        updated = current_weight - self.learning.beta_loss * overlap_sum * current_weight
        return max(updated, 0.0)

    def compression_threshold(self, base: int, recurrence_threshold: float | None) -> float:
        return recurrence_threshold if recurrence_threshold is not None else float(base)

    def compress_score(self, count: int, threshold: float) -> float:
        return float(count) - float(threshold)

    def should_compress(self, count: int, threshold: float) -> bool:
        return count >= threshold

    def should_query_gap(self, coverage_gap: float, gap_query_threshold: float) -> bool:
        return coverage_gap >= gap_query_threshold

    def arbitrate_structure_actions(self, ctx: StructureArbitrationContext) -> StructureActions:
        # Keep both create/compress available, but make ordering explicit.
        create_first = ctx.create_allowed and (ctx.create_score >= ctx.compress_score or not ctx.compress_allowed)
        compress_first = ctx.compress_allowed and (ctx.compress_score > ctx.create_score or not ctx.create_allowed)
        run_gap_query = ctx.gap_allowed and (not ctx.create_allowed and not ctx.compress_allowed)
        return StructureActions(
            create_first=create_first,
            compress_first=compress_first,
            run_gap_query=run_gap_query,
        )
