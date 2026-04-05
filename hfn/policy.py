"""
Decision policy layer for HFN Observer.

This module centralizes scoring and gating logic so Observer can orchestrate
without embedding multiple ad hoc heuristics.
"""
from __future__ import annotations

from dataclasses import dataclass


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


class DecisionPolicy:
    """Pure decision scoring and gating helpers."""

    def expand_score(self, surprise: float, weight: float) -> float:
        return surprise - weight

    def node_utility(self, accuracy: float, complexity: float, lambda_complexity: float) -> float:
        return accuracy - (lambda_complexity * complexity)

    def should_create(self, ctx: CreateContext) -> bool:
        if ctx.forest_size == 0:
            return True
        if ctx.residual_surprise < ctx.residual_threshold:
            return False
        if ctx.lacunarity_enabled and ctx.density_ratio > ctx.density_factor:
            return False
        return True

    def effective_miss_threshold(self, ctx: AbsorptionContext) -> int:
        threshold = ctx.base_miss_threshold
        if ctx.persistence_guided and ctx.persistence_max > 0:
            norm_p = min(ctx.node_persistence, ctx.persistence_max) / ctx.persistence_max
            threshold = int(round(threshold * (1.0 + norm_p)))
        if ctx.crowding_hotspot:
            threshold = max(1, threshold // 2)
        coherence_scaled = max(1, int(threshold * (0.5 + 0.5 * ctx.coherence)))
        return coherence_scaled

    def compression_threshold(self, base: int, recurrence_threshold: float | None) -> float:
        return recurrence_threshold if recurrence_threshold is not None else float(base)

    def should_compress(self, count: int, threshold: float) -> bool:
        return count >= threshold
