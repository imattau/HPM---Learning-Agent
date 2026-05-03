"""Central configuration for the v5 core."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CoreConfig:
    """Small, explicit configuration for the v5 core."""

    canonicalization_mode: str = "rotation_compression"
    distance_scale: float = 1.0
    history_limit: int = 10
    exact_threshold: float = 0.0
    near_threshold: float = 1.0
    max_patterns: int = 32
    density_decay: float = 0.01
    utility_decay: float = 0.005
    context_memory_limit: int = 8
    max_sequences: int = 256
