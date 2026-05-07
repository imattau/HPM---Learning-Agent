"""PatternVariant — promoted abstraction over near-identical patterns."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .pattern import Pattern


@dataclass
class PatternVariant:
    name: str
    member_names: list[str]
    centroid: tuple[float, ...]
    hit_count: int
    context_signature: dict[str, Any]
    score: float


def make_variant(patterns: list[Pattern], name: str) -> PatternVariant:
    templates = [p.template for p in patterns if p.template]
    if not templates:
        centroid: tuple[float, ...] = ()
    else:
        min_len = min(len(t) for t in templates)
        arr = np.array([list(t[:min_len]) for t in templates], dtype=float)
        centroid = tuple(float(x) for x in arr.mean(axis=0))
    hit_count = sum(p.support for p in patterns)
    ctx: dict[str, Any] = {}
    for p in patterns:
        if hasattr(p, "context") and p.context:
            ctx.update(p.context)
    score = float(np.mean([p.utility for p in patterns]))
    return PatternVariant(
        name=name,
        member_names=[p.name for p in patterns],
        centroid=centroid,
        hit_count=hit_count,
        context_signature=ctx,
        score=score,
    )
