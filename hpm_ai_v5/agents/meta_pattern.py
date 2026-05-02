"""Meta-pattern discovery for v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _topological_order(nodes: Sequence[str], dependencies: Mapping[str, Sequence[str]]) -> list[str]:
    remaining = {node: set(dependencies.get(node, ())) for node in nodes}
    ordered: list[str] = []
    available = sorted(node for node in nodes if not remaining[node])
    while available:
        current = available.pop(0)
        if current in ordered:
            continue
        ordered.append(current)
        for node, prereqs in remaining.items():
            if current in prereqs:
                prereqs.remove(current)
                if not prereqs and node not in ordered and node not in available:
                    available.append(node)
        available.sort()
    for node in nodes:
        if node not in ordered:
            ordered.append(node)
    return ordered


def _canonical_dependencies(nodes: Sequence[str], dependencies: Mapping[str, Sequence[str]]) -> tuple[tuple[int, int], ...]:
    ordered = _topological_order(nodes, dependencies)
    index = {node: position for position, node in enumerate(ordered)}
    edges: list[tuple[int, int]] = []
    for node, prereqs in dependencies.items():
        for prereq in prereqs:
            if prereq in index and node in index:
                edges.append((index[prereq], index[node]))
    return tuple(sorted(edges))


@dataclass(frozen=True, slots=True)
class MetaPattern:
    signature: str
    structure: str
    length: int
    dependencies: tuple[tuple[int, int], ...]
    placeholder_names: tuple[str, ...]
    reward_ranks: tuple[float, ...]
    exemplars: tuple[str, ...] = ()
    confidence: float = 0.0

    def instantiate(self, concrete_names: Sequence[str]) -> list[str]:
        ordered = list(concrete_names[: self.length])
        if len(ordered) < self.length:
            return []
        return [f"collect_{name}" for name in ordered]


@dataclass(frozen=True, slots=True)
class MetaPatternDecision:
    signature: str | None
    confidence: float
    plan: list[str]
    matched: bool
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaPatternDiscoveryAgent:
    """Discover reusable meta-patterns across structurally similar tasks."""

    meta_patterns: dict[str, MetaPattern] = field(default_factory=dict)
    exemplar_count: int = 0

    @staticmethod
    def _signature(nodes: Sequence[str], dependencies: Mapping[str, Sequence[str]]) -> str:
        edges = _canonical_dependencies(nodes, dependencies)
        return f"chain:{len(nodes)}|edges={edges}"

    @staticmethod
    def _reward_ranks(rewards: Mapping[str, float], order: Sequence[str]) -> tuple[float, ...]:
        values = [float(rewards.get(name, 0.0)) for name in order]
        if not values:
            return ()
        low = min(values)
        high = max(values)
        if abs(high - low) <= 1e-9:
            return tuple(0.5 for _ in values)
        scale = high - low
        return tuple((value - low) / scale for value in values)

    def observe(self, task_name: str, items: Sequence[str], dependencies: Mapping[str, Sequence[str]], rewards: Mapping[str, float]) -> MetaPatternDecision:
        signature = self._signature(items, dependencies)
        order = _topological_order(items, dependencies)
        reward_ranks = self._reward_ranks(rewards, order)
        if signature not in self.meta_patterns:
            pattern = MetaPattern(
                signature=signature,
                structure="chain",
                length=len(order),
                dependencies=_canonical_dependencies(items, dependencies),
                placeholder_names=tuple(f"slot_{index}" for index in range(len(order))),
                reward_ranks=reward_ranks,
                exemplars=(task_name,),
                confidence=0.75 if len(order) >= 3 else 0.5,
            )
            self.meta_patterns[signature] = pattern
            matched = False
        else:
            pattern = self.meta_patterns[signature]
            matched = True
            exemplars = tuple(sorted({*pattern.exemplars, task_name}))
            confidence = min(1.0, pattern.confidence + 0.05)
            self.meta_patterns[signature] = MetaPattern(
                signature=pattern.signature,
                structure=pattern.structure,
                length=pattern.length,
                dependencies=pattern.dependencies,
                placeholder_names=pattern.placeholder_names,
                reward_ranks=pattern.reward_ranks,
                exemplars=exemplars,
                confidence=confidence,
            )
        self.exemplar_count = len(self.meta_patterns)
        return MetaPatternDecision(
            signature=signature,
            confidence=self.meta_patterns[signature].confidence,
            plan=[f"collect_{name}" for name in order],
            matched=matched,
            trace={
                "task": task_name,
                "signature": signature,
                "order": list(order),
                "reward_ranks": list(reward_ranks),
                "exemplars": list(self.meta_patterns[signature].exemplars),
            },
        )

    def solve(self, task_name: str, items: Sequence[str], dependencies: Mapping[str, Sequence[str]], rewards: Mapping[str, float]) -> MetaPatternDecision:
        signature = self._signature(items, dependencies)
        order = _topological_order(items, dependencies)
        pattern = self.meta_patterns.get(signature)
        if pattern is None:
            return MetaPatternDecision(
                signature=None,
                confidence=0.0,
                plan=[f"collect_{name}" for name in order],
                matched=False,
                trace={
                    "task": task_name,
                    "reason": "no_meta_pattern",
                    "order": list(order),
                    "signature": signature,
                },
            )
        plan = pattern.instantiate(order)
        confidence = min(1.0, pattern.confidence + 0.15)
        if len(order) >= 3:
            confidence = max(confidence, 0.85)
        return MetaPatternDecision(
            signature=pattern.signature,
            confidence=confidence,
            plan=plan,
            matched=True,
            trace={
                "task": task_name,
                "signature": pattern.signature,
                "order": list(order),
                "meta_signature": pattern.signature,
                "reward_ranks": list(pattern.reward_ranks),
                "exemplars": list(pattern.exemplars),
            },
        )
