"""Long-term pattern archive across episodes."""

from __future__ import annotations

from .engine import PatternEngine
from .pattern import Pattern


def build_context_signature(context: dict) -> str:
    """Derive a compact string key from adapter packet context."""
    parts: list[str] = []

    if "dominant_period" in context:
        period = context["dominant_period"]
        if period == 0:
            parts.append("period:0")
        elif period == 2:
            parts.append("period:2")
        elif period == 3:
            parts.append("period:3")
        else:
            parts.append("period:4+")

    if "entropy" in context:
        entropy = context["entropy"]
        if entropy < 0.3:
            parts.append("entropy:low")
        elif entropy <= 0.7:
            parts.append("entropy:mid")
        else:
            parts.append("entropy:high")

    if "value_kind" in context:
        parts.append(f"kind:{context['value_kind']}")

    return ",".join(parts) if parts else "generic"


class PatternManager:
    """Persistent archive for long-term pattern storage across episodes."""

    def __init__(
        self,
        promotion_threshold: float = 2.0,
        min_support: int = 3,
        max_archive_size: int = 512,
        retrieval_top_k: int = 8,
        archive_decay_rate: float = 0.05,
    ) -> None:
        self.promotion_threshold = promotion_threshold
        self.min_support = min_support
        self.max_archive_size = max_archive_size
        self.retrieval_top_k = retrieval_top_k
        self.archive_decay_rate = archive_decay_rate

        self.archive: dict[str, Pattern] = {}
        self.archive_signatures: dict[str, str] = {}
        self.retrieval_counts: dict[str, int] = {}
        self.episode_count: int = 0

    def promote_from(self, engine: PatternEngine, context: dict | None = None) -> list[str]:
        sig = build_context_signature(context or {})
        promoted: list[str] = []
        for pattern in engine.store.patterns:
            if pattern.utility >= self.promotion_threshold and pattern.support >= self.min_support:
                existing = self.archive.get(pattern.name)
                if existing is None or pattern.utility > existing.utility:
                    self.archive[pattern.name] = pattern
                    self.archive_signatures[pattern.name] = sig
                promoted.append(pattern.name)

        # prune if over size limit
        if len(self.archive) > self.max_archive_size:
            ordered = sorted(self.archive.items(), key=lambda item: item[1].utility)
            to_remove = len(self.archive) - self.max_archive_size
            for name, _ in ordered[:to_remove]:
                del self.archive[name]
                self.archive_signatures.pop(name, None)

        return promoted

    def seed_engine(self, engine: PatternEngine, context: dict | None = None, *, fallback_global: bool = True) -> int:
        target_sig = build_context_signature(context or {})

        if target_sig != "generic":
            candidates = [
                p for name, p in self.archive.items()
                if self.archive_signatures.get(name) == target_sig
            ]
        else:
            candidates = list(self.archive.values())

        if not candidates and fallback_global:
            candidates = list(self.archive.values())

        top = sorted(candidates, key=lambda p: p.utility, reverse=True)[: self.retrieval_top_k]
        injected = 0
        for pattern in top:
            if engine.store.get(pattern.name) is None:
                engine.store.add(pattern)
                self.retrieval_counts[pattern.name] = self.retrieval_counts.get(pattern.name, 0) + 1
                injected += 1
        return injected

    def end_episode(self, engine: PatternEngine, context: dict | None = None) -> dict:
        promoted_names = self.promote_from(engine, context=context)
        sig = build_context_signature(context or {})

        pruned = 0
        to_delete = []
        for name, pattern in self.archive.items():
            pattern.decay(utility_decay=self.archive_decay_rate)
            if pattern.utility < 0.1:
                to_delete.append(name)
        for name in to_delete:
            del self.archive[name]
            self.archive_signatures.pop(name, None)
        pruned = len(to_delete)

        self.episode_count += 1
        return {
            "promoted": len(promoted_names),
            "archive_size": len(self.archive),
            "pruned": pruned,
            "episode": self.episode_count,
            "context_signature": sig,
        }

    def start_episode(self, engine: PatternEngine, context: dict | None = None) -> dict:
        sig = build_context_signature(context or {})
        seeded = self.seed_engine(engine, context=context)
        return {
            "seeded": seeded,
            "archive_size": len(self.archive),
            "episode": self.episode_count,
            "context_signature": sig,
        }

    def archive_stats_by_signature(self) -> dict[str, int]:
        """Return count of archived patterns per context signature."""
        counts: dict[str, int] = {}
        for name in self.archive:
            sig = self.archive_signatures.get(name, "generic")
            counts[sig] = counts.get(sig, 0) + 1
        return counts

    def stats(self) -> dict:
        top = sorted(self.archive.values(), key=lambda p: p.utility, reverse=True)[:5]
        total_retrievals = sum(self.retrieval_counts.values())
        return {
            "archive_size": len(self.archive),
            "episode_count": self.episode_count,
            "total_retrievals": total_retrievals,
            "top_patterns": [p.name for p in top],
        }
