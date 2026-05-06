"""Long-term pattern archive across episodes."""

from __future__ import annotations

from .engine import PatternEngine
from .pattern import Pattern


def build_context_signature(context: dict) -> str:
    """Derive a compact string key from adapter packet context."""
    if not context:
        return "generic"
    
    # Check for domain-specific hints
    if "view" in context:
        return f"view:{context['view']}"
        
    parts: list[str] = []
    if "env" in context:
        parts.append(f"env:{context['env']}")
    
    return ",".join(parts) if parts else "generic"


class PatternManager:
    """Persistent archive for long-term pattern storage across episodes."""

    def __init__(
        self,
        promotion_threshold: float = 1.0,
        min_support: int = 2,
        max_archive_size: int = 512,
        max_meta_archive_size: int = 128, # V5 Extension
        retrieval_top_k: int = 8,
        archive_decay_rate: float = 0.05,
    ) -> None:
        self.promotion_threshold = promotion_threshold
        self.min_support = min_support
        self.max_archive_size = max_archive_size
        self.max_meta_archive_size = max_meta_archive_size # V5 Extension
        self.retrieval_top_k = retrieval_top_k
        self.archive_decay_rate = archive_decay_rate

        self.archive: dict[str, Pattern] = {}
        self.meta_archive: dict[str, Pattern] = {} # V5 Extension
        self.archive_signatures: dict[str, str] = {}
        self.retrieval_counts: dict[str, int] = {}
        self.episode_count: int = 0

    def promote_from(self, engine: PatternEngine, context: dict | None = None) -> list[str]:
        sig = build_context_signature(context or {})
        promoted: list[str] = []
        
        # 1. Promote Leaf Patterns
        for pattern in engine.store.patterns:
            combined_score = pattern.utility + (0.1 * pattern.support) + (0.5 * pattern.density)
            if combined_score >= self.promotion_threshold and pattern.support >= self.min_support:
                existing = self.archive.get(pattern.name)
                if existing is None or combined_score > (existing.utility + 0.1 * existing.support):
                    self.archive[pattern.name] = pattern
                    self.archive_signatures[pattern.name] = sig
                promoted.append(pattern.name)
        
        # 2. Promote Meta-Patterns
        for meta in engine.store.meta_patterns:
            # Meta-patterns have higher base utility from composition
            if meta.utility >= self.promotion_threshold:
                existing_meta = self.meta_archive.get(meta.name)
                if existing_meta is None or meta.utility > existing_meta.utility:
                    self.meta_archive[meta.name] = meta
                    self.archive_signatures[meta.name] = sig
                promoted.append(meta.name)

        # 3. Prune Archives
        if len(self.archive) > self.max_archive_size:
            ordered = sorted(self.archive.items(), key=lambda item: item[1].utility)
            to_remove = len(self.archive) - self.max_archive_size
            for name, _ in ordered[:to_remove]:
                del self.archive[name]
                self.archive_signatures.pop(name, None)
                
        if len(self.meta_archive) > self.max_meta_archive_size:
            ordered_meta = sorted(self.meta_archive.items(), key=lambda item: item[1].utility)
            to_remove_meta = len(self.meta_archive) - self.max_meta_archive_size
            for name, _ in ordered_meta[:to_remove_meta]:
                del self.meta_archive[name]
                self.archive_signatures.pop(name, None)

        return promoted

    def seed_engine(self, engine: PatternEngine, context: dict | None = None, *, fallback_global: bool = True) -> int:
        target_sig = build_context_signature(context or {})

        # Combine leaf and meta patterns for seeding
        all_candidates = list(self.archive.values()) + list(self.meta_archive.values())
        
        if target_sig != "generic":
            candidates = [
                p for p in all_candidates
                if self.archive_signatures.get(p.name) == target_sig
            ]
        else:
            candidates = all_candidates

        if not candidates and fallback_global:
            candidates = all_candidates

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
        to_delete_leaf = []
        for name, pattern in self.archive.items():
            pattern.decay(utility_decay=self.archive_decay_rate)
            if pattern.utility < 0.1:
                to_delete_leaf.append(name)
        for name in to_delete_leaf:
            del self.archive[name]
            self.archive_signatures.pop(name, None)
            
        to_delete_meta = []
        for name, pattern in self.meta_archive.items():
            pattern.decay(utility_decay=self.archive_decay_rate)
            if pattern.utility < 0.1:
                to_delete_meta.append(name)
        for name in to_delete_meta:
            del self.meta_archive[name]
            self.archive_signatures.pop(name, None)

        pruned = len(to_delete_leaf) + len(to_delete_meta)
        self.episode_count += 1
        return {
            "promoted": len(promoted_names),
            "archive_size": len(self.archive) + len(self.meta_archive),
            "leaf_archive": len(self.archive),
            "meta_archive": len(self.meta_archive),
            "pruned": pruned,
            "episode": self.episode_count,
            "context_signature": sig,
        }

    def start_episode(self, engine: PatternEngine, context: dict | None = None) -> dict:
        sig = build_context_signature(context or {})
        seeded = self.seed_engine(engine, context=context)
        return {
            "seeded": seeded,
            "archive_size": len(self.archive) + len(self.meta_archive),
            "episode": self.episode_count,
            "context_signature": sig,
        }

    def stats(self) -> dict:
        top_leaf = sorted(self.archive.values(), key=lambda p: p.utility, reverse=True)[:3]
        top_meta = sorted(self.meta_archive.values(), key=lambda p: p.utility, reverse=True)[:3]
        total_retrievals = sum(self.retrieval_counts.values())
        return {
            "archive_size": len(self.archive) + len(self.meta_archive),
            "leaf_count": len(self.archive),
            "meta_count": len(self.meta_archive),
            "episode_count": self.episode_count,
            "total_retrievals": total_retrievals,
            "top_leaf_patterns": [p.name for p in top_leaf],
            "top_meta_patterns": [p.name for p in top_meta],
        }
