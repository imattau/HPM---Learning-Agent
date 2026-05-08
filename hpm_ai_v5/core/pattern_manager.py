"""Long-term pattern archive across episodes."""

from __future__ import annotations

import pickle  # local-only checkpoints of trusted learned patterns
import pathlib
from .engine import PatternEngine
from .pattern import Pattern
from .variant import make_variant


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
    if "intent_label" in context:
        parts.append(f"intent:{context['intent_label']}")
    # Dominant period buckets
    if "dominant_period" in context:
        dp = context["dominant_period"]
        if dp >= 4:
            parts.append("period:4+")
        else:
            parts.append(f"period:{dp}")
            
    # Entropy buckets
    if "entropy" in context:
        ent = float(context["entropy"])
        if ent > 0.7:
            parts.append("entropy:high")
        elif ent < 0.3:
            parts.append("entropy:low")
    
    return ",".join(parts) if parts else "generic"


class PatternManager:
    """Persistent archive for long-term pattern storage across episodes."""

    def __init__(
        self,
        promotion_threshold: float = 1.0,
        min_support: int = 2,
        max_archive_size: int | None = None,
        max_meta_archive_size: int | None = None,
        retrieval_top_k: int = 8,
        archive_decay_rate: float = 0.05,
    ) -> None:
        self.promotion_threshold = promotion_threshold
        self.min_support = min_support
        # None = dynamically match engine.config.max_patterns at prune time
        self.max_archive_size = max_archive_size
        self.max_meta_archive_size = max_meta_archive_size
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

        # 3. Prune Archives — sizes default to engine.config.max_patterns when None
        max_leaf = self.max_archive_size if self.max_archive_size is not None else engine.config.max_patterns
        max_meta = self.max_meta_archive_size if self.max_meta_archive_size is not None else max(32, engine.config.max_patterns // 8)

        if len(self.archive) > max_leaf:
            ordered = sorted(self.archive.items(), key=lambda item: item[1].utility)
            for name, _ in ordered[:len(self.archive) - max_leaf]:
                del self.archive[name]
                self.archive_signatures.pop(name, None)

        if len(self.meta_archive) > max_meta:
            ordered_meta = sorted(self.meta_archive.items(), key=lambda item: item[1].utility)
            for name, _ in ordered_meta[:len(self.meta_archive) - max_meta]:
                del self.meta_archive[name]
                self.archive_signatures.pop(name, None)

        return promoted

    def _consolidate_variants(self, engine: "PatternEngine") -> int:
        threshold = engine.config.max_patterns * engine.config.consolidation_threshold
        if len(engine.store.patterns) < threshold:
            return 0
        # Use dedicated consolidation distance if set; fall back to near_threshold
        cluster_dist = (
            engine.config.consolidation_distance
            if engine.config.consolidation_distance is not None
            else engine.config.near_threshold
        )
        patterns = list(engine.store.patterns)
        used: set[str] = set()
        promoted = 0
        to_remove: list[str] = []

        for i, p1 in enumerate(patterns):
            if p1.name in used:
                continue
            cluster = [p1]
            for p2 in patterns[i + 1:]:
                if p2.name in used:
                    continue
                dist = p1.distance(
                    p2.template,
                    canonicalization_mode=engine.store.canonicalization_mode,
                    distance_scale=engine.store.distance_scale or 1.0,
                )
                if dist < cluster_dist:
                    cluster.append(p2)
                    used.add(p2.name)
            if len(cluster) >= 2:
                used.add(p1.name)
                vname = f"variant_{len(engine.store.variants)}"
                engine.store.register_variant(make_variant(cluster, name=vname))
                promoted += 1
                to_remove.extend([p.name for p in cluster])

        # Actually remove consolidated patterns to free up space
        for name in to_remove:
            # We don't use store.remove because we want to keep them in the archive/manager
            # but they should leave the engine's "hot" patterns list.
            engine.store.patterns = [p for p in engine.store.patterns if p.name != name]
            engine.store._pattern_index.pop(name, None)

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

        # Use a balanced score for seeding to ensure high-support leaf patterns
        # can compete with composite meta-patterns.
        top = sorted(candidates, key=lambda p: p.utility + 0.1 * p.support, reverse=True)[: self.retrieval_top_k]
        injected = 0
        for pattern in top:
            if engine.store.get(pattern.name) is None:
                engine.store.add(pattern)
                self.retrieval_counts[pattern.name] = self.retrieval_counts.get(pattern.name, 0) + 1
                injected += 1
        return injected

    def end_episode(self, engine: PatternEngine, context: dict | None = None) -> dict:
        promoted_names = self.promote_from(engine, context=context)
        consolidated = self._consolidate_variants(engine)
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
            "consolidated": consolidated,
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

    def save(self, path: str | pathlib.Path) -> None:
        """Persist archive to disk. Saves archive, meta_archive, signatures, counts."""
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({
                "archive": self.archive,
                "meta_archive": self.meta_archive,
                "archive_signatures": self.archive_signatures,
                "retrieval_counts": self.retrieval_counts,
                "episode_count": self.episode_count,
            }, f)

    def load(self, path: str | pathlib.Path) -> bool:
        """Load archive from disk. Returns True if loaded, False if file not found."""
        p = pathlib.Path(path)
        if not p.exists():
            return False
        with open(p, "rb") as f:
            state = pickle.load(f)
        self.archive = state["archive"]
        self.meta_archive = state.get("meta_archive", {})
        self.archive_signatures = state.get("archive_signatures", {})
        self.retrieval_counts = state.get("retrieval_counts", {})
        self.episode_count = state.get("episode_count", 0)
        return True

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

    def archive_stats_by_signature(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(self.archive_signatures.values()))
