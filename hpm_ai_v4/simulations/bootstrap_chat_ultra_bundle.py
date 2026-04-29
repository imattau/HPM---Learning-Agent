"""Build a coherent stacked chat bundle from the ultra chat library."""
from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.tools.serializer import PatternSerializer


@dataclass
class ChatUltraBundleResult:
    bundle_base: str
    pattern_count: int
    source_library: str
    level_counts: Dict[str, int]


def _clone_and_normalize(patterns: List[object]) -> List[object]:
    if not patterns:
        return []
    clones = []
    total = 0.0
    for idx, pattern in enumerate(patterns):
        clone = copy.deepcopy(pattern)
        clone.id = idx
        clones.append(clone)
        total += max(0.0, float(getattr(clone, "weight", 0.0)))
    if total <= 0.0:
        weight = 1.0 / float(len(clones))
        for clone in clones:
            clone.weight = weight
        return clones
    for clone in clones:
        clone.weight = max(0.0, float(getattr(clone, "weight", 0.0))) / total
    return clones


def _split_by_obs_dim(patterns: List[object], obs_dim: int) -> List[object]:
    return [p for p in patterns if int(getattr(p, "obs_dim", -1)) == obs_dim]


def _split_two_way(patterns: List[object]) -> tuple[List[object], List[object]]:
    ordered = sorted(patterns, key=lambda p: float(getattr(p, "weight", 0.0)), reverse=True)
    mid = max(1, len(ordered) // 2)
    left = ordered[:mid]
    right = ordered[mid:]
    if not right:
        right = left[:]
    return left, right


def bootstrap_chat_ultra_bundle(
    *,
    output_dir: str,
    registry_path: Optional[str] = None,
    source_library: str = "library_bootstrap/chat_ultra/chat_ultra_library.pkl",
) -> ChatUltraBundleResult:
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(source_library):
        raise RuntimeError(f"Missing ultra source library: {source_library}")

    patterns = PatternSerializer.load(source_library)
    groups = {
        "l1": _clone_and_normalize(_split_by_obs_dim(patterns, 5)),
        "l2": [],
        "l3": [],
        "l4": _clone_and_normalize(_split_by_obs_dim(patterns, 32)),
        "l5": _clone_and_normalize(_split_by_obs_dim(patterns, 64)),
    }
    l2_source = _split_by_obs_dim(patterns, 2)
    l2_left, l2_right = _split_two_way(l2_source)
    groups["l2"] = _clone_and_normalize(l2_left)
    groups["l3"] = _clone_and_normalize(l2_right)

    if not groups["l1"] or not groups["l2"] or not groups["l3"] or not groups["l4"] or not groups["l5"]:
        raise RuntimeError("Ultra source library did not yield all required level partitions")

    bundle_base = os.path.join(output_dir, "chat_ultra_bundle")
    from hpm_ai_v4.simulations.layered_agent import LayeredAgent

    stacked = LayeredAgent(num_workers=1, surface_mode="word")
    stacked.l1.patterns = groups["l1"]
    stacked.l2.patterns = groups["l2"]
    stacked.l3.patterns = groups["l3"]
    stacked.l4.patterns = groups["l4"]
    stacked.l5.patterns = groups["l5"]
    stacked.save_bundle(bundle_base)

    level_counts = {level: len(items) for level, items in groups.items()}
    pattern_count = sum(level_counts.values())
    if registry_path:
        registry = LibraryRegistry(registry_path)
        registry.upsert(
            name="chat_ultra_bundle_seed",
            path=bundle_base,
            domain="chat",
            status="seed",
            bundle_kind="stacked",
            level_contract="l1-l5",
            obs_dims=[5, 10, 10, 32, 64],
            source=source_library,
            density_mean=0.0,
            density_min=0.0,
            density_max=0.0,
            pattern_count=pattern_count,
            notes="coherent level-partitioned bundle derived from the ultra chat library",
        )

    return ChatUltraBundleResult(
        bundle_base=bundle_base,
        pattern_count=pattern_count,
        source_library=source_library,
        level_counts=level_counts,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a coherent stacked bundle from the chat ultra library")
    p.add_argument("--output-dir", default="library_bootstrap/chat_ultra_bundle", help="Directory for the bundle")
    p.add_argument("--registry", default=None, help="Optional registry JSON path")
    p.add_argument("--source", default="library_bootstrap/chat_ultra/chat_ultra_library.pkl", help="Ultra source library .pkl")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = bootstrap_chat_ultra_bundle(
        output_dir=args.output_dir,
        registry_path=args.registry,
        source_library=args.source,
    )
    print(f"[done] bundle: {result.bundle_base}")
    print(f"[done] patterns: {result.pattern_count}")
    for level, count in sorted(result.level_counts.items()):
        print(f"[done] {level}: {count}")
