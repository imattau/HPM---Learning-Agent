"""Merge the dialogue-focused seed libraries into one larger chat library."""
from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.tools.serializer import PatternSerializer


@dataclass
class ChatSuperBootstrapResult:
    library_path: str
    pattern_count: int
    sources: List[str]


def _default_sources() -> List[str]:
    candidates = [
        "/tmp/hpm_dailydialog_chat/daily_dialog_chat_library.pkl",
        "library_bootstrap/chat_mixed/conversational_chat_library.pkl",
        "library_bootstrap/chat_dailydialog/daily_dialog_chat_library.pkl",
    ]
    return [path for path in candidates if os.path.exists(path)]


def merge_chat_libraries(
    *,
    output_path: str,
    sources: Optional[Sequence[str]] = None,
    registry_path: Optional[str] = None,
) -> ChatSuperBootstrapResult:
    sources = list(sources or _default_sources())
    if not sources:
        raise RuntimeError("No chat seed libraries found to merge")

    merged = []
    next_id = 0
    for source in sources:
        patterns = PatternSerializer.load(source)
        for pattern in patterns:
            clone = copy.deepcopy(pattern)
            clone.id = next_id
            next_id += 1
            merged.append(clone)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    PatternSerializer.save(merged, output_path)

    if registry_path:
        registry = LibraryRegistry(registry_path)
        registry.upsert(
            name="chat_super_seed",
            path=output_path,
            domain="chat",
            status="seed",
            bundle_kind="flat",
            level_contract="l1",
            obs_dims=[5],
            source=";".join(sources),
            density_mean=0.0,
            density_min=0.0,
            density_max=0.0,
            pattern_count=len(merged),
            notes="merged dialogue-focused seeds",
        )

    return ChatSuperBootstrapResult(
        library_path=output_path,
        pattern_count=len(merged),
        sources=sources,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge chat seed libraries into a larger library")
    p.add_argument("--output", default="library_bootstrap/chat_super/chat_super_library.pkl", help="Output .pkl path")
    p.add_argument("--registry", default=None, help="Optional registry JSON path")
    p.add_argument("--source", action="append", default=[], help="Seed library path; may be repeated")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = merge_chat_libraries(
        output_path=args.output,
        sources=args.source or None,
        registry_path=args.registry,
    )
    print(f"[done] merged {len(result.sources)} libraries -> {result.library_path}")
    print(f"[done] patterns: {result.pattern_count}")
