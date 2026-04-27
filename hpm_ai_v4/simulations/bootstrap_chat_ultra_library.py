"""Build a 500+ pattern chat library by merging dialogue and docs seeds."""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from hpm_ai_v4.simulations.build_library import build_library
from hpm_ai_v4.simulations.bootstrap_chat_super_library import merge_chat_libraries
from hpm_ai_v4.tools.library_registry import LibraryRegistry


@dataclass
class ChatUltraBootstrapResult:
    corpus_paths: List[str]
    library_path: str
    pattern_count: int
    source_libraries: List[str]


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def _write_corpus(output_path: str, sources: Sequence[str]) -> str:
    lines: List[str] = []
    for idx, source in enumerate(sources, start=1):
        if not os.path.exists(source):
            continue
        text = _read_text(source)
        if not text:
            continue
        lines.append(f"# Source {idx}: {os.path.basename(source)}")
        lines.extend(text.splitlines())
        lines.append("")
    if not lines:
        raise RuntimeError(f"No corpus text found for {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return output_path


def _partition(items: Sequence[str], parts: int) -> List[List[str]]:
    items = list(items)
    if parts <= 1 or len(items) <= 1:
        return [items]
    chunks: List[List[str]] = [[] for _ in range(parts)]
    for idx, item in enumerate(items):
        chunks[idx % parts].append(item)
    return [chunk for chunk in chunks if chunk]


def _docs_sources() -> List[str]:
    candidates = sorted(glob.glob(os.path.join("hpm_fractal_node", "experiments", "README_*.md")))
    extras = [
        "README.md",
        os.path.join("hpm_ai_v4", "README.md"),
        os.path.join("docs", "paper", "HPM_Validation_Paper.md"),
        os.path.join("hpm_fractal_node", "README.md"),
        os.path.join("hpm_fractal_node", "docs", "hfn_design_spec.md"),
        os.path.join("hpm_fractal_node", "docs", "2026-03-27-hpm-fractal-node-design.md"),
        os.path.join("hpm_fractal_node", "docs", "2026-03-28-hfn-design-spec.md"),
    ]
    out = [path for path in candidates + extras if os.path.exists(path)]
    return out


def _seed_libraries() -> List[str]:
    candidates = [
        "library_bootstrap/chat_super/chat_super_library.pkl",
        "library_bootstrap/chat_mixed/conversational_chat_library.pkl",
        "library_bootstrap/code_dsl_library.l1.pkl",
        "library_bootstrap/code_dsl_library.l2.pkl",
        "library_bootstrap/code_dsl_library.l3.pkl",
        "library_bootstrap/code_dsl_library.l4.pkl",
        "library_bootstrap/code_dsl_library.l5.pkl",
        "library_bootstrap/structured_text_library.l1.pkl",
        "library_bootstrap/structured_text_library.l2.pkl",
        "library_bootstrap/structured_text_library.l3.pkl",
        "library_bootstrap/structured_text_library.l4.pkl",
        "library_bootstrap/structured_text_library.l5.pkl",
        "library_bootstrap/text_seed.pkl",
        "library_bootstrap/hpm_environment_library.pkl",
        "library_bootstrap/hpm_tool_library.pkl",
        "library_bootstrap/hpm_curriculum_library.pkl",
    ]
    return [path for path in candidates if os.path.exists(path)]


def bootstrap_chat_ultra_library(
    *,
    output_dir: str,
    registry_path: Optional[str] = None,
    steps: int = 6_000,
    min_density: float = -5.0,
    num_initial_patterns: int = 200,
    fast: bool = False,
) -> ChatUltraBootstrapResult:
    os.makedirs(output_dir, exist_ok=True)
    corpus_paths: List[str] = []
    doc_libraries: List[str] = []
    if not fast:
        docs_sources = _docs_sources()
        if len(docs_sources) < 3:
            raise RuntimeError("Not enough docs sources to build the chat ultra library")

        doc_chunks = _partition(docs_sources, 3)
        for idx, chunk in enumerate(doc_chunks, start=1):
            corpus_path = os.path.join(output_dir, f"docs_mix_{idx}.txt")
            library_path = os.path.join(output_dir, f"docs_mix_{idx}.pkl")
            _write_corpus(corpus_path, chunk)
            build_library(
                corpus=corpus_path,
                output=library_path,
                steps=steps,
                min_density=min_density,
                num_initial_patterns=num_initial_patterns,
                registry_path=None,
                name=None,
                domain="chat_docs",
            )
            corpus_paths.append(corpus_path)
            doc_libraries.append(library_path)

    merged_sources = _seed_libraries() + (_seed_libraries() if fast else doc_libraries)
    output_path = os.path.join(output_dir, "chat_ultra_library.pkl")
    result = merge_chat_libraries(
        output_path=output_path,
        sources=merged_sources,
        registry_path=None,
    )
    if registry_path:
        registry = LibraryRegistry(registry_path)
        registry.upsert(
            name="chat_ultra_seed",
            path=output_path,
            domain="chat",
            status="seed",
            source=";".join(merged_sources),
            density_mean=0.0,
            density_min=0.0,
            density_max=0.0,
            pattern_count=result.pattern_count,
            notes="fast-merged ultra chat seed library" if fast else "docs-expanded ultra chat seed library",
        )
    return ChatUltraBootstrapResult(
        corpus_paths=corpus_paths,
        library_path=result.library_path,
        pattern_count=result.pattern_count,
        source_libraries=merged_sources,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a 500+ pattern chat ultra-library")
    p.add_argument("--output-dir", default="library_bootstrap/chat_ultra", help="Directory for corpora and output library")
    p.add_argument("--registry", default=None, help="Optional registry JSON path")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--min-density", type=float, default=-5.0)
    p.add_argument("--num-initial-patterns", type=int, default=200)
    p.add_argument("--fast", action="store_true", help="Skip docs builds and merge seed libraries directly")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = bootstrap_chat_ultra_library(
        output_dir=args.output_dir,
        registry_path=args.registry,
        steps=args.steps,
        min_density=args.min_density,
        num_initial_patterns=args.num_initial_patterns,
        fast=args.fast,
    )
    print(f"[done] patterns: {result.pattern_count}")
    print(f"[done] library: {result.library_path}")
    for corpus_path in result.corpus_paths:
        print(f"[done] corpus: {corpus_path}")
