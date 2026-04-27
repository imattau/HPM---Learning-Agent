"""Build a larger conversational library from multiple local and dialogue corpora."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from hpm_ai_v4.simulations.build_library import build_library
from hpm_ai_v4.simulations.build_dailydialog_corpus import build_dailydialog_corpus


@dataclass
class ConversationalBootstrapResult:
    corpus_path: str
    library_path: str
    dialogs_written: int
    lines_written: int


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def _local_corpus_sources() -> Sequence[str]:
    candidates = [
        "/tmp/hpm_dailydialog_chat/daily_dialog_chat_corpus.txt",
        os.path.join("library_bootstrap", "chat_dailydialog", "daily_dialog_chat_corpus.txt"),
        os.path.join("hpm_ai_v4", "simulations", "data", "chat_seed.txt"),
        os.path.join("hpm_ai_v4", "simulations", "data", "wiki_sample.txt"),
        os.path.join("data", "corpus", "peter_rabbit.txt"),
        os.path.join("data", "history_of_science.txt"),
        "README.md",
        os.path.join("hpm_ai_v4", "README.md"),
        os.path.join("docs", "paper", "HPM_Validation_Paper.md"),
        os.path.join("hpm_fractal_node", "README.md"),
        os.path.join("hpm_fractal_node", "docs", "hfn_design_spec.md"),
        os.path.join("hpm_fractal_node", "docs", "2026-03-27-hpm-fractal-node-design.md"),
        os.path.join("hpm_fractal_node", "docs", "2026-03-28-hfn-design-spec.md"),
        os.path.join("hpm_fractal_node", "experiments", "README_nlp.md"),
        os.path.join("hpm_fractal_node", "experiments", "README_code.md"),
        os.path.join("hpm_fractal_node", "experiments", "README_multi_step_reasoning.md"),
        os.path.join("hpm_fractal_node", "experiments", "README_lexical_transfer.md"),
        os.path.join("hpm_fractal_node", "experiments", "README_lexical_math_cross_stream_replay.md"),
        os.path.join("hpm_fractal_node", "experiments", "README_lexical_semantic_forest.md"),
    ]
    return [path for path in candidates if os.path.exists(path)]


def _truncate_text(text: str, limit: Optional[int]) -> str:
    if limit is None or limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].strip()


def _build_local_mixed_corpus(output_path: str, sources: Sequence[str]) -> tuple[str, int, int]:
    segments = []
    for source in sources:
        text = _read_text(source)
        if not text:
            continue
        basename = os.path.basename(source)
        if "daily_dialog_chat_corpus" in basename:
            segments.extend([text, text])
        elif basename == "chat_seed.txt":
            segments.append(text)
        elif basename == "wiki_sample.txt":
            segments.append(_truncate_text(text, 9000))
        elif basename == "peter_rabbit.txt":
            segments.append(_truncate_text(text, 10000))
        elif basename == "history_of_science.txt":
            segments.append(_truncate_text(text, 12000))
        else:
            segments.append(text)

    segments = [segment.strip() for segment in segments if segment and segment.strip()]
    if not segments:
        raise RuntimeError("No local corpus sources available for mixed chat bootstrap")

    mixed_text = "\n\n".join(segments).strip() + "\n"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mixed_text)
    return output_path, len(segments), sum(1 for line in mixed_text.splitlines() if line.strip())


def bootstrap_conversational_library(
    *,
    output_dir: str,
    registry_path: Optional[str] = None,
    steps: int = 28_000,
    min_density: float = 0.2,
    num_initial_patterns: int = 48,
    corpus_name: str = "conversational_chat_corpus.txt",
    library_name: str = "conversational_chat_library.pkl",
) -> ConversationalBootstrapResult:
    os.makedirs(output_dir, exist_ok=True)
    corpus_path = os.path.join(output_dir, corpus_name)
    library_path = os.path.join(output_dir, library_name)

    local_sources = _local_corpus_sources()
    if not local_sources:
        raise RuntimeError("No local chat corpus sources found")

    dailydialog_path = next((path for path in local_sources if "daily_dialog_chat_corpus.txt" in os.path.basename(path)), None)
    if dailydialog_path is None:
        dailydialog_output = os.path.join(output_dir, "daily_dialog_chat_corpus.txt")
        daily_result = build_dailydialog_corpus(
            output_path=dailydialog_output,
            split="train",
            dataset_name="OpenRL/daily_dialog",
            limit=500,
            max_turns=8,
        )
        local_sources = [daily_result.corpus_path, *local_sources]

    corpus_path, dialogs_written, lines_written = _build_local_mixed_corpus(corpus_path, local_sources)

    build_library(
        corpus=corpus_path,
        output=library_path,
        steps=steps,
        min_density=min_density,
        num_initial_patterns=num_initial_patterns,
        registry_path=registry_path,
        name="conversational_chat_seed",
        domain="chat",
    )

    return ConversationalBootstrapResult(
        corpus_path=corpus_path,
        library_path=library_path,
        dialogs_written=dialogs_written,
        lines_written=lines_written,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap a larger conversational library")
    p.add_argument("--output-dir", default="library_bootstrap/chat_mixed", help="Directory for corpus and library")
    p.add_argument("--registry", default=None, help="Optional registry JSON path")
    p.add_argument("--steps", type=int, default=28_000, help="Library build steps")
    p.add_argument("--min-density", type=float, default=0.2, help="Minimum density filter")
    p.add_argument("--num-initial-patterns", type=int, default=48, help="Initial pattern budget for the seed builder")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = bootstrap_conversational_library(
        output_dir=args.output_dir,
        registry_path=args.registry,
        steps=args.steps,
        min_density=args.min_density,
        num_initial_patterns=args.num_initial_patterns,
    )
    print(f"[done] corpus:  {result.corpus_path}")
    print(f"[done] library: {result.library_path}")
    print(f"[done] dialogs:  {result.dialogs_written}")
    print(f"[done] lines:    {result.lines_written}")
