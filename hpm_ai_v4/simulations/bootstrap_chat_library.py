"""Build a chat-focused library from a conversational dataset."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from hpm_ai_v4.simulations.build_dailydialog_corpus import build_dailydialog_corpus
from hpm_ai_v4.simulations.build_library import build_library


@dataclass
class ChatBootstrapResult:
    corpus_path: str
    library_path: str
    dialogs_written: int
    lines_written: int


def bootstrap_chat_library(
    *,
    output_dir: str,
    registry_path: Optional[str] = None,
    dataset_name: str = "OpenRL/daily_dialog",
    split: str = "train",
    limit: Optional[int] = 4000,
    max_turns: int = 8,
    steps: int = 20_000,
    min_density: float = 0.2,
    corpus_name: str = "daily_dialog_chat_corpus.txt",
    library_name: str = "daily_dialog_chat_library.pkl",
) -> ChatBootstrapResult:
    os.makedirs(output_dir, exist_ok=True)
    corpus_path = os.path.join(output_dir, corpus_name)
    library_path = os.path.join(output_dir, library_name)

    corpus_result = build_dailydialog_corpus(
        output_path=corpus_path,
        split=split,
        dataset_name=dataset_name,
        limit=limit,
        max_turns=max_turns,
    )

    build_library(
        corpus=corpus_path,
        output=library_path,
        steps=steps,
        min_density=min_density,
        registry_path=registry_path,
        name="daily_dialog_chat_seed",
        domain="chat",
    )

    return ChatBootstrapResult(
        corpus_path=corpus_result.corpus_path,
        library_path=library_path,
        dialogs_written=corpus_result.dialogs_written,
        lines_written=corpus_result.lines_written,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap a chat library from DailyDialog")
    p.add_argument("--output-dir", default="library_bootstrap/chat_dailydialog", help="Directory for corpus and library")
    p.add_argument("--registry", default=None, help="Optional registry JSON path")
    p.add_argument("--dataset-name", default="OpenRL/daily_dialog", help="Hugging Face dataset name")
    p.add_argument("--split", default="train", help="Dataset split to use")
    p.add_argument("--limit", type=int, default=4000, help="Max number of dialogs to include")
    p.add_argument("--max-turns", type=int, default=8, help="Max turns per dialog")
    p.add_argument("--steps", type=int, default=20_000, help="Library build steps")
    p.add_argument("--min-density", type=float, default=0.2, help="Minimum density filter")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = bootstrap_chat_library(
        output_dir=args.output_dir,
        registry_path=args.registry,
        dataset_name=args.dataset_name,
        split=args.split,
        limit=args.limit,
        max_turns=args.max_turns,
        steps=args.steps,
        min_density=args.min_density,
    )
    print(f"[done] corpus:  {result.corpus_path}")
    print(f"[done] library: {result.library_path}")
    print(f"[done] dialogs:  {result.dialogs_written}")
    print(f"[done] lines:    {result.lines_written}")
