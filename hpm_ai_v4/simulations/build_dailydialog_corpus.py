"""Build a dialogue corpus from the DailyDialog dataset."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


DAILYDIALOG_PARQUET_FILES = {
    "train": "data/train-00000-of-00001-f151c79abb2c1fd5.parquet",
    "validation": "data/validation-00000-of-00001-2407eb323af19881.parquet",
    "test": "data/test-00000-of-00001-66dc7d981b70c918.parquet",
}


def _load_dataset(name: str, split: str):
    from datasets import load_dataset  # type: ignore

    if name in {"daily_dialog", "OpenRL/daily_dialog"}:
        from huggingface_hub import hf_hub_download  # type: ignore

        file_path = hf_hub_download(
            repo_id="OpenRL/daily_dialog",
            repo_type="dataset",
            filename=DAILYDIALOG_PARQUET_FILES.get(split, DAILYDIALOG_PARQUET_FILES["train"]),
        )
        return load_dataset("parquet", data_files=file_path, split="train")

    return load_dataset(name, split=split)


def _clean_text(text: str) -> str:
    text = " ".join(str(text).replace("\r", " ").replace("\n", " ").split())
    return text.strip()


def _extract_dialog(dialog_row: Dict[str, Any], max_turns: int = 8) -> List[str]:
    raw_turns = dialog_row.get("dialog") or dialog_row.get("utterances") or []
    turns: List[str] = []
    for idx, turn in enumerate(raw_turns[:max_turns]):
        cleaned = _clean_text(turn)
        if not cleaned:
            continue
        role = "User" if idx % 2 == 0 else "Assistant"
        turns.append(f"{role}: {cleaned}")
    return turns


@dataclass
class DialogueCorpusBuildResult:
    corpus_path: str
    dialogs_written: int
    lines_written: int
    preview: List[str]
    split: str


def build_dailydialog_corpus(
    output_path: str,
    *,
    split: str = "train",
    dataset_name: str = "daily_dialog",
    limit: Optional[int] = None,
    max_turns: int = 8,
) -> DialogueCorpusBuildResult:
    """Convert DailyDialog into a flat User/Assistant corpus."""
    dataset = _load_dataset(dataset_name, split=split)
    lines: List[str] = []
    preview: List[str] = []
    dialogs_written = 0

    for row in dataset:
        if limit is not None and dialogs_written >= limit:
            break
        dialog_lines = _extract_dialog(row, max_turns=max_turns)
        if len(dialog_lines) < 2:
            continue
        lines.extend(dialog_lines)
        lines.append("")
        dialogs_written += 1
        if len(preview) < 6:
            preview.extend(dialog_lines[: max(0, 6 - len(preview))])

    if not lines:
        raise RuntimeError(f"No dialogue lines produced from {dataset_name!r} split {split!r}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    return DialogueCorpusBuildResult(
        corpus_path=output_path,
        dialogs_written=dialogs_written,
        lines_written=sum(1 for line in lines if line.strip()),
        preview=preview,
        split=split,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a DailyDialog corpus")
    p.add_argument("--output", required=True, help="Output corpus path")
    p.add_argument("--split", default="train", help="Dataset split to load")
    p.add_argument("--dataset-name", default="daily_dialog", help="Hugging Face dataset name")
    p.add_argument("--limit", type=int, default=None, help="Max number of dialogs to write")
    p.add_argument("--max-turns", type=int, default=8, help="Max turns per dialog")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_dailydialog_corpus(
        output_path=args.output,
        split=args.split,
        dataset_name=args.dataset_name,
        limit=args.limit,
        max_turns=args.max_turns,
    )
    print(f"[done] wrote {result.dialogs_written} dialogs / {result.lines_written} lines to {result.corpus_path}")
    if result.preview:
        print("[preview]")
        for line in result.preview:
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
