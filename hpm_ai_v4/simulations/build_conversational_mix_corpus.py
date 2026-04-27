"""Build a mixed conversational corpus from several public dialogue datasets."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hpm_ai_v4.simulations.build_dailydialog_corpus import _clean_text


@dataclass(frozen=True)
class DialogueSourceSpec:
    name: str
    repo_id: str
    parquet_file: str
    row_limit: Optional[int]
    max_turns: int = 8


DEFAULT_SOURCES: List[DialogueSourceSpec] = [
    DialogueSourceSpec(
        name="daily_dialog",
        repo_id="OpenRL/daily_dialog",
        parquet_file="data/train-00000-of-00001-f151c79abb2c1fd5.parquet",
        row_limit=500,
        max_turns=8,
    ),
    DialogueSourceSpec(
        name="empathetic_dialogues",
        repo_id="Ahren09/empathetic_dialogues",
        parquet_file="data/train-00000-of-00001.parquet",
        row_limit=800,
        max_turns=8,
    ),
    DialogueSourceSpec(
        name="persona_chat",
        repo_id="Cynaptics/persona-chat",
        parquet_file="data/train-00000-of-00001.parquet",
        row_limit=600,
        max_turns=8,
    ),
]


def _load_parquet_dataset(repo_id: str, parquet_file: str):
    from datasets import load_dataset  # type: ignore
    from huggingface_hub import hf_hub_download  # type: ignore

    file_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=parquet_file)
    return load_dataset("parquet", data_files=file_path, split="train")


def _role_for_index(idx: int, role_hint: Optional[str] = None) -> str:
    if role_hint:
        lower = role_hint.lower()
        if lower.startswith("assistant") or lower.startswith("persona b") or lower in {"b", "bot", "assistant"}:
            return "Assistant"
        if lower.startswith("user") or lower.startswith("persona a") or lower in {"a", "human", "user"}:
            return "User"
    return "User" if idx % 2 == 0 else "Assistant"


def _extract_messages_from_row(row: Dict[str, Any], max_turns: int) -> List[str]:
    dialog = row.get("dialog") or row.get("dialogue") or row.get("utterances") or row.get("messages") or []
    turns: List[str] = []

    if isinstance(dialog, list):
        for idx, turn in enumerate(dialog[:max_turns]):
            if isinstance(turn, dict):
                content = turn.get("content") or turn.get("text") or turn.get("utterance") or turn.get("value") or ""
                role = _role_for_index(idx, str(turn.get("role", "")))
                cleaned = _clean_text(content)
            else:
                cleaned = _clean_text(turn)
                role = _role_for_index(idx)
            if cleaned:
                turns.append(f"{role}: {cleaned}")
        return turns

    if isinstance(dialog, str):
        cleaned = _clean_text(dialog)
        if cleaned:
            turns.append(f"User: {cleaned}")
        return turns

    # Fallback for rows that store a single utterance / response pair
    for key in ("context", "full_topic", "previous_utterance", "reference", "response", "sys_response"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            turns.append(f"User: {_clean_text(value)}")
            break

    return turns[:max_turns]


@dataclass
class MixedDialogueCorpusResult:
    corpus_path: str
    dialogs_written: int
    lines_written: int
    source_counts: Dict[str, int]
    preview: List[str]


def build_conversational_mix_corpus(
    output_path: str,
    *,
    sources: Optional[Sequence[DialogueSourceSpec]] = None,
    max_dialogs_total: Optional[int] = None,
    max_turns: int = 8,
) -> MixedDialogueCorpusResult:
    sources = list(sources or DEFAULT_SOURCES)
    lines: List[str] = []
    preview: List[str] = []
    source_counts: Dict[str, int] = {}
    dialogs_written = 0

    for spec in sources:
        dataset = _load_parquet_dataset(spec.repo_id, spec.parquet_file)
        source_limit = spec.row_limit if spec.row_limit is not None else len(dataset)
        local_count = 0
        for row in dataset:
            if max_dialogs_total is not None and dialogs_written >= max_dialogs_total:
                break
            if local_count >= source_limit:
                break
            dialog_lines = _extract_messages_from_row(row, max_turns=min(max_turns, spec.max_turns))
            if len(dialog_lines) < 2:
                continue
            lines.extend(dialog_lines)
            lines.append("")
            dialogs_written += 1
            local_count += 1
            source_counts[spec.name] = source_counts.get(spec.name, 0) + 1
            if len(preview) < 8:
                preview.extend(dialog_lines[: max(0, 8 - len(preview))])
        if max_dialogs_total is not None and dialogs_written >= max_dialogs_total:
            break

    if not lines:
        raise RuntimeError("No dialogue lines were produced from the selected sources")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    return MixedDialogueCorpusResult(
        corpus_path=output_path,
        dialogs_written=dialogs_written,
        lines_written=sum(1 for line in lines if line.strip()),
        source_counts=source_counts,
        preview=preview,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a mixed conversational corpus")
    p.add_argument("--output", required=True, help="Output corpus path")
    p.add_argument("--max-dialogs", type=int, default=1800, help="Max number of dialogs to include")
    p.add_argument("--max-turns", type=int, default=8, help="Max turns per dialog")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    result = build_conversational_mix_corpus(
        output_path=args.output,
        max_dialogs_total=args.max_dialogs,
        max_turns=args.max_turns,
    )
    print(f"[done] wrote {result.dialogs_written} dialogs / {result.lines_written} lines to {result.corpus_path}")
    print("[done] source counts:")
    for key, value in sorted(result.source_counts.items()):
        print(f"  - {key}: {value}")
    if result.preview:
        print("[preview]")
        for line in result.preview:
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
