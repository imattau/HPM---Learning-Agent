from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Optional


_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "quiz_banks" / "generated"
_ARC_DATASET = "allenai/ai2_arc"
_MMLU_DATASET = "cais/mmlu"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _dump_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _hf_tree(repo_id: str, path: str = "") -> list[dict]:
    """Fetch a dataset repo tree from the Hugging Face API."""
    base = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"
    url = f"{base}/{path}" if path else base
    if "?" in url:
        url = f"{url}&recursive=1"
    else:
        url = f"{url}?recursive=1"
    raw = subprocess.check_output(
        ["curl", "-sS", "--fail", "--max-time", "120", url],
        text=True,
    )
    data = json.loads(raw)
    return data if isinstance(data, list) else []


def _repo_parquet_paths(repo_id: str, path_prefix: str = "") -> list[str]:
    """Return parquet file paths from a Hugging Face dataset repo tree."""
    entries = _hf_tree(repo_id, path_prefix)
    paths: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path", ""))
        if path.endswith(".parquet"):
            paths.append(path)
    return sorted(paths)


def _read_parquet_rows(repo_id: str, path: str) -> list[dict]:
    """Download a parquet file from the hub and return rows as dictionaries."""
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise ImportError("pyarrow is required for parquet import") from exc

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.check_call(
            ["curl", "-L", "-sS", "--fail", "--max-time", "300", url, "-o", str(tmp_path)],
        )
        table = pq.read_table(tmp_path)
        return table.to_pylist()
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _coerce_answer_index(answer: object, labels: list[str], options: list[str]) -> Optional[int]:
    if isinstance(answer, int):
        return answer if 0 <= answer < len(options) else None
    answer_text = str(answer).strip()
    if not answer_text:
        return None
    if answer_text in labels:
        return labels.index(answer_text)
    if answer_text.isdigit():
        idx = int(answer_text)
        if 0 <= idx < len(options):
            return idx
    for idx, option in enumerate(options):
        if option == answer_text:
            return idx
    return None


def _arc_example_to_item(example: dict, difficulty: str, source: str) -> Optional[dict]:
    question = str(example.get("question", "")).strip()
    choices = example.get("choices") or {}
    labels = [str(label).strip() for label in choices.get("label", [])]
    texts = [str(text).strip() for text in choices.get("text", [])]
    if not question or len(texts) != 4:
        return None

    answer_key = example.get("answerKey")
    correct_index = _coerce_answer_index(answer_key, labels, texts)
    if correct_index is None:
        return None

    return {
        "id": f"arc_{difficulty}_{_slug(question)[:80] or 'question'}",
        "question": question,
        "options": texts,
        "correct_index": correct_index,
        "topic": "science" if difficulty != "easy" else "science",
        "explanation": f"ARC {difficulty} answer: {texts[correct_index]}.",
        "difficulty": difficulty,
        "source": source,
    }


def _mmlu_example_to_item(example: dict, subject: str) -> Optional[dict]:
    question = str(example.get("question", "")).strip()
    options = [str(opt).strip() for opt in (example.get("choices") or [])]
    if not question or len(options) != 4:
        return None

    correct_index = _coerce_answer_index(example.get("answer"), ["A", "B", "C", "D"], options)
    if correct_index is None:
        return None

    topic = subject.replace("_", " ").strip() or "general knowledge"
    return {
        "id": f"mmlu_{_slug(subject)}_{_slug(question)[:80] or 'question'}",
        "question": question,
        "options": options,
        "correct_index": correct_index,
        "topic": topic,
        "explanation": f"MMLU {subject} answer: {options[correct_index]}.",
        "difficulty": "hard",
        "source": f"mmlu:{subject}",
    }


def _merge_unique(existing: list[dict], additions: Iterable[dict]) -> list[dict]:
    seen = {
        (
            str(item.get("question", "")).strip().lower(),
            tuple(item.get("options", [])),
            int(item.get("correct_index", -1)),
        )
        for item in existing
    }
    merged = list(existing)
    for item in additions:
        key = (
            str(item.get("question", "")).strip().lower(),
            tuple(item.get("options", [])),
            int(item.get("correct_index", -1)),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def build_arc_mmlu_quiz_banks(
    output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
    mmlu_split: str = "test",
    mmlu_max_per_subject: int | None = 50,
) -> dict[str, int]:
    """Download ARC + MMLU data and write quiz-bank additions by difficulty."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    easy_items: list[dict] = []
    medium_items: list[dict] = []
    hard_items: list[dict] = []

    for path in _repo_parquet_paths(_ARC_DATASET):
        rows = _read_parquet_rows(_ARC_DATASET, path)
        difficulty = "medium" if "challenge" in path.lower() else "easy"
        source = "arc_challenge" if difficulty == "medium" else "arc_easy"
        target = medium_items if difficulty == "medium" else easy_items
        for example in rows:
            item = _arc_example_to_item(example, difficulty, source)
            if item is not None:
                target.append(item)

    subject_dirs = [path for path in _repo_parquet_paths(_MMLU_DATASET) if f"/{mmlu_split}" in path.lower()]
    if not subject_dirs:
        # Fallback: discover subject folders from the repo tree and scan their parquet files.
        root_entries = _hf_tree(_MMLU_DATASET)
        subject_dirs = sorted(
            str(entry.get("path", "")).rstrip("/")
            for entry in root_entries
            if isinstance(entry, dict) and entry.get("type") == "directory" and str(entry.get("path", "")).strip()
        )

    seen_subjects: set[str] = set()
    for path in subject_dirs:
        subject = Path(path).parts[0] if Path(path).parts else path
        if subject in seen_subjects:
            continue
        seen_subjects.add(subject)
        parquet_paths = [p for p in _repo_parquet_paths(_MMLU_DATASET, subject) if mmlu_split in p.lower()]
        for parquet_path in parquet_paths:
            rows = _read_parquet_rows(_MMLU_DATASET, parquet_path)
            for index, example in enumerate(rows):
                if mmlu_max_per_subject is not None and index >= mmlu_max_per_subject:
                    break
                item = _mmlu_example_to_item(example, subject)
                if item is not None:
                    hard_items.append(item)

    generated_files = {
        "easy": output_path / "easy.json",
        "medium": output_path / "medium.json",
        "hard": output_path / "hard.json",
    }
    _dump_json(generated_files["easy"], easy_items)
    _dump_json(generated_files["medium"], medium_items)
    _dump_json(generated_files["hard"], hard_items)

    manifest = {
        "easy": len(easy_items),
        "medium": len(medium_items),
        "hard": len(hard_items),
        "mmlu_split": mmlu_split,
        "arc_dataset": _ARC_DATASET,
        "mmlu_dataset": _MMLU_DATASET,
    }
    _dump_json(output_path / "manifest.json", [manifest])
    return {k: int(v) for k, v in manifest.items() if k in {"easy", "medium", "hard"}}


def build_merged_quiz_banks(output_dir: Path | str = _DEFAULT_OUTPUT_DIR) -> dict[str, int]:
    """Merge generated ARC/MMLU additions with the base quiz-bank files."""
    output_path = Path(output_dir)
    generated_dir = output_path
    base_dir = output_path.parent
    counts: dict[str, int] = {}

    for difficulty in ("easy", "medium", "hard"):
        merged = _merge_unique(
            _load_json(base_dir / f"{difficulty}.json"),
            _load_json(generated_dir / f"{difficulty}.json"),
        )
        _dump_json(generated_dir / f"{difficulty}.json", merged)
        counts[difficulty] = len(merged)
    return counts


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build ARC + MMLU quiz-bank additions for HPM v6")
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Directory to write generated quiz-bank additions into",
    )
    parser.add_argument(
        "--mmlu-split",
        default="test",
        help="MMLU split to import (default: test)",
    )
    parser.add_argument(
        "--mmlu-max-per-subject",
        type=int,
        default=50,
        help="Maximum MMLU questions to import per subject (default: 50; use 0 for all)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge generated additions with the base quiz-bank files",
    )
    args = parser.parse_args(argv)

    max_per_subject = None if args.mmlu_max_per_subject == 0 else args.mmlu_max_per_subject
    counts = build_arc_mmlu_quiz_banks(
        args.output_dir,
        mmlu_split=args.mmlu_split,
        mmlu_max_per_subject=max_per_subject,
    )
    if args.merge:
        counts = build_merged_quiz_banks(args.output_dir)

    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
