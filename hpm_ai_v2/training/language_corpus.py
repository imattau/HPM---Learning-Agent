"""Corpus reading, source-grouped splitting, and document-local counts."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from hpm_ai_v2.training.language_constants import BOUNDARY, TOKEN_RE, UNK

def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
def split_sentences(text: str) -> list[str]:
    """Split text into punctuation/newline-delimited units for local context."""
    return [part for part in re.split(r"(?<=[.!?])\s+|\n+", text) if tokenize(part)]
def _count_raw_text_batch(texts: list[str], max_order: int):
    """Count independent document-local lexical and character statistics."""
    tokenized = []
    word_counts = Counter()
    char_counts = defaultdict(Counter)
    for text in texts:
        sentence_tokens = [tokenize(sentence) for sentence in split_sentences(text)]
        tokenized.append(sentence_tokens)
        for tokens in sentence_tokens:
            word_counts.update(tokens)
        for sentence in split_sentences(text):
            chars = list(sentence.lower()) + [BOUNDARY]
            padded = [BOUNDARY] * (max_order - 1) + chars
            for i in range(max_order - 1, len(padded)):
                target = padded[i]
                for order in range(max_order):
                    char_counts[tuple(padded[i - order:i]) if order else ()][target] += 1
    return tokenized, word_counts, dict(char_counts)
def _batches(items: list[str], size: int = 16):
    for start in range(0, len(items), size):
        yield items[start:start + size]
def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    source_label = Path(path).name
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                text = row.get("text")
                if isinstance(text, str) and text.strip():
                    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                    source_id = str(row.get("source_id", row.get("id", len(rows))))
                    group = metadata.get("group_id") or metadata.get("conversation_id") or metadata.get("message_tree_id")
                    if not group and all(metadata.get(key) for key in (
                        "huggingface_dataset_id", "huggingface_revision", "huggingface_split",
                        "huggingface_row_id",
                    )):
                        group = "|".join(str(metadata[key]) for key in (
                            "huggingface_dataset_id", "huggingface_revision", "huggingface_split",
                            "huggingface_row_id",
                        ))
                    if not group and metadata.get("source_path"):
                        group = f"{metadata['source_path']}:{metadata.get('document_index', 0)}"
                    group = group or metadata.get("parent_id") or source_id
                    source_path = str(metadata.get("source_path", ""))
                    source_type = metadata.get("source_type", "unknown")
                    if Path(source_path).suffix.lower() in {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".h", ".cpp", ".cs", ".rb", ".php", ".sh"}:
                        source_type = "code"
                    rows.append({"id": source_id, "group_id": f"{source_label}:{group}", "text": text,
                                 "source_file": source_label, "source_type": source_type,
                                 "source_path": source_path, "license": metadata.get("license", "unspecified"),
                                 "metadata": metadata})
    return rows
def split_documents(rows: list[dict], seed: int = 17) -> dict[str, list[dict]]:
    """Stable source-level split; small mixed-source corpora are stratified.

    With at most twelve independent sources, a plain 80/10/10 hash can leave
    a whole source type out of validation. When every source type has at least
    three groups, allocate each type approximately 60/20/20 by group so train,
    validation and test all cover the same source types. Larger corpora retain
    the usual 80/10/10 split. No rows from one source group are separated.
    """
    def source_group(row: dict) -> str:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        source_path = row.get("source_path") or metadata.get("source_path")
        source_url = row.get("source_url") or metadata.get("source_url")
        if source_path:
            return f"source_path:{source_path}"
        if source_url:
            return f"source_url:{source_url}"
        return str(row.get("group_id", row["id"]))

    group_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        group_rows[source_group(row)].append(row)
    group_types = {
        group: {str(row.get("source_type", "unknown")) for row in group_members}
        for group, group_members in group_rows.items()
    }
    source_groups: dict[str, list[str]] = defaultdict(list)
    for group, types in group_types.items():
        if len(types) != 1:
            source_groups.clear()
            break
        source_groups[next(iter(types))].append(group)
    if (1 < len(source_groups) and len(group_rows) <= 12
            and all(len(groups) >= 3 for groups in source_groups.values())):
        buckets: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
        for source_type, groups in sorted(source_groups.items()):
            ordered = sorted(groups, key=lambda group: hashlib.sha256(
                f"{seed}:{source_type}:{group}".encode()).digest())
            train_count = max(1, int(0.6 * len(ordered)))
            validation_count = max(1, int(0.2 * len(ordered)))
            assignments = {
                "train": ordered[:train_count],
                "validation": ordered[train_count:train_count + validation_count],
                "test": ordered[train_count + validation_count:],
            }
            for name, assigned_groups in assignments.items():
                buckets[name].extend(row for group in assigned_groups for row in group_rows[group])
        return buckets

    buckets: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        digest = hashlib.sha256(f"{seed}:{source_group(row)}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        name = "train" if value < .8 else "validation" if value < .9 else "test"
        buckets[name].append(row)
    # For tiny corpora, deterministic sorted fallback ensures all splits exist.
    if any(not buckets[name] for name in buckets):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[source_group(row)].append(row)
        ordered_groups = sorted(groups.items(), key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).digest())
        n = len(ordered_groups)
        a, b = max(1, int(n * .8)), max(2, int(n * .9))
        buckets = {"train": [], "validation": [], "test": []}
        for name, group_slice in (("train", ordered_groups[:a]), ("validation", ordered_groups[a:b]), ("test", ordered_groups[b:])):
            buckets[name] = [row for _, group_rows in group_slice for row in group_rows]
    return buckets
