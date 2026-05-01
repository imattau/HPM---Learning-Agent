"""Build a synthetic relational corpus for binding/emergence probes."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


DEFAULT_SUBJECTS = ["cat", "dog", "robot", "scientist", "child", "teacher"]
DEFAULT_OBJECTS = ["mat", "book", "cup", "signal", "toy", "trophy"]
DEFAULT_VERBS = ["chased", "noticed", "moved toward", "approached", "admired", "watched"]
DEFAULT_PROPERTIES = ["red", "blue", "round", "small", "bright", "wet"]
DEFAULT_RELATION_MARKERS = ["that", "which", "who"]


@dataclass(frozen=True)
class RelationalCorpusResult:
    corpus_path: str
    lines_written: int
    family_counts: Dict[str, int]


def _write_lines(path: str, lines: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.rstrip() + "\n")


def _cycle(words: Sequence[str], index: int) -> str:
    if not words:
        return ""
    return words[index % len(words)]


def build_relational_corpus(
    output_path: str,
    *,
    sentence_count: int = 50000,
    subjects: Optional[Sequence[str]] = None,
    objects: Optional[Sequence[str]] = None,
    verbs: Optional[Sequence[str]] = None,
    properties: Optional[Sequence[str]] = None,
    markers: Optional[Sequence[str]] = None,
) -> RelationalCorpusResult:
    """Build a large corpus of active, passive, and nested relational sentences."""
    subjects = list(subjects or DEFAULT_SUBJECTS)
    objects = list(objects or DEFAULT_OBJECTS)
    verbs = list(verbs or DEFAULT_VERBS)
    properties = list(properties or DEFAULT_PROPERTIES)
    markers = list(markers or DEFAULT_RELATION_MARKERS)
    sentence_count = max(1, int(sentence_count))

    lines: List[str] = []
    family_counts: Dict[str, int] = {"active": 0, "passive": 0, "nested_active": 0, "nested_passive": 0, "property": 0}
    families = ["active", "passive", "nested_active", "nested_passive", "property"]

    for idx in range(sentence_count):
        family = families[idx % len(families)]
        subject = _cycle(subjects, idx)
        obj = _cycle(objects, idx + 1)
        other_subject = _cycle(subjects, idx + 2)
        verb = _cycle(verbs, idx)
        prop = _cycle(properties, idx)
        marker = _cycle(markers, idx)

        if family == "active":
            lines.append(f"The {subject} {verb} the {obj}.")
        elif family == "passive":
            lines.append(f"The {obj} was {verb} by the {subject}.")
        elif family == "nested_active":
            lines.append(f"The {subject} {marker} the {other_subject} {verb} sat on the {obj}.")
        elif family == "nested_passive":
            lines.append(f"The {obj} was sat on by the {subject} {marker} the {other_subject} {verb}.")
        else:
            lines.append(f"The {subject} was {prop}.")
        family_counts[family] += 1

    _write_lines(output_path, lines)
    return RelationalCorpusResult(
        corpus_path=output_path,
        lines_written=len(lines),
        family_counts=family_counts,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a synthetic relational corpus")
    p.add_argument("--output", required=True)
    p.add_argument("--sentence-count", type=int, default=50000)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_relational_corpus(args.output, sentence_count=args.sentence_count)
    print(result)


if __name__ == "__main__":
    main()
