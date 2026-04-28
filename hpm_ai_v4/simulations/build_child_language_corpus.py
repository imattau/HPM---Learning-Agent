"""Build a small child-directed corpus for early-language benchmarks."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


DEFAULT_NOUNS = ["ball", "cup", "dog", "cat", "book", "toy"]
DEFAULT_VERBS = ["go", "run", "see", "give", "take", "look"]
DEFAULT_ADJECTIVES = ["big", "small", "red", "blue", "little", "new"]
DEFAULT_LOCATIONS = ["here", "there", "on the mat", "under the chair"]
DEFAULT_NAMES = ["mom", "dad", "baby", "child", "friend"]


@dataclass(frozen=True)
class ChildCorpusResult:
    corpus_path: str
    lines_written: int
    held_out_words: Dict[str, List[str]]


def _write_lines(path: str, lines: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def build_child_language_corpus(
    output_path: str,
    *,
    nouns: Optional[Sequence[str]] = None,
    verbs: Optional[Sequence[str]] = None,
    adjectives: Optional[Sequence[str]] = None,
    locations: Optional[Sequence[str]] = None,
    names: Optional[Sequence[str]] = None,
    held_out_nouns: Optional[Sequence[str]] = None,
    held_out_verbs: Optional[Sequence[str]] = None,
) -> ChildCorpusResult:
    """Build a compact child-directed corpus with simple templates."""
    nouns = list(nouns or DEFAULT_NOUNS)
    verbs = list(verbs or DEFAULT_VERBS)
    adjectives = list(adjectives or DEFAULT_ADJECTIVES)
    locations = list(locations or DEFAULT_LOCATIONS)
    names = list(names or DEFAULT_NAMES)
    held_out_nouns = list(held_out_nouns or [])
    held_out_verbs = list(held_out_verbs or [])

    train_nouns = [word for word in nouns if word not in held_out_nouns] or list(nouns)
    train_verbs = [word for word in verbs if word not in held_out_verbs] or list(verbs)
    train_adjectives = list(adjectives)

    lines: List[str] = []
    for noun in train_nouns:
        lines.extend(
            [
                f"This is a {noun}.",
                f"Where is the {noun}?",
                f"Give me the {noun}.",
                f"The {noun} is here.",
            ]
        )
        for adjective in train_adjectives[:2]:
            lines.append(f"The {noun} is {adjective}.")
    for verb in train_verbs:
        lines.extend(
            [
                f"The child can {verb}.",
                f"Can you {verb}?",
                f"Please {verb} now.",
            ]
        )
    for name in names:
        lines.extend(
            [
                f"{name} has a toy.",
                f"{name} sees the ball.",
                f"{name} likes the cup.",
            ]
        )
    for location in locations:
        lines.extend(
            [
                f"The ball is {location}.",
                f"The cup is {location}.",
            ]
        )
    lines.extend(
        [
            "What is this?",
            "Can you help me?",
            "This is nice.",
            "I want the ball.",
            "The dog runs.",
        ]
    )
    if "book" not in held_out_nouns:
        lines.append("Look at the book.")
    if "go" not in held_out_verbs:
        lines.append("We can go now.")
    _write_lines(output_path, lines)
    return ChildCorpusResult(
        corpus_path=output_path,
        lines_written=len(lines),
        held_out_words={
            "nouns": held_out_nouns,
            "verbs": held_out_verbs,
        },
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a child-directed corpus")
    p.add_argument("--output", required=True)
    p.add_argument("--held-out-noun", action="append", default=[])
    p.add_argument("--held-out-verb", action="append", default=[])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_child_language_corpus(
        args.output,
        held_out_nouns=args.held_out_noun,
        held_out_verbs=args.held_out_verb,
    )
    print(result)


if __name__ == "__main__":
    main()
