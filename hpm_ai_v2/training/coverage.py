"""Support-set coverage: does a response contain what a complete answer needs?

Each answerable case names the source sentences a complete answer needs (its
*support set*). A system's response is the set of source sentences it cites.
Per case:

* recall    = |returned ∩ support| / |support|        (did it find what was needed)
* precision = |returned ∩ support| / |returned|        (how much of it was needed)
* complete  = recall == 1

Unanswerable cases only record whether the system abstained. Results are
reported per question type with counts: single-fact, process, compound and
cross-source questions behave differently and an aggregate would hide that.
Labels are one annotator's judgement of a minimal sufficient support set.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from hpm_ai_v2.training.study import SourceDocument, _wilson_interval

SentenceKey = tuple[str, int]          # (source_id, sentence ordinal within the source)
QUESTION_TYPES = ("specific", "process", "compound", "cross", "unanswerable")


@dataclass(frozen=True)
class CoverageCase:
    question_id: str
    split: str
    type: str
    question: str
    support: frozenset[SentenceKey]

    @property
    def answerable(self) -> bool:
        return bool(self.support)


def load_coverage_set(path: str | Path) -> tuple[list[SourceDocument], list[CoverageCase]]:
    sources, cases = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["kind"] == "source":
            sources.append(SourceDocument(row["source_id"], row["text"], row.get("title", "")))
        elif row["kind"] == "coverage":
            cases.append(CoverageCase(
                row["question_id"], row["split"], row["type"], row["question"],
                frozenset((s, int(i)) for s, i in row["support"]),
            ))
        else:
            raise ValueError(f"unknown row kind {row['kind']!r}")
    return sources, cases


def sentence_keys(trainer, spans: Iterable[tuple[str, int, int]]) -> set[SentenceKey]:
    """Map cited ``(source_id, start, end)`` spans to ``(source_id, ordinal)``."""
    keys: set[SentenceKey] = set()
    for source_id, start, end in spans:
        for ordinal, node_id in enumerate(trainer.source_to_answer_nodes.get(source_id, ())):
            data = trainer.answer_node_data[node_id]
            if data["start"] == start and data["end"] == end:
                keys.add((source_id, ordinal))
    return keys


def score_case(case: CoverageCase, returned: set[SentenceKey], abstained: bool) -> dict:
    row = {"question_id": case.question_id, "type": case.type, "split": case.split,
           "abstained": abstained, "width": len(returned)}
    if not case.answerable:
        return row
    hit = len(returned & case.support)
    row.update(
        recall=hit / len(case.support),
        precision=hit / len(returned) if returned else 0.0,
        complete=hit == len(case.support),
        support=len(case.support),
    )
    return row


def aggregate(rows: list[dict]) -> dict:
    """Per-type and overall summaries with counts (no aggregate-only score)."""
    out: dict[str, dict] = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["type"]].append(row)
    answerable_rows = [r for r in rows if "recall" in r]
    for name, members in [*sorted(groups.items()), ("all_answerable", answerable_rows)]:
        if name == "unanswerable":
            out[name] = {"n": len(members), "abstained": sum(r["abstained"] for r in members)}
            continue
        scored = [r for r in members if "recall" in r]
        if not scored:
            continue
        n = len(scored)
        complete = sum(r["complete"] for r in scored)
        recall = sum(r["recall"] for r in scored) / n
        precision = sum(r["precision"] for r in scored) / n
        out[name] = {
            "n": n,
            "recall": recall,
            "precision": precision,
            "f1": (2 * recall * precision / (recall + precision)) if recall + precision else 0.0,
            "complete": complete,
            "complete_rate": complete / n,
            "complete_wilson95": _wilson_interval(complete, n),
            "mean_width": sum(r["width"] for r in scored) / n,
            "wrongly_abstained": sum(r["abstained"] for r in scored),
        }
    return out


def source_stem_coverage(trainer, source_id: str, returned: set[SentenceKey]) -> float:
    """Fraction of a source's content stems (idf-weighted) present in the returned sentences."""
    from hpm_ai_v2.agents.text_structure import content_stems

    sentences = [trainer.answer_node_data[n]["text"] for n in trainer.source_to_answer_nodes[source_id]]
    stems = [set(content_stems(s)) for s in sentences]
    df: dict[str, int] = defaultdict(int)
    for group in stems:
        for stem in group:
            df[stem] += 1
    weight = {s: math.log(1 + len(stems) / df[s]) for s in df}
    total = sum(weight.values())
    covered = {s for (sid, i) in returned if sid == source_id for s in stems[i]}
    return sum(weight[s] for s in covered) / total if total else 0.0


System = Callable[[object, str], tuple[set[SentenceKey], bool]]


def run_system(trainer, cases: list[CoverageCase], system: System) -> list[dict]:
    rows = []
    for case in cases:
        returned, abstained = system(trainer, case.question)
        rows.append(score_case(case, returned, abstained))
    return rows
