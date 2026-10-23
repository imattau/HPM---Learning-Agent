"""Calibrate a pretrained checkpoint's abstention from labelled validation requests.

    PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.chat.calibrate CHECKPOINT \
        --validation questions.jsonl [--validation requests.jsonl ...] --output calibration.json \
        [--test held_out.jsonl ...]

A row is ``{"id": ..., "question": ...}`` plus one label:

* ``"answer_contains": "quoted text"``: a snippet of the sentence that answers it; every source containing it is
  a right source. Quoting keeps labels valid when a corpus is re-chunked.
* ``"title_contains": "topic"`` (or a list of topics): the right sources are those whose title contains the text.
  A list means the request needs every topic, as a comparison does.
* neither, or ``null``: the corpus cannot answer it.

The threshold is the study trainer's own (``SupervisedStudyTrainer.fit`` with validation episodes), one for all
request types: a threshold per request type (summary, outline, flashcards, comparison) was measured on held-out
requests and was not better than this one (see the chat design doc). The checkpoint is not modified: the result
is a small JSON file bound to the checkpoint's fingerprint, applied by
``python -m hpm_ai_v2.chat.server CHECKPOINT --calibration calibration.json``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hpm_ai_v2.chat.calibration import Calibration
from hpm_ai_v2.chat.evidence import ExpandingPlanner
from hpm_ai_v2.chat.feedback import file_digest
from hpm_ai_v2.training import StudyEpisode

@dataclass(frozen=True)
class Item:
    question_id: str
    question: str
    groups: tuple                          # right sources, one set per thing the request needs; empty: unanswerable

    @property
    def answerable(self) -> bool:
        return bool(self.groups)

    @property
    def acceptable_sources(self) -> frozenset:
        return frozenset().union(*self.groups) if self.groups else frozenset()

    def satisfied_by(self, cited: frozenset) -> bool:
        return self.answerable and all(cited & group for group in self.groups)


def read_questions(*paths: str | Path) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows += [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    return rows


def resolve(trainer, rows: list[dict]) -> list[Item]:
    items = []
    for row in rows:
        snippet, topics = row.get("answer_contains"), row.get("title_contains")
        if snippet is None and topics is None:
            items.append(Item(row["id"], row["question"], ()))
            continue
        if topics is not None:
            wanted = [topics] if isinstance(topics, str) else list(topics)
            groups = tuple(
                frozenset(sid for sid, source in trainer.sources.items() if topic in (source.title or ""))
                for topic in wanted
            )
            missing = [topic for topic, group in zip(wanted, groups) if not group]
            if missing:
                raise ValueError(f"{row['id']}: no source title contains {missing[0]!r}")
        else:
            group = frozenset(sid for sid, source in trainer.sources.items() if snippet in source.text)
            if not group:
                raise ValueError(f"{row['id']}: no source contains {snippet!r}")
            groups = (group,)
        items.append(Item(row["id"], row["question"], groups))
    return items


def cited_sources(plan) -> frozenset:
    if plan.act == "clarify":
        return frozenset(sid for sid, _label in plan.options)
    return frozenset(c.source_id for claim in plan.claims for c in claim.citations)


def _rates(right: int, answerable: int, abstained: int, unanswerable: int) -> float:
    rates = []
    if answerable:
        rates.append(right / answerable)
    if unanswerable:
        rates.append(abstained / unanswerable)
    return sum(rates) / len(rates) if rates else 0.0


def evaluate(items: list[Item], planner) -> dict:
    """How a planner behaves on labelled requests: counts overall and by request type, never one score."""
    overall: Counter = Counter()
    by_family: dict[str, Counter] = defaultdict(Counter)
    for item in items:
        plan = planner.plan("evaluation", item.question)
        family = plan.task_family
        if item.answerable:
            if plan.act == "abstain":
                key = "answerable: abstained"
            elif item.satisfied_by(cited_sources(plan)):
                key = "answerable: right source"
            else:
                key = "answerable: wrong source"
        else:
            key = "unanswerable: abstained" if plan.act == "abstain" else "unanswerable: answered or clarified"
        overall[key] += 1
        by_family[family][key] += 1
    def summary(counts: Counter) -> dict:
        answerable = sum(v for k, v in counts.items() if k.startswith("answerable"))
        unanswerable = sum(v for k, v in counts.items() if k.startswith("unanswerable"))
        return {"counts": dict(sorted(counts.items())), "answerable": answerable, "unanswerable": unanswerable,
                "balanced_accuracy": _rates(counts["answerable: right source"], answerable,
                                            counts["unanswerable: abstained"], unanswerable)}
    result = summary(overall)
    result["by_family"] = {family: summary(counts) for family, counts in sorted(by_family.items())}
    return result


def fit_calibration(trainer, items: list[Item], *, checkpoint_digest: Optional[str] = None) -> Calibration:
    """Calibrate ``trainer`` in memory with the trainer's own procedure and return the result.

    The trainer is left calibrated, so pass a freshly loaded one. When a question has several right sources the
    one the trainer ranks highest is used as its evidence id, so ties between equivalent sources do not count
    as retrieval errors; a request that needs several sources uses the first group.
    """
    if not any(i.answerable for i in items) or not any(not i.answerable for i in items):
        raise ValueError("calibration needs both answerable and unanswerable validation questions")
    before = evaluate(items, ExpandingPlanner(trainer))
    episodes = []
    for item in items:
        if item.answerable:
            ranked = [sid for sid, _score in trainer.predict_question(item.question).ranked_evidence]
            first = item.groups[0]
            evidence = next((sid for sid in ranked if sid in first), min(first))
            episodes.append(StudyEpisode(f"cal-{item.question_id}", "validation", "grounded_qa", item.question, evidence))
        else:
            episodes.append(StudyEpisode(f"cal-{item.question_id}", "validation", "abstain", item.question, None))
    trainer.fit([], [], episodes)
    after = evaluate(items, ExpandingPlanner(trainer))
    return Calibration(
        abstain_threshold=trainer.abstain_threshold, checkpoint_digest=checkpoint_digest,
        fitted_on={"answerable": before["answerable"], "unanswerable": before["unanswerable"],
                   "balanced_accuracy_before": before["balanced_accuracy"],
                   "balanced_accuracy_after": after["balanced_accuracy"]},
    )


def load_trainer(checkpoint: Path):
    from hpm_ai_v2.agents.reader_agent import ReaderAgent
    from hpm_ai_v2.training import SupervisedStudyTrainer

    trainer = SupervisedStudyTrainer(ReaderAgent.load_agent(str(checkpoint)))
    trainer.load_checkpoint_state(checkpoint)
    return trainer


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--validation", type=Path, action="append", required=True)
    parser.add_argument("--test", type=Path, action="append", help="held-out questions: evaluated before and after, never fitted")
    parser.add_argument("--output", type=Path, required=True, help="where to write the calibration JSON")
    args = parser.parse_args(argv)
    trainer = load_trainer(args.checkpoint)
    digest = file_digest(args.checkpoint / "study_training_state.json")
    validation = resolve(trainer, read_questions(*args.validation))
    report: dict = {}
    held_out = resolve(trainer, read_questions(*args.test)) if args.test else None
    if held_out is not None:
        report["test_before"] = evaluate(held_out, ExpandingPlanner(trainer))
    calibration = fit_calibration(trainer, validation, checkpoint_digest=digest)
    calibration.save(args.output)
    report["calibration"] = calibration.to_dict()
    report["written"] = str(args.output)
    if held_out is not None:
        report["test_after"] = evaluate(held_out, ExpandingPlanner(trainer))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
