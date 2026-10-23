"""Calibrate a pretrained checkpoint's abstention from labelled validation questions.

    PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.chat.calibrate CHECKPOINT \
        --validation questions.jsonl --output calibration.json [--test held_out.jsonl]

Each row of a question file is ``{"id": ..., "question": ..., "answer_contains": "quoted text" | null}``.
``answer_contains`` is a snippet of the source sentence that answers the question; every source containing it
is a right source. ``null`` marks a question the corpus cannot answer. Resolving by quoted text keeps the labels
valid when a corpus is re-chunked.

The threshold itself is the study trainer's own (``SupervisedStudyTrainer.fit`` with validation episodes). The
checkpoint is not modified: the result is a small JSON file bound to the checkpoint's fingerprint, and
``python -m hpm_ai_v2.chat.server CHECKPOINT --calibration calibration.json`` applies it when loading.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
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
    acceptable_sources: frozenset          # empty: the corpus cannot answer it

    @property
    def answerable(self) -> bool:
        return bool(self.acceptable_sources)


def read_questions(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate question ids in {path}")
    return rows


def resolve(trainer, rows: list[dict]) -> list[Item]:
    items = []
    for row in rows:
        snippet = row.get("answer_contains")
        if snippet is None:
            items.append(Item(row["id"], row["question"], frozenset()))
            continue
        sources = frozenset(sid for sid, source in trainer.sources.items() if snippet in source.text)
        if not sources:
            raise ValueError(f"{row['id']}: no source contains {snippet!r}")
        items.append(Item(row["id"], row["question"], sources))
    return items


def cited_sources(plan) -> frozenset:
    if plan.act == "clarify":
        return frozenset(sid for sid, _label in plan.options)
    return frozenset(c.source_id for claim in plan.claims for c in claim.citations)


def evaluate(items: list[Item], planner) -> dict:
    """How a planner behaves on labelled questions: counts, and a balanced score for comparison only."""
    outcome: Counter = Counter()
    for item in items:
        plan = planner.plan("evaluation", item.question)
        if item.answerable:
            if plan.act == "abstain":
                outcome["answerable: abstained"] += 1
            elif cited_sources(plan) & item.acceptable_sources:
                outcome["answerable: right source"] += 1
            else:
                outcome["answerable: wrong source"] += 1
        else:
            outcome["unanswerable: abstained" if plan.act == "abstain" else "unanswerable: answered or clarified"] += 1
    answerable = sum(item.answerable for item in items)
    unanswerable = len(items) - answerable
    rates = []
    if answerable:
        rates.append(outcome["answerable: right source"] / answerable)
    if unanswerable:
        rates.append(outcome["unanswerable: abstained"] / unanswerable)
    return {"counts": dict(sorted(outcome.items())), "answerable": answerable, "unanswerable": unanswerable,
            "balanced_accuracy": sum(rates) / len(rates) if rates else 0.0}


def fit_calibration(trainer, items: list[Item], *, checkpoint_digest: Optional[str] = None) -> Calibration:
    """Calibrate ``trainer`` in memory with the trainer's own procedure and return the result.

    The trainer is left calibrated, so pass a freshly loaded one. When a question has several right sources the
    one the trainer ranks highest is used as its evidence id, so ties between equivalent sources do not count
    as retrieval errors.
    """
    if not any(i.answerable for i in items) or not any(not i.answerable for i in items):
        raise ValueError("calibration needs both answerable and unanswerable validation questions")
    before = evaluate(items, ExpandingPlanner(trainer))
    episodes = []
    for item in items:
        if item.answerable:
            ranked = [sid for sid, _score in trainer.predict_question(item.question).ranked_evidence]
            evidence = next((sid for sid in ranked if sid in item.acceptable_sources), min(item.acceptable_sources))
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
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, help="held-out questions: evaluated before and after, never fitted")
    parser.add_argument("--output", type=Path, required=True, help="where to write the calibration JSON")
    args = parser.parse_args(argv)
    trainer = load_trainer(args.checkpoint)
    digest = file_digest(args.checkpoint / "study_training_state.json")
    validation = resolve(trainer, read_questions(args.validation))
    report: dict = {}
    held_out = resolve(trainer, read_questions(args.test)) if args.test else None
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
