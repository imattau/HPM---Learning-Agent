"""Support-set coverage evaluation of chat evidence selection.

    PYTHONPATH=. .venv/bin/python -m hpm_ai_v2.experiments.run_coverage_eval

Reports the dev split by default. The held-out split is scored only with
``--include-heldout``, to keep it out of tuning.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.chat import ChatSession
from hpm_ai_v2.domains.text_domain import TextDomainConfig, tokenise
from hpm_ai_v2.training import SupervisedStudyTrainer
from hpm_ai_v2.training.coverage import (
    aggregate, load_coverage_set, run_system, sentence_keys, source_stem_coverage,
)

DEFAULT_SET = Path(__file__).parents[1] / "training/eval_sets/coverage_v1.jsonl"


def build_trainer(sources, workdir: str):
    # No truncation: a vocabulary smaller than the sources' own would blind every system alike.
    distinct = len({w for s in sources for w in tokenise(s.text)})
    config = TextDomainConfig.from_passages([s.text for s in sources], max_vocab=max(300, distinct))
    agent = ReaderAgent(config, dynamic_vocab=False, cold_dir=workdir)
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest(sources)
    return trainer


def chat_system(planner_factory=None):
    """The chat as a system: cited sentences of the reply."""
    def system(trainer, question):
        session = ChatSession(trainer) if planner_factory is None else ChatSession(trainer, planner=planner_factory(trainer))
        reply = session.send(question)
        spans = [(c.source_id, c.start, c.end) for c in reply.citations]
        return sentence_keys(trainer, spans), reply.plan.act != "answer"
    return system


def naive_top_k(k: int):
    """Baseline: the k best-scoring sentences of the selected source, no structure."""
    def system(trainer, question):
        prediction = trainer.predict_question(question)
        if prediction.selected_evidence_id is None:
            return set(), True
        source_id = prediction.selected_evidence_id
        ranked = [n for n, _ in trainer._rank_answers(question, source_id)][:k]
        spans = [(source_id, trainer.answer_node_data[n]["start"], trainer.answer_node_data[n]["end"]) for n in ranked]
        return sentence_keys(trainer, spans), False
    return system


def summary_coverage(trainer, system):
    rows = []
    for source_id, source in trainer.sources.items():
        returned, _ = system(trainer, f"Summarize {source.title}")
        rows.append({"source_id": source_id, "width": len(returned),
                     "stem_coverage": source_stem_coverage(trainer, source_id, returned)})
    n = len(rows)
    return {"n": n, "mean_width": sum(r["width"] for r in rows) / n,
            "mean_stem_coverage": sum(r["stem_coverage"] for r in rows) / n, "per_source": rows}


def systems():
    table = {"current_chat": chat_system(), **{f"naive_top{k}": naive_top_k(k) for k in (2, 3, 4)}}
    try:
        from hpm_ai_v2.chat.evidence import ExpandingPlanner
        table["expanded"] = chat_system(ExpandingPlanner)
    except ImportError:
        pass
    return table


def format_table(name, summary) -> str:
    lines = [f"== {name}"]
    for kind, s in summary.items():
        if kind == "unanswerable":
            lines.append(f"  {kind:<14} n={s['n']:<3} abstained {s['abstained']}/{s['n']}")
        else:
            lines.append(
                f"  {kind:<14} n={s['n']:<3} recall {s['recall']:.2f}  precision {s['precision']:.2f}  "
                f"complete {s['complete']}/{s['n']}  width {s['mean_width']:.1f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--include-heldout", action="store_true")
    parser.add_argument("--only", nargs="*", help="system names to run")
    parser.add_argument("--json", type=Path, help="write full results here")
    args = parser.parse_args(argv)
    sources, cases = load_coverage_set(args.set)
    splits = ["dev"] + (["heldout"] if args.include_heldout else [])
    report = {}
    with tempfile.TemporaryDirectory() as work:
        trainer = build_trainer(sources, work)
        for name, system in systems().items():
            if args.only and name not in args.only:
                continue
            report[name] = {}
            for split in splits:
                rows = run_system(trainer, [c for c in cases if c.split == split], system)
                report[name][split] = {"summary": aggregate(rows), "rows": rows}
                print(format_table(f"{name} [{split}]", report[name][split]["summary"]))
            sc = summary_coverage(trainer, system)
            report[name]["summary_coverage"] = sc
            print(f"  summaries      width {sc['mean_width']:.1f}  source stem coverage {sc['mean_stem_coverage']:.2f}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
