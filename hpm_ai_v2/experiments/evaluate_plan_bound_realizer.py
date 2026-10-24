"""Held-out comparison of the plan-bound and legacy extractive chat realizers.

Both answers use the same verified ResponsePlan. The script reports structural
grounding metrics and creates a randomized usefulness packet whose private key
is kept outside the shareable output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from hpm_ai_v2.chat.evidence import ExpandingPlanner
from hpm_ai_v2.chat.realizers import ExtractiveRealizer, PlanBoundRealizer
from hpm_ai_v2.chat.session import ChatSession
from hpm_ai_v2.experiments.run_coverage_eval import DEFAULT_SET, build_trainer
from hpm_ai_v2.training.coverage import load_coverage_set


def _coverage(rows):
    total = sum(row["plan_claim_count"] for row in rows)
    returned = sum(row["returned_claim_count"] for row in rows)
    sentences = sum(row["factual_sentence_count"] for row in rows)
    mapped = sum(row["mapped_sentence_count"] for row in rows)
    unsupported = sum(row["unsupported_sentence_count"] for row in rows)
    return {
        "cases": len(rows),
        "planned_claims": total,
        "returned_claims": returned,
        "claim_coverage": returned / max(total, 1),
        "factual_sentences": sentences,
        "citation_faithful_sentences": mapped,
        "citation_faithfulness": mapped / max(sentences, 1),
        "unsupported_sentences": unsupported,
        "unsupported_sentence_rate": unsupported / max(sentences, 1),
        "verified_answer_plans": sum(row["plan_verified"] for row in rows),
    }


def run(*, output_dir: Path, private_key: Path, seed: int = 6247,
        evaluation_set: Path = DEFAULT_SET) -> dict:
    sources, all_cases = load_coverage_set(evaluation_set)
    heldout = [case for case in all_cases if case.split == "heldout"]
    if not heldout:
        raise ValueError("evaluation set has no held-out questions")
    output_dir.mkdir(parents=True, exist_ok=True)
    if private_key.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("private key must be outside the shareable output directory")

    extractor = ExtractiveRealizer()
    realizer = PlanBoundRealizer()
    paired_rows = []
    packet_rows = []
    key_rows = []
    rating_rows = []
    with __import__("tempfile").TemporaryDirectory() as work:
        trainer = build_trainer(sources, work)
        session = ChatSession(trainer, planner=ExpandingPlanner(trainer))
        for case in heldout:
            reply = session.send(case.question)
            plan = reply.plan
            plan_verified = session.verifier.verify(plan)
            bound = realizer.realize_with_map(plan)
            bound_verified = plan_verified and session.verifier.verify_realization(plan, bound)
            extractive_text = extractor.realize(plan)
            if plan.act == "answer" and plan.claims:
                first = plan.claims[0]
                extractive_supported = (
                    plan_verified and first.text in extractive_text
                    and bool(first.citations)
                )
                extractive_claims = 1 if extractive_supported else 0
                extractive_sentences = 1
                extractive_mapped = int(extractive_supported)
                extractive_unsupported = int(not extractive_supported)
            else:
                extractive_claims = extractive_sentences = extractive_mapped = extractive_unsupported = 0
                extractive_supported = True

            bound_claims = len(bound.sentences) if bound_verified else 0
            bound_sentences = len(bound.sentences)
            bound_mapped = bound_claims
            bound_unsupported = bound_sentences - bound_mapped
            source_ids = sorted({citation.source_id for claim in plan.claims for citation in claim.citations})
            labels = {source_id: f"Source {chr(65 + index)}" for index, source_id in enumerate(source_ids)}
            bound_body, _, _ = bound.text.partition("\n\nSources:\n")
            if plan.act == "answer" and plan.claims:
                extractive_body = plan.claims[0].text + " [1]"
            else:
                extractive_body = extractive_text
            references = []
            for index, claim in enumerate(plan.claims, start=1):
                names = list(dict.fromkeys(labels[c.source_id] for c in claim.citations))
                references.append(f"[{index}] " + "; ".join(names))
            connected_source_ids = list(dict.fromkeys(
                citation.source_id for claim in plan.claims for citation in claim.citations
            ))
            baseline_source_ids = list(dict.fromkeys(
                citation.source_id for citation in (plan.claims[0].citations if plan.claims else ())
            ))
            connected_sources = [
                {"label": labels[source_id], "title": labels[source_id],
                 "text": trainer.sources[source_id].text}
                for source_id in connected_source_ids
            ]
            baseline_sources = [
                {"label": labels[source_id], "title": labels[source_id],
                 "text": trainer.sources[source_id].text}
                for source_id in baseline_source_ids
            ]
            connected_footer = "\n\nSources:\n" + "\n".join(references) if references else ""
            baseline_references = references[:1]
            baseline_footer = ("\n\nSources:\n" + "\n".join(baseline_references)
                               if baseline_references else "")
            connected_answer = {"text": bound_body + connected_footer, "sources": connected_sources}
            extractive_answer = {
                "text": extractive_body + baseline_footer,
                "sources": baseline_sources,
            }
            paired_rows.append({
                "question_id": case.question_id,
                "question": case.question,
                "plan_act": plan.act,
                "plan_verified": plan_verified,
                "plan_bound_verified": bound_verified,
                "claim_count": len(plan.claims),
                "plan_bound_text": bound.text,
                "extractive_text": extractive_text,
                "plan_bound_claims": [claim.text for claim in plan.claims],
            })
            pair = [("HPM plan-bound realizer", connected_answer),
                    ("Legacy extractive realizer", extractive_answer)]
            if random.Random(f"{seed}:{case.question_id}").random() < .5:
                pair.reverse()
            case_id = f"case-{len(packet_rows) + 1:04d}"
            packet_rows.append({
                "case_id": case_id,
                "task_type": "grounded_qa",
                "question": case.question,
                "response_a": pair[0][1],
                "response_b": pair[1][1],
            })
            key_rows.append({
                "case_id": case_id,
                "response_a_system": pair[0][0],
                "response_b_system": pair[1][0],
                "question_id": case.question_id,
            })
            rating_rows.append({"case_id": case_id, "rater_id": "",
                                "usefulness_a": "", "usefulness_b": "",
                                "preferred": "", "comment": ""})
            paired_rows[-1]["_metrics"] = {
                "plan_claim_count": len(plan.claims),
                "returned_claim_count": bound_claims,
                "factual_sentence_count": bound_sentences,
                "mapped_sentence_count": bound_mapped,
                "unsupported_sentence_count": bound_unsupported,
                "plan_verified": plan_verified,
            }
            paired_rows[-1]["_extractive_metrics"] = {
                "plan_claim_count": len(plan.claims),
                "returned_claim_count": extractive_claims,
                "factual_sentence_count": extractive_sentences,
                "mapped_sentence_count": extractive_mapped,
                "unsupported_sentence_count": extractive_unsupported,
                "plan_verified": plan_verified,
            }

    bound_metrics = _coverage([row["_metrics"] for row in paired_rows])
    extractive_metrics = _coverage([row["_extractive_metrics"] for row in paired_rows])
    for row in paired_rows:
        del row["_metrics"]
        del row["_extractive_metrics"]
    (output_dir / "comparison.json").write_text(json.dumps({
        "evaluation_set": str(evaluation_set),
        "heldout_question_count": len(heldout),
        "same_response_plan_for_both_realizers": True,
        "plan_bound_realizer": bound_metrics,
        "legacy_extractive_realizer": extractive_metrics,
        "blinded_usefulness": "pending independent ratings",
        "paired_cases": paired_rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    packet_path = output_dir / "rating_packet.jsonl"
    packet_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet_rows),
                           encoding="utf-8")
    (output_dir / "ratings_template.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rating_rows), encoding="utf-8"
    )
    with (output_dir / "ratings_template.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rating_rows[0]))
        writer.writeheader()
        writer.writerows(rating_rows)
    (output_dir / "rating_instructions.md").write_text(
        "# Blinded realizer comparison\n\n"
        "For each question, score response A and B independently from 1 (not useful) to 5 "
        "(extremely useful), then choose A, B, or tie. The cited source material is shown with "
        "anonymized labels. Do not infer system identity or inspect the private key before ratings "
        "are complete. Return the completed `ratings_template.csv` for analysis with "
        "`hpm_ai_v2.experiments.blinded_usefulness analyze`.\n",
        encoding="utf-8",
    )
    private_key.parent.mkdir(parents=True, exist_ok=True)
    private_key.write_text(json.dumps({"seed": seed,
        "hpm_model": "HPM plan-bound realizer",
        "baseline_model": "Legacy extractive realizer",
        "case_mapping": key_rows}, indent=2) + "\n", encoding="utf-8")
    private_key.chmod(0o600)
    manifest = {"case_count": len(packet_rows), "seed": seed,
                "rating_scale": "1=not useful, 2=slightly useful, 3=moderately useful, 4=very useful, 5=extremely useful",
                "private_key_path": str(private_key),
                "task_type_counts": {"grounded_qa": len(packet_rows)}}
    (output_dir / "packet_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                       encoding="utf-8")
    return {"plan_bound": bound_metrics, "extractive": extractive_metrics,
            "case_count": len(packet_rows), "packet": str(packet_path),
            "private_key": str(private_key)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--output-dir", type=Path, default=Path(
        "hpm_ai_v2/experiments/results/2026-10-25-plan-bound-realizer-heldout-v1"))
    parser.add_argument("--private-key", type=Path, default=Path(
        "/tmp/hpm-planbound-realizer-heldout-key.json"))
    parser.add_argument("--seed", type=int, default=6247)
    args = parser.parse_args()
    print(json.dumps(run(output_dir=args.output_dir, private_key=args.private_key,
                         seed=args.seed, evaluation_set=args.set), indent=2))


if __name__ == "__main__":
    main()
