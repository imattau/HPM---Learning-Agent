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
        evaluation_set: Path = DEFAULT_SET,
        ranker_checkpoint: Path | None = None) -> dict:
    sources, all_cases = load_coverage_set(evaluation_set)
    heldout = [case for case in all_cases if case.split == "heldout"]
    if not heldout:
        raise ValueError("evaluation set has no held-out questions")
    output_dir.mkdir(parents=True, exist_ok=True)
    if private_key.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("private key must be outside the shareable output directory")

    extractor = ExtractiveRealizer()
    baseline_realizer = PlanBoundRealizer()
    language_model = None
    if ranker_checkpoint is not None:
        from hpm_ai_v2.training.language_prediction import HierarchicalNextUnitModel

        language_model = HierarchicalNextUnitModel.load(ranker_checkpoint)
        realizer = PlanBoundRealizer(
            phrase_scorer=lambda candidate: language_model.score_surface_cross_entropy([candidate])
        )
        baseline_name = "rule-based plan-bound realizer"
    else:
        realizer = baseline_realizer
        baseline_name = "legacy extractive realizer"
    paired_rows = []
    packet_rows = []
    key_rows = []
    rating_rows = []
    with __import__("tempfile").TemporaryDirectory() as work:
        trainer = build_trainer(sources, work)
        session = ChatSession(trainer, planner=ExpandingPlanner(trainer), realizer=realizer)
        for case in heldout:
            reply = session.send(case.question)
            plan = reply.plan
            plan_verified = session.verifier.verify(plan)
            bound = realizer.realize_with_map(plan)
            if reply.realizer != realizer.name or reply.text != bound.text:
                raise RuntimeError("ChatSession did not return the configured realizer output")
            bound_verified = plan_verified and session.verifier.verify_realization(plan, bound)
            rule_based = baseline_realizer.realize_with_map(plan) if language_model is not None else None
            rule_based_verified = (
                plan_verified and session.verifier.verify_realization(plan, rule_based)
                if rule_based is not None else False
            )
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
            if rule_based is not None:
                baseline_text = rule_based.text
                baseline_claims = len(rule_based.sentences) if rule_based_verified else 0
                baseline_sentence_count = len(rule_based.sentences)
                baseline_mapped = baseline_claims
                baseline_unsupported = baseline_sentence_count - baseline_mapped
            else:
                baseline_text = extractive_text
                baseline_claims = extractive_claims
                baseline_sentence_count = extractive_sentences
                baseline_mapped = extractive_mapped
                baseline_unsupported = extractive_unsupported
            source_ids = sorted({citation.source_id for claim in plan.claims for citation in claim.citations})
            labels = {source_id: f"Source {chr(65 + index)}" for index, source_id in enumerate(source_ids)}
            bound_body, _, _ = bound.text.partition("\n\nSources:\n")
            references = []
            for index, claim in enumerate(plan.claims, start=1):
                names = list(dict.fromkeys(labels[c.source_id] for c in claim.citations))
                references.append(f"[{index}] " + "; ".join(names))
            connected_source_ids = list(dict.fromkeys(
                citation.source_id for claim in plan.claims for citation in claim.citations
            ))
            if rule_based is not None:
                baseline_body, _, _ = rule_based.text.partition("\n\nSources:\n")
                baseline_source_ids = list(connected_source_ids)
                baseline_references = references
            else:
                if plan.act == "answer" and plan.claims:
                    baseline_body = plan.claims[0].text + " [1]"
                    baseline_source_ids = list(dict.fromkeys(
                        citation.source_id for citation in plan.claims[0].citations
                    ))
                    baseline_references = references[:1]
                else:
                    baseline_body = extractive_text
                    baseline_source_ids = []
                    baseline_references = []
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
            baseline_footer = ("\n\nSources:\n" + "\n".join(baseline_references)
                               if baseline_references else "")
            connected_answer = {"text": bound_body + connected_footer, "sources": connected_sources}
            baseline_answer = {
                "text": baseline_body + baseline_footer,
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
                "rule_based_plan_bound_text": rule_based.text if rule_based is not None else None,
                "extractive_text": extractive_text,
                "plan_bound_claims": [claim.text for claim in plan.claims],
            })
            pair = [("HPM plan-bound realizer", connected_answer),
                    (baseline_name, baseline_answer)]
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
            paired_rows[-1]["_baseline_metrics"] = {
                "plan_claim_count": len(plan.claims),
                "returned_claim_count": baseline_claims,
                "factual_sentence_count": baseline_sentence_count,
                "mapped_sentence_count": baseline_mapped,
                "unsupported_sentence_count": baseline_unsupported,
                "plan_verified": plan_verified,
            }

    bound_metrics = _coverage([row["_metrics"] for row in paired_rows])
    baseline_metrics = _coverage([row["_baseline_metrics"] for row in paired_rows])
    for row in paired_rows:
        del row["_metrics"]
        del row["_baseline_metrics"]
    comparison = {
        "evaluation_set": str(evaluation_set),
        "heldout_question_count": len(heldout),
        "comparison_baseline": baseline_name,
        "same_response_plan_for_both_realizers": True,
        "plan_bound_realizer": bound_metrics,
        "baseline_realizer": baseline_metrics,
        "ranker_checkpoint": str(ranker_checkpoint) if ranker_checkpoint else None,
        "ranker_max_order": language_model.max_order if language_model else None,
        "ranker_strategy": (
            "minimum surface cross-entropy among fixed source-attribution phrases; claim text and citations remain unchanged"
            if language_model else None
        ),
        "blinded_usefulness": "pending independent ratings",
        "paired_cases": paired_rows,
    }
    if language_model is None:
        comparison["legacy_extractive_realizer"] = baseline_metrics
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
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
        "baseline_model": baseline_name,
        "case_mapping": key_rows}, indent=2) + "\n", encoding="utf-8")
    private_key.chmod(0o600)
    manifest = {"case_count": len(packet_rows), "seed": seed,
                "rating_scale": "1=not useful, 2=slightly useful, 3=moderately useful, 4=very useful, 5=extremely useful",
                "private_key_path": str(private_key),
                "task_type_counts": {"grounded_qa": len(packet_rows)}}
    (output_dir / "packet_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                       encoding="utf-8")
    result = {"plan_bound": bound_metrics, "baseline": baseline_metrics,
            "case_count": len(packet_rows), "packet": str(packet_path),
            "private_key": str(private_key)}
    if language_model is None:
        result["extractive"] = baseline_metrics
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=DEFAULT_SET)
    parser.add_argument("--output-dir", type=Path, default=Path(
        "hpm_ai_v2/experiments/results/2026-10-25-plan-bound-realizer-heldout-v1"))
    parser.add_argument("--private-key", type=Path, default=Path(
        "/tmp/hpm-planbound-realizer-heldout-key.json"))
    parser.add_argument("--ranker-checkpoint", type=Path, default=None,
                        help="optional HFN language checkpoint for fixed phrase ranking; compares against rule-based plan-bound")
    parser.add_argument("--seed", type=int, default=6247)
    args = parser.parse_args()
    print(json.dumps(run(output_dir=args.output_dir, private_key=args.private_key,
                         seed=args.seed, evaluation_set=args.set,
                         ranker_checkpoint=args.ranker_checkpoint), indent=2))


if __name__ == "__main__":
    main()
