"""Compare structured vs unstructured DailyDialog polygraph memory."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

from hpm_ai_v4.agents.reasoning import EpisodeRecord
from hpm_ai_v4.simulations import build_dailydialog_corpus as dd
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.episode_mappers import map_dailydialog_dialog


@dataclass
class PolygraphArmResult:
    name: str
    seed_dialogs: int
    seed_episodes: int
    retrieval_hits: int
    field_match_rate: float
    exact_match_rate: float
    dominant_intent_match_rate: float
    dominant_task_family_match_rate: float
    dominant_domain_match_rate: float
    average_retrieved_count: float
    average_summary_count: float
    bundle_base: str
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PolygraphComparisonResult:
    structured: PolygraphArmResult
    baseline: PolygraphArmResult
    eval_dialogs: int
    eval_episodes: int
    corpus_name: str


def _clean_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text else "unknown"


def _field_score(left: EpisodeRecord, right: EpisodeRecord) -> float:
    fields = ("domain", "task_family", "intent", "action_label", "outcome_label", "stage", "policy")
    matches = 0
    total = 0
    for field_name in fields:
        lval = _clean_label(getattr(left, field_name, "unknown"))
        rval = _clean_label(getattr(right, field_name, "unknown"))
        if lval == "unknown" or rval == "unknown":
            continue
        total += 1
        if lval == rval:
            matches += 1
    if total <= 0:
        return 0.0
    return float(matches / total)


def _exact_field_match(left: EpisodeRecord, right: EpisodeRecord) -> bool:
    relevant = ("domain", "task_family", "intent", "action_label", "outcome_label", "stage", "policy")
    for field_name in relevant:
        lval = _clean_label(getattr(left, field_name, "unknown"))
        rval = _clean_label(getattr(right, field_name, "unknown"))
        if lval == "unknown" or rval == "unknown":
            continue
        if lval != rval:
            return False
    return True


def _seed_reasoner(
    agent: LayeredAgent,
    dialogs: Sequence[Dict[str, Any]],
    *,
    structured: bool,
    max_turns: int,
) -> int:
    dialogs_written = 0
    episodes_written = 0
    for dialog_idx, row in enumerate(dialogs):
        episodes = map_dailydialog_dialog(row, max_turns=max_turns)
        if not episodes:
            continue
        dialogs_written += 1
        for turn_idx, episode in enumerate(episodes):
            if structured:
                agent.l1.reasoner.record_structured_episode(episode)
            else:
                agent.l1.reasoner.record_episode(
                    episode.context,
                    episode.action,
                    episode.reward,
                    tag=episode.tag,
                    metadata={
                        "source": "daily_dialog",
                        "dialog_index": dialog_idx,
                        "turn_index": turn_idx,
                    },
                )
            episodes_written += 1
    return dialogs_written, episodes_written


def _evaluate_arm(
    agent: LayeredAgent,
    dialogs: Sequence[Dict[str, Any]],
    *,
    structured: bool,
    bundle_base: str,
    max_turns: int,
) -> PolygraphArmResult:
    match_scores: List[float] = []
    exact_matches = 0
    dominant_intent_matches = 0
    dominant_task_family_matches = 0
    dominant_domain_matches = 0
    retrieved_counts: List[int] = []
    summary_counts: List[int] = []
    examples: List[Dict[str, Any]] = []
    eval_episodes = 0

    for row in dialogs:
        episodes = map_dailydialog_dialog(row, max_turns=max_turns)
        for episode in episodes:
            eval_episodes += 1
            feature_pack = {
                "domain": episode.domain,
                "task_family": episode.task_family,
                "intent": episode.intent,
                "action_label": episode.action_label,
                "outcome_label": episode.outcome_label,
                "stage": episode.stage,
            }
            control = agent.l1.reasoner.control_context(episode.context, feature_pack=feature_pack)
            hits = agent.l1.reasoner.retrieve_memory(
                episode.context,
                query_stage=episode.stage,
                query_policy=episode.policy,
                query_intent=episode.intent,
                query_task_family=episode.task_family,
                query_domain=episode.domain,
                query_action_label=episode.action_label,
                query_outcome_label=episode.outcome_label,
            )
            top_hit = hits[0] if hits else None
            if top_hit is not None:
                match_scores.append(_field_score(episode, top_hit))
                if _exact_field_match(episode, top_hit):
                    exact_matches += 1
            retrieved_counts.append(int(control.get("retrieved_count", 0)))
            summary_counts.append(int(control.get("summary_count", 0)))
            if control.get("dominant_intent") == episode.intent:
                dominant_intent_matches += 1
            if control.get("dominant_task_family") == episode.task_family:
                dominant_task_family_matches += 1
            if control.get("dominant_domain") == episode.domain:
                dominant_domain_matches += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "intent": episode.intent,
                        "task_family": episode.task_family,
                        "domain": episode.domain,
                        "top_hit_intent": getattr(top_hit, "intent", "none") if top_hit else "none",
                        "top_hit_task_family": getattr(top_hit, "task_family", "none") if top_hit else "none",
                        "top_hit_domain": getattr(top_hit, "domain", "none") if top_hit else "none",
                        "retrieved_count": int(control.get("retrieved_count", 0)),
                        "summary_count": int(control.get("summary_count", 0)),
                    }
                )

    seed_dialogs = len(dialogs)
    return PolygraphArmResult(
        name="structured" if structured else "baseline",
        seed_dialogs=seed_dialogs,
        seed_episodes=0,
        retrieval_hits=sum(1 for score in match_scores if score > 0.0),
        field_match_rate=float(mean(match_scores)) if match_scores else 0.0,
        exact_match_rate=float(exact_matches / max(1, eval_episodes)),
        dominant_intent_match_rate=float(dominant_intent_matches / max(1, eval_episodes)),
        dominant_task_family_match_rate=float(dominant_task_family_matches / max(1, eval_episodes)),
        dominant_domain_match_rate=float(dominant_domain_matches / max(1, eval_episodes)),
        average_retrieved_count=float(mean(retrieved_counts)) if retrieved_counts else 0.0,
        average_summary_count=float(mean(summary_counts)) if summary_counts else 0.0,
        bundle_base=bundle_base,
        examples=examples,
    )


def run_dailydialog_polygraph_benchmark(
    *,
    output_dir: str,
    dataset_name: str = "OpenRL/daily_dialog",
    split: str = "train",
    train_limit: int = 128,
    eval_limit: int = 32,
    max_turns: int = 8,
) -> PolygraphComparisonResult:
    os.makedirs(output_dir, exist_ok=True)
    dataset = list(dd._load_dataset(dataset_name, split=split))
    train_dialogs = dataset[:train_limit]
    eval_dialogs = dataset[train_limit : train_limit + eval_limit]
    if not eval_dialogs:
        eval_dialogs = dataset[max(0, len(dataset) - eval_limit) :]
    if not train_dialogs:
        raise RuntimeError("No training dialogs available for polygraph benchmark")
    if not eval_dialogs:
        raise RuntimeError("No evaluation dialogs available for polygraph benchmark")

    structured_agent = LayeredAgent(num_workers=1)
    baseline_agent = LayeredAgent(num_workers=1)

    structured_seed_dialogs, structured_seed_episodes = _seed_reasoner(
        structured_agent,
        train_dialogs,
        structured=True,
        max_turns=max_turns,
    )
    baseline_seed_dialogs, baseline_seed_episodes = _seed_reasoner(
        baseline_agent,
        train_dialogs,
        structured=False,
        max_turns=max_turns,
    )

    structured_bundle = os.path.join(output_dir, "structured_polygraph_bundle")
    baseline_bundle = os.path.join(output_dir, "baseline_polygraph_bundle")
    structured_agent.save_bundle(structured_bundle)
    baseline_agent.save_bundle(baseline_bundle)

    structured_result = _evaluate_arm(
        structured_agent,
        eval_dialogs,
        structured=True,
        bundle_base=structured_bundle,
        max_turns=max_turns,
    )
    baseline_result = _evaluate_arm(
        baseline_agent,
        eval_dialogs,
        structured=False,
        bundle_base=baseline_bundle,
        max_turns=max_turns,
    )
    structured_result.seed_dialogs = structured_seed_dialogs
    structured_result.seed_episodes = structured_seed_episodes
    baseline_result.seed_dialogs = baseline_seed_dialogs
    baseline_result.seed_episodes = baseline_seed_episodes

    report_path = os.path.join(output_dir, "polygraph_comparison.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "structured": structured_result.__dict__,
                "baseline": baseline_result.__dict__,
                "eval_dialogs": len(eval_dialogs),
                "eval_episodes": sum(len(map_dailydialog_dialog(row, max_turns=max_turns)) for row in eval_dialogs),
                "dataset_name": dataset_name,
                "split": split,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    return PolygraphComparisonResult(
        structured=structured_result,
        baseline=baseline_result,
        eval_dialogs=len(eval_dialogs),
        eval_episodes=sum(len(map_dailydialog_dialog(row, max_turns=max_turns)) for row in eval_dialogs),
        corpus_name=f"{dataset_name}:{split}",
    )


def _print_report(result: PolygraphComparisonResult) -> None:
    print("\n" + "=" * 72)
    print("HPM DAILYDIALOG POLYGRAPH COMPARISON")
    print("=" * 72)
    for arm in (result.structured, result.baseline):
        print(
            f"{arm.name}: field_match={arm.field_match_rate:.3f} "
            f"exact_match={arm.exact_match_rate:.3f} "
            f"intent={arm.dominant_intent_match_rate:.3f} "
            f"task_family={arm.dominant_task_family_match_rate:.3f} "
            f"domain={arm.dominant_domain_match_rate:.3f} "
            f"retrieved={arm.average_retrieved_count:.2f} "
            f"summaries={arm.average_summary_count:.2f}"
        )
    print(
        f"delta_field_match={(result.structured.field_match_rate - result.baseline.field_match_rate):.3f} "
        f"delta_exact={(result.structured.exact_match_rate - result.baseline.exact_match_rate):.3f} "
        f"delta_summary={(result.structured.average_summary_count - result.baseline.average_summary_count):.3f}"
    )
    print("=" * 72)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare structured vs unstructured DailyDialog polygraphs")
    p.add_argument("--output-dir", default="library_bootstrap/dailydialog_polygraph_compare", help="Output directory")
    p.add_argument("--dataset-name", default="OpenRL/daily_dialog", help="Hugging Face dataset name")
    p.add_argument("--split", default="train", help="Dataset split")
    p.add_argument("--train-limit", type=int, default=128, help="Training dialogs")
    p.add_argument("--eval-limit", type=int, default=32, help="Evaluation dialogs")
    p.add_argument("--max-turns", type=int, default=8, help="Max turns per dialog")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_dailydialog_polygraph_benchmark(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        max_turns=args.max_turns,
    )
    _print_report(result)
