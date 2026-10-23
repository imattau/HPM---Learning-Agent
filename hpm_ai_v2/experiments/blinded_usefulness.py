"""Build and analyze blinded pairwise usefulness packets for HPM vs a baseline.

The rating packet contains no system names or benchmark labels. Keep its
separate deblinding key private until ratings are complete.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from hpm_ai_v2.training import load_curriculum_jsonl


DEFAULT_TASK_TYPES = ("grounded_qa", "correction", "compare", "preference", "abstain")


def build_blinded_packet(
    hpm_report_path: str | Path,
    baseline_report_path: str | Path,
    curriculum_paths: list[str | Path],
    source_state_path: str | Path,
    output_dir: str | Path,
    private_key_path: str | Path,
    *,
    seed: int = 20261024,
    task_types: tuple[str, ...] = DEFAULT_TASK_TYPES,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Prepare randomized A/B answer pairs and a separate private mapping key."""
    hpm_report = json.loads(Path(hpm_report_path).read_text(encoding="utf-8"))
    baseline_report = json.loads(Path(baseline_report_path).read_text(encoding="utf-8"))
    hpm_rows = hpm_report.get("final_exam", {}).get("episode_results", [])
    if not hpm_rows:
        hpm_rows = hpm_report.get("retention_after_all_phases", {}).get("episode_results", [])
    baseline_key = next(
        (
            key for key in ("results", "ollama_retrieval_baseline", "ollama_baseline")
            if baseline_report.get(key, {}).get("episodes")
        ),
        None,
    )
    baseline_rows = baseline_report.get(baseline_key, {}).get("episodes", []) if baseline_key else []
    if not hpm_rows or not baseline_rows:
        raise ValueError("reports must contain HPM final_exam and baseline episodes")
    hpm_by_id = _unique_index(hpm_rows, "episode_id", "HPM report")
    baseline_by_id = _unique_index(baseline_rows, "episode_id", "baseline report")
    if hpm_by_id.keys() != baseline_by_id.keys():
        raise ValueError("HPM and baseline reports must contain the same episode IDs")

    episode_by_id = {}
    conflicting_duplicate_ids = set()
    for path in curriculum_paths:
        _sources, episodes = load_curriculum_jsonl(path)
        for episode in episodes:
            previous = episode_by_id.get(episode.episode_id)
            if previous is not None and previous != episode:
                # Preserve first-seen precedence, matching the cumulative exam
                # assembly. Expose collisions in the manifest for review.
                conflicting_duplicate_ids.add(episode.episode_id)
                continue
            episode_by_id.setdefault(episode.episode_id, episode)
    missing_episodes = hpm_by_id.keys() - episode_by_id.keys()
    if missing_episodes:
        raise ValueError(f"curricula are missing report episodes: {sorted(missing_episodes)[:5]}")

    source_state = json.loads(Path(source_state_path).read_text(encoding="utf-8"))
    if isinstance(source_state, list):
        sources = source_state
    elif isinstance(source_state, dict):
        sources = source_state.get("sources", [])
    else:
        raise ValueError("source state must be a list or an object with a sources list")
    if not isinstance(sources, list):
        raise ValueError("source state must be a list or an object with a sources list")
    source_by_id = _unique_index(sources, "source_id", "source state")

    selected_by_task = defaultdict(list)
    excluded = defaultdict(int)
    for episode_id in hpm_by_id:
        episode = episode_by_id[episode_id]
        if episode.task_type not in task_types:
            excluded[episode.task_type] += 1
            continue
        selected_by_task[episode.task_type].append(episode_id)
    selected = [episode_id for values in selected_by_task.values() for episode_id in values]
    if not selected:
        raise ValueError("no answer-bearing episodes remain after task-family filtering")
    if sample_size is not None and not 1 <= sample_size <= len(selected):
        raise ValueError(f"sample_size must be between 1 and {len(selected)}")

    rng = random.Random(seed)
    hpm_system_name = hpm_report.get("configuration", {}).get("hpm_model", "HPM")
    baseline_block = baseline_report.get(baseline_key, {}) if baseline_key else {}
    baseline_system_name = (
        baseline_block.get("model") or baseline_report.get("model") or "9B baseline"
    )
    if hpm_system_name == baseline_system_name:
        raise ValueError("HPM and baseline system labels must be distinct for deblinding")
    for values in selected_by_task.values():
        rng.shuffle(values)
    if sample_size is not None:
        # Round-robin families so a small per-rater packet still covers the
        # different answer types instead of being dominated by grounded QA.
        balanced = []
        task_names = sorted(selected_by_task)
        while len(balanced) < sample_size:
            advanced = False
            for task_name in task_names:
                if selected_by_task[task_name]:
                    balanced.append(selected_by_task[task_name].pop())
                    advanced = True
                    if len(balanced) == sample_size:
                        break
            if not advanced:
                break
        selected = balanced
    rng.shuffle(selected)
    packet_rows = []
    key_rows = []
    template_rows = []
    for index, episode_id in enumerate(selected, start=1):
        episode = episode_by_id[episode_id]
        hpm_row = hpm_by_id[episode_id]
        baseline_row = baseline_by_id[episode_id]
        pair = [
            (hpm_system_name, hpm_row),
            (baseline_system_name, baseline_row),
        ]
        if rng.random() < 0.5:
            pair.reverse()
        case_id = f"case-{index:04d}"
        selected_sources = {
            source_id
            for _system, result_row in pair
            for source_id in (
                list(result_row.get("selected_evidence_ids") or [])
                or ([result_row["selected_evidence_id"]] if result_row.get("selected_evidence_id") else [])
            )
            if source_id in source_by_id
        }
        referenced_ids = {
            source_id
            for _system, result_row in pair
            for source_id in (
                list(result_row.get("selected_evidence_ids") or [])
                or ([result_row["selected_evidence_id"]] if result_row.get("selected_evidence_id") else [])
            )
        }
        missing_sources = referenced_ids - source_by_id.keys()
        if missing_sources:
            raise ValueError(f"reports cite sources absent from source state: {sorted(missing_sources)[:5]}")
        display_sources = {
            source_id: f"Source {i}"
            for i, source_id in enumerate(sorted(selected_sources), start=1)
        }
        answer_a = _display_answer(pair[0][1], source_by_id, display_sources)
        answer_b = _display_answer(pair[1][1], source_by_id, display_sources)
        packet_rows.append({
            "case_id": case_id,
            "task_type": episode.task_type,
            "question": episode.question,
            "response_a": answer_a,
            "response_b": answer_b,
        })
        key_rows.append({
            "case_id": case_id,
            "episode_id": episode_id,
            "response_a_system": pair[0][0],
            "response_b_system": pair[1][0],
            "expected_evidence_id": episode.evidence_id,
            "expected_evidence_ids": list(episode.evidence_ids),
            "hpm_task_correct": _episode_correct(hpm_row, episode),
            "baseline_task_correct": bool(baseline_row.get("task_correct")),
            "hpm_answer_token_f1": hpm_row.get("answer_token_f1"),
            "baseline_answer_token_f1": baseline_row.get("answer_token_f1"),
        })
        template_rows.append({
            "case_id": case_id,
            "rater_id": "",
            "usefulness_a": None,
            "usefulness_b": None,
            "preferred": None,
            "comment": "",
        })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    key_path = Path(private_key_path)
    if key_path.resolve().is_relative_to(out.resolve()):
        raise ValueError("private key output must be outside the shareable packet directory")
    _write_jsonl(out / "rating_packet.jsonl", packet_rows)
    _write_jsonl(out / "ratings_template.jsonl", template_rows)
    _write_rating_csv(out / "ratings_template.csv", template_rows)
    (out / "rating_instructions.md").write_text(
        "# Pairwise usefulness ratings\n\n"
        "Rate the two anonymous answers for each case. Treat any cited source text as part of that answer. "
        "First score each answer independently: 1 = not useful, 2 = slightly useful, 3 = moderately useful, "
        "4 = very useful, 5 = extremely useful. Then choose whether you prefer A, B, or neither (tie). "
        "Comments are optional. Do not try to identify which system produced an answer. The separate deblinding key must remain private until all ratings are collected. "
        "For a browser interface, open `hpm_ai_v2/experiments/blinded_rating_app.html`, load `rating_packet.jsonl`, enter a rater ID, complete the cases, and download `ratings.csv`. The app stores drafts only in that browser on that device.\n",
        encoding="utf-8",
    )
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(
        json.dumps({
            "seed": seed,
            "hpm_model": hpm_system_name,
            "baseline_model": baseline_system_name,
            "case_mapping": key_rows,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    key_path.chmod(0o600)
    manifest = {
        "packet_count": len(packet_rows),
        "sample_size": sample_size,
        "excluded_task_counts": dict(sorted(excluded.items())),
        "conflicting_duplicate_episode_ids": sorted(conflicting_duplicate_ids),
        "task_counts": dict(sorted(_count_rows(packet_rows, "task_type").items())),
        "seed": seed,
        "rating_scale": "1=not useful, 2=slightly useful, 3=moderately useful, 4=very useful, 5=extremely useful",
        "instructions": "Rate each answer independently, then choose which you prefer or tie. Do not open the private deblinding key until collection is complete.",
    }
    (out / "packet_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def analyze_ratings(
    packet_path: str | Path,
    ratings_path: str | Path,
    key_path: str | Path,
    output_path: str | Path | None = None,
    *,
    min_ratings_per_rater: int = 5,
) -> dict[str, Any]:
    """Unblind completed ratings and summarize paired usefulness and preference."""
    if min_ratings_per_rater < 1:
        raise ValueError("min_ratings_per_rater must be at least one")
    packet = _read_jsonl(packet_path)
    ratings = _read_ratings(ratings_path)
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    packet_ids = {row["case_id"] for row in packet}
    key_by_case = _unique_index(key.get("case_mapping", []), "case_id", "private key")
    if packet_ids != key_by_case.keys():
        raise ValueError("rating packet and private key case IDs do not match")
    seen = set()
    enriched = []
    for row in ratings:
        case_id, rater_id = row.get("case_id"), str(row.get("rater_id", "")).strip()
        if case_id not in packet_ids:
            raise ValueError(f"rating references unknown case_id: {case_id}")
        if not rater_id:
            raise ValueError(f"rating for {case_id} is missing rater_id")
        pair_key = (rater_id, case_id)
        if pair_key in seen:
            raise ValueError(f"duplicate rating by rater {rater_id} for {case_id}")
        seen.add(pair_key)
        score_a = _rating_score(row, "usefulness_a", case_id)
        score_b = _rating_score(row, "usefulness_b", case_id)
        preference = row.get("preferred")
        if preference not in {"A", "B", "tie"}:
            raise ValueError(f"rating for {case_id} must set preferred to A, B, or tie")
        hidden = key_by_case[case_id]
        scores = {
            hidden["response_a_system"]: score_a,
            hidden["response_b_system"]: score_b,
        }
        preference_winner = (
            "tie" if preference == "tie"
            else hidden[f"response_{preference.lower()}_system"]
        )
        enriched.append({
            "rater_id": rater_id,
            "case_id": case_id,
            "task_type": next(item["task_type"] for item in packet if item["case_id"] == case_id),
            "scores": scores,
            "useful": {system: score >= 4 for system, score in scores.items()},
            "preference_winner": preference_winner,
        })
    if not enriched:
        raise ValueError("ratings file is empty")

    by_rater: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_rater[row["rater_id"]].append(row)
    systems = sorted({name for row in enriched for name in row["scores"]})
    aggregate = {}
    for system in systems:
        scores = [row["scores"][system] for row in enriched]
        aggregate[system] = {
            "ratings": len(scores),
            "mean_usefulness_1_to_5": _mean(scores),
            "useful_or_very_useful_rate": sum(score >= 4 for score in scores) / len(scores),
            "preference_wins": sum(row["preference_winner"] == system for row in enriched),
        }
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_task[row["task_type"]].append(row)
    per_task_type = {}
    for task_type, rows in sorted(by_task.items()):
        per_task_type[task_type] = {}
        for system in systems:
            scores = [row["scores"][system] for row in rows]
            per_task_type[task_type][system] = {
                "ratings": len(scores),
                "mean_usefulness_1_to_5": _mean(scores),
                "useful_or_very_useful_rate": sum(score >= 4 for score in scores) / len(scores),
            }
    hpm_name = key.get("hpm_model", "HPM")
    baseline_name = key.get("baseline_model", "9B baseline")
    rater_results = {}
    for rater, rows in sorted(by_rater.items()):
        n = len(rows)
        hpm_rate = sum(row["useful"][hpm_name] for row in rows) / n
        baseline_rate = sum(row["useful"][baseline_name] for row in rows) / n
        rater_results[rater] = {
            "rated_pairs": n,
            "hpm_useful_rate": hpm_rate,
            "baseline_useful_rate": baseline_rate,
            "hpm_minus_baseline": hpm_rate - baseline_rate,
            "eligible_for_user_gate": n >= min_ratings_per_rater,
            "passes_user_gate": (
                hpm_rate >= 0.80 and hpm_rate - baseline_rate >= -0.10
                if n >= min_ratings_per_rater else None
            ),
        }
    eligible = [row for row in rater_results.values() if row["eligible_for_user_gate"]]
    user_gate_rate = (
        sum(bool(row["passes_user_gate"]) for row in eligible) / len(eligible)
        if eligible else None
    )
    user_level_summary = None
    user_gate_ci = None
    if eligible:
        hpm_user_rates = [row["hpm_useful_rate"] for row in eligible]
        baseline_user_rates = [row["baseline_useful_rate"] for row in eligible]
        difference_user_rates = [row["hpm_minus_baseline"] for row in eligible]
        bootstrap_seed = int(key.get("seed", 20261024))
        user_level_summary = {
            "eligible_raters": len(eligible),
            "mean_hpm_useful_rate": _mean_float(hpm_user_rates),
            "mean_hpm_useful_rate_cluster_bootstrap_ci95": _bootstrap_ci(hpm_user_rates, bootstrap_seed),
            "mean_baseline_useful_rate": _mean_float(baseline_user_rates),
            "mean_baseline_useful_rate_cluster_bootstrap_ci95": _bootstrap_ci(baseline_user_rates, bootstrap_seed + 1),
            "mean_hpm_minus_baseline": _mean_float(difference_user_rates),
            "mean_hpm_minus_baseline_cluster_bootstrap_ci95": _bootstrap_ci(difference_user_rates, bootstrap_seed + 2),
        }
        passed_users = sum(bool(row["passes_user_gate"]) for row in eligible)
        user_gate_ci = _wilson_interval(passed_users, len(eligible))
    hpm_wins = aggregate[hpm_name]["preference_wins"]
    baseline_wins = aggregate[baseline_name]["preference_wins"]
    ties = sum(row["preference_winner"] == "tie" for row in enriched)
    result = {
        "rating_count": len(enriched),
        "case_count_rated": len({row["case_id"] for row in enriched}),
        "rater_count": len(by_rater),
        "per_system": aggregate,
        "per_task_type": per_task_type,
        "pairwise_preference": {
            "hpm_wins": hpm_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
        },
        "per_rater": rater_results,
        "user_acceptance_gate": {
            "minimum_ratings_per_user": min_ratings_per_rater,
            "eligible_users": len(eligible),
            "users_passing_both_thresholds": sum(bool(row["passes_user_gate"]) for row in eligible),
            "passing_user_rate": user_gate_rate,
            "passing_user_rate_wilson_ci95": user_gate_ci,
            "user_level_summary": user_level_summary,
            "threshold": "at least 80% of eligible raters must give HPM >=80% useful ratings and HPM may be at most 10 percentage points below the baseline",
            "passed": user_gate_rate >= 0.80 if user_gate_rate is not None else None,
        },
        "interpretation_limit": "Descriptive ratings reflect this packet and these raters; repeated ratings are clustered by person and do not establish population-wide preference.",
    }
    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _display_answer(row: dict, source_by_id: dict, source_label_by_id: dict) -> dict[str, Any]:
    text = str(row.get("answer") or "").strip()
    if row.get("abstained") and not text:
        text = "I do not have enough information in the available material to answer."
    elif not text:
        text = "[No answer returned]"
    evidence_ids = list(row.get("selected_evidence_ids") or [])
    if not evidence_ids and row.get("selected_evidence_id"):
        evidence_ids = [row["selected_evidence_id"]]
    citations = []
    for source_id in evidence_ids:
        source = source_by_id.get(source_id)
        if source is None:
            continue
        label = source_label_by_id[source_id]
        citations.append({"label": label, "title": source.get("title") or source_id, "text": source["text"]})
    return {"text": text, "sources": citations}


def _unique_index(rows: list[dict], key: str, label: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        value = row.get(key)
        if value in indexed:
            raise ValueError(f"duplicate {key} {value!r} in {label}")
        indexed[value] = row
    return indexed


def _rating_score(row: dict, key: str, case_id: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise ValueError(f"{case_id} {key} must be an integer from 1 to 5")
    return value


def _episode_correct(row: dict, episode) -> bool:
    if episode.task_type == "compare":
        return set(row.get("selected_evidence_ids", [])) == set(episode.evidence_ids)
    if episode.task_type == "preference":
        return row.get("answer") == episode.expected_answer
    if episode.task_type == "procedure":
        return row.get("selected_workflow_key") == episode.workflow_key
    return row.get("selected_evidence_id") == episode.evidence_id


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _read_ratings(path: str | Path) -> list[dict]:
    target = Path(path)
    if target.suffix.lower() != ".csv":
        return _read_jsonl(target)
    with target.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in ("usefulness_a", "usefulness_b"):
            row[field] = int(row[field]) if row.get(field, "").strip() else None
        row["preferred"] = row.get("preferred") or None
        row["comment"] = row.get("comment") or ""
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_rating_csv(path: Path, rows: list[dict]) -> None:
    fields = ("case_id", "rater_id", "usefulness_a", "usefulness_b", "preferred", "comment")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _count_rows(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row[key]] += 1
    return counts


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_float(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bootstrap_ci(values: list[float], seed: int, replicates: int = 2000) -> list[float] | None:
    """Percentile interval resampling raters as clusters, not individual rows."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(
        _mean_float([rng.choice(values) for _ in values])
        for _ in range(replicates)
    )
    return [means[int(0.025 * replicates)], means[min(replicates - 1, int(0.975 * replicates))]]


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="create a blinded rating packet and private key")
    build.add_argument("--hpm-report", type=Path, required=True)
    build.add_argument("--baseline-report", type=Path, required=True)
    build.add_argument("--curriculum", type=Path, action="append", required=True)
    build.add_argument("--source-state", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--private-key-output", type=Path, required=True)
    build.add_argument("--seed", type=int, default=20261024)
    build.add_argument("--sample-size", type=int, help="balanced item count for one rater; omit to include all eligible items")
    analyze = sub.add_parser("analyze", help="analyze completed ratings after deblinding")
    analyze.add_argument("--packet", type=Path, required=True)
    analyze.add_argument("--ratings", type=Path, required=True)
    analyze.add_argument("--key", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--min-ratings-per-rater", type=int, default=5)
    args = parser.parse_args()
    if args.command == "build":
        result = build_blinded_packet(
            args.hpm_report, args.baseline_report, args.curriculum, args.source_state,
            args.output_dir, args.private_key_output, seed=args.seed,
            sample_size=args.sample_size,
        )
    else:
        result = analyze_ratings(
            args.packet, args.ratings, args.key, args.output,
            min_ratings_per_rater=args.min_ratings_per_rater,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
