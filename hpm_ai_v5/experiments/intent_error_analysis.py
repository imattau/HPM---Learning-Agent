"""Error analysis helpers for intent benchmarks."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class IntentErrorRecord:
    text: str
    gold_intent: str
    predicted_intent: str | None
    top_source: str | None
    top_margin: float
    source_conflict: bool
    slot_sensitive: bool
    categories: tuple[str, ...]


def _top_margin(votes: dict[str, float]) -> float:
    if len(votes) < 2:
        return next(iter(votes.values()), 0.0)
    ordered = sorted(votes.values(), reverse=True)
    return ordered[0] - ordered[1]


def _dominant_source(vote_trace: list[dict]) -> str | None:
    source_scores: dict[str, float] = defaultdict(float)
    for row in vote_trace:
        source_scores[str(row.get("source"))] += float(row.get("contribution", 0.0))
    if not source_scores:
        return None
    return max(source_scores, key=source_scores.__getitem__)


def _source_conflict(vote_trace: list[dict]) -> bool:
    seen: dict[str, set[str]] = defaultdict(set)
    for row in vote_trace:
        source = str(row.get("source"))
        intent = row.get("intent")
        if intent:
            seen[source].add(str(intent))
    intents = {next(iter(values)) for values in seen.values() if values}
    return len(intents) >= 2


def _slot_sensitive(text: str) -> bool:
    lowered = f" {text.lower()} "
    return " from " in lowered and " to " in lowered


def classify_error(text: str, gold_intent: str, predicted_intent: str | None, context: dict) -> IntentErrorRecord:
    votes = dict(context.get("intent_votes", {}))
    vote_trace = list(context.get("intent_vote_trace", []))
    top_source = _dominant_source(vote_trace)
    margin = _top_margin(votes)
    conflict = _source_conflict(vote_trace)
    slot_sensitive = _slot_sensitive(text)

    categories: list[str] = []
    if predicted_intent is None:
        categories.append("no_prediction")
    if margin < 0.2:
        categories.append("low_margin")
    if top_source is not None:
        categories.append(f"dominant_source:{top_source}")
    if conflict:
        categories.append("source_conflict")
    if slot_sensitive:
        categories.append("slot_sensitive_phrase")
    if predicted_intent is not None and predicted_intent != gold_intent:
        categories.append(f"confused_with:{predicted_intent}")

    return IntentErrorRecord(
        text=text,
        gold_intent=gold_intent,
        predicted_intent=predicted_intent,
        top_source=top_source,
        top_margin=margin,
        source_conflict=conflict,
        slot_sensitive=slot_sensitive,
        categories=tuple(categories),
    )


def analyze_intent_errors(
    dataset: list[dict],
    predict_with_context: Callable[[str], tuple[str | None, dict]],
) -> dict[str, object]:
    errors: list[IntentErrorRecord] = []
    category_counts: Counter[str] = Counter()
    confusion: Counter[tuple[str, str | None]] = Counter()

    for item in dataset:
        predicted, context = predict_with_context(item["text"])
        if predicted == item["intent"]:
            continue
        record = classify_error(item["text"], item["intent"], predicted, context)
        errors.append(record)
        category_counts.update(record.categories)
        confusion[(item["intent"], predicted)] += 1

    top_confusions = [
        {"gold": gold, "predicted": predicted, "count": count}
        for (gold, predicted), count in confusion.most_common(10)
    ]
    samples = [
        {
            "text": record.text,
            "gold": record.gold_intent,
            "predicted": record.predicted_intent,
            "top_source": record.top_source,
            "top_margin": record.top_margin,
            "categories": list(record.categories),
        }
        for record in errors[:10]
    ]
    return {
        "n_errors": len(errors),
        "category_counts": dict(category_counts),
        "top_confusions": top_confusions,
        "samples": samples,
    }
