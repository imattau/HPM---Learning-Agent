"""ATIS-specific wrappers around the shared intent benchmark scaffold."""

from __future__ import annotations

from hpm_ai_v5.adapter.atis import ATISBridgeAnchorAdapter, IntentLabelAdapter

from .intent_shared import IntentBenchmarkSupport, build_intent_pipeline, top_k_projection_candidates


class ATISBenchmarkSupport(IntentBenchmarkSupport):
    """Backwards-compatible ATIS wrapper over the shared intent benchmark support."""

    def __init__(self, *, intent_adapter, **kwargs):
        super().__init__(label_adapter=intent_adapter, **kwargs)


def build_atis_pipeline(engine, **kwargs):
    return build_intent_pipeline(
        engine,
        label_adapter=IntentLabelAdapter(),
        bridge_adapter=ATISBridgeAnchorAdapter(),
        **kwargs,
    )


def evaluate_novel_entity_accuracy(
    train: list[dict],
    test: list[dict],
    *,
    nlp,
    predictor,
    entity_types: set[str] | None = None,
    train_limit: int | None = None,
) -> tuple[float, int]:
    entity_types = entity_types or {"GPE", "ORG", "LOC", "FAC"}

    def _entity_set(text: str) -> set[str]:
        return {ent.text.lower() for ent in nlp(text).ents if ent.label_ in entity_types}

    train_entities: set[str] = set()
    train_iter = train if train_limit is None else train[:train_limit]
    for item in train_iter:
        train_entities.update(_entity_set(item["text"]))

    novel = [i for i in test if _entity_set(i["text"]) - train_entities]
    if not novel:
        return 0.0, 0

    correct = sum(1 for item in novel if predictor(item["text"]) == item["intent"])
    return correct / len(novel), len(novel)
