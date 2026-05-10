"""Shared concept-profile helpers for intent benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ConceptRecord:
    concept_id: str
    concept_kind: str
    evidence: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class IntentProfile:
    route_keywords: dict[str, frozenset[str]]
    route_concepts: dict[str, str]
    query_terms: dict[str, frozenset[str]]
    entity_concepts: dict[str, str]
    broad_concepts: frozenset[str] = field(default_factory=frozenset)


def build_shared_anchor_payload(
    *,
    tokens: list[str],
    canonical_tokens: list[str],
    ent_types: list[str],
    content_words: list[str],
    skeleton: list[str],
    ngrams: list[str],
    profile: IntentProfile,
    extra_route_detector=None,
    extra_concept_builder=None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, object]]]:
    token_set = {token.lower() for token in tokens}
    canon_set = set(canonical_tokens)

    routes: list[str] = []
    for route, keywords in profile.route_keywords.items():
        hits = len(canon_set.intersection(keywords))
        if hits >= 2 or (hits >= 1 and route in token_set):
            routes.append(route)
    if extra_route_detector is not None:
        routes.extend(extra_route_detector(tokens=tokens, canonical_tokens=canonical_tokens, ent_types=ent_types))
    if not routes:
        routes.append("general")
    routes = sorted(set(routes))

    concepts: dict[str, ConceptRecord] = {}

    def add(concept_id: str, concept_kind: str, *evidence: str) -> None:
        prior = concepts.get(concept_id)
        merged = set(prior.evidence if prior is not None else ())
        merged.update(item for item in evidence if item)
        concepts[concept_id] = ConceptRecord(
            concept_id=concept_id,
            concept_kind=concept_kind,
            evidence=tuple(sorted(merged)),
        )

    for route in routes:
        concept_id = profile.route_concepts.get(route)
        if concept_id:
            add(concept_id, "route_concept", route)

    for concept_id, words in profile.query_terms.items():
        overlap = token_set.intersection(words)
        if overlap:
            add(concept_id, "query_concept", *sorted(overlap))

    for ent_type in set(ent_types):
        concept_id = profile.entity_concepts.get(ent_type)
        if concept_id:
            add(concept_id, "slot_concept", ent_type)

    if extra_concept_builder is not None:
        for record in extra_concept_builder(
            tokens=tokens,
            canonical_tokens=canonical_tokens,
            ent_types=ent_types,
            routes=routes,
        ):
            add(record["concept_id"], record["concept_kind"], *record.get("evidence", ()))

    anchors: list[dict[str, object]] = [{
        "anchor_id": "intent::utterance",
        "anchor_kind": "intent_anchor",
        "features": tuple(sorted(set(canonical_tokens))),
    }]
    for route in routes:
        anchors.append({
            "anchor_id": f"route::{route}",
            "anchor_kind": "route_anchor",
            "features": (route,),
        })

    for ent in sorted({ent for ent in ent_types if ent}):
        anchors.append({
            "anchor_id": f"slot::{ent.lower()}",
            "anchor_kind": "slot_anchor",
            "features": (ent,),
        })

    if skeleton:
        anchors.append({
            "anchor_id": f"relation::{'_'.join(skeleton[:3])}",
            "anchor_kind": "relation_anchor",
            "features": tuple(skeleton[:3]),
        })

    concept_ids = tuple(sorted(concepts))
    common_anchor_ids = ("intent::utterance", *[f"route::{route}" for route in routes])
    slot_anchor_ids = tuple(a["anchor_id"] for a in anchors if str(a["anchor_id"]).startswith("slot::"))
    relation_anchor_ids = tuple(a["anchor_id"] for a in anchors if str(a["anchor_id"]).startswith("relation::"))
    view_anchor_map = {
        "token_view": {
            "leaf_keys": tuple(f"tok:{token}" for token in tokens),
            "anchor_ids": tuple(common_anchor_ids + slot_anchor_ids),
            "concept_ids": concept_ids,
        },
        "canonical_view": {
            "leaf_keys": tuple(f"canon:{token}" for token in canonical_tokens),
            "anchor_ids": tuple(common_anchor_ids + slot_anchor_ids),
            "concept_ids": concept_ids,
        },
        "content_view": {
            "leaf_keys": tuple(f"content:{word}" for word in content_words),
            "anchor_ids": tuple(common_anchor_ids),
            "concept_ids": concept_ids,
        },
        "skeleton_view": {
            "leaf_keys": tuple(f"sk:{tag}" for tag in skeleton),
            "anchor_ids": tuple(common_anchor_ids + relation_anchor_ids),
            "concept_ids": concept_ids,
        },
        "skeleton_bigram_view": {
            "leaf_keys": tuple(f"sk2:{ngram}" for ngram in ngrams),
            "anchor_ids": tuple(common_anchor_ids + relation_anchor_ids),
            "concept_ids": concept_ids,
        },
    }

    return anchors, [
        {"concept_id": record.concept_id, "concept_kind": record.concept_kind, "evidence": record.evidence}
        for _, record in sorted(concepts.items())
    ], view_anchor_map
