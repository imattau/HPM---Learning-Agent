"""ATIS dataset loader and bridge-aware adapters for HPM v5."""
from __future__ import annotations
import csv
import os
from dataclasses import dataclass, field
from .base import Adapter
from .packet import AdapterPacket
from .intent_concepts import IntentProfile, build_shared_anchor_payload


def load_atis() -> tuple[list[dict], list[dict]]:
    """Return (train, test) as lists of {text, intent} dicts from local CSVs."""
    
    def _read_csv(file_path: str) -> list[dict]:
        data = []
        if not os.path.exists(file_path):
            return data
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({"text": row["text"], "intent": row["intent"]})
        return data

    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train = _read_csv(os.path.join(_repo, "data", "atis", "atis_train.csv"))
    test = _read_csv(os.path.join(_repo, "data", "atis", "atis_test.csv"))
    
    return train, test


@dataclass(slots=True)
class IntentLabelAdapter(Adapter):
    """Inject gold intent label during training; omit during inference."""

    name: str = "intent_label"
    label: str | None = None
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["intent_label"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        if self.label is not None:
            packet.context["intent_label"] = self.label
        return packet


@dataclass(slots=True)
class ATISBridgeAnchorAdapter(Adapter):
    """Emit lightweight utterance-local bridge anchors for interconnected polygraphs."""

    name: str = "atis_bridge_anchor"
    requires: list[str] = field(default_factory=lambda: [
        "ner_canonicaliser",
        "content_word_extractor",
        "skeleton_extractor",
        "skeleton_ngram",
    ])
    provides: list[str] = field(default_factory=lambda: ["atis_bridge_anchors", "atis_view_anchor_map", "atis_concepts"])

    _PROFILE: IntentProfile = field(default_factory=lambda: IntentProfile(
        route_keywords={
            "flight": frozenset({"FLIGHT", "BOOK", "LEAVE", "ARRIVE"}),
            "airfare": frozenset({"AIRFARE", "FARE", "PRICE"}),
            "airport": frozenset({"AIRPORT", "TERMINAL"}),
            "meal": frozenset({"MEAL", "BREAKFAST", "LUNCH", "DINNER"}),
        },
        route_concepts={
            "flight": "concept::flight_query",
            "airfare": "concept::airfare_query",
            "airport": "concept::airport_query",
            "ground_transport": "concept::ground_transport_query",
        },
        query_terms={
            "concept::listing_request": frozenset({"list", "show", "find", "give", "display"}),
            "concept::availability_request": frozenset({"available", "availability", "leaving", "departing", "arriving"}),
        },
        entity_concepts={
            "DATE": "concept::date_constraint",
            "TIME": "concept::time_constraint",
            "ORG": "concept::airline_constraint",
            "FAC": "concept::airport_constraint",
        },
        broad_concepts=frozenset({"concept::listing_request", "concept::availability_request"}),
    ), init=False)
    _ENTITY_TAGS: frozenset[str] = field(
        default_factory=lambda: frozenset({"GPE", "ORG", "DATE", "TIME", "FAC", "LOC"}),
        init=False,
    )
    _TRANSPORT_WORDS: frozenset[str] = field(
        default_factory=lambda: frozenset({"limo", "taxi", "ground", "transport", "car", "rental"}),
        init=False,
    )
    _AIRLINE_WORDS: frozenset[str] = field(
        default_factory=lambda: frozenset({"airline", "airlines", "carrier", "carriers", "serve", "serves"}),
        init=False,
    )
    _AIRFARE_WORDS: frozenset[str] = field(
        default_factory=lambda: frozenset({"fare", "fares", "airfare", "cost", "price", "prices"}),
        init=False,
    )
    _DAY_NAME_WORDS: frozenset[str] = field(
        default_factory=lambda: frozenset({"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "days", "day", "week"}),
        init=False,
    )
    _TIME_WORDS: frozenset[str] = field(
        default_factory=lambda: frozenset({"time", "times", "arrival", "arrive", "depart", "departure", "leaving", "arriving"}),
        init=False,
    )
    _LISTING_WORDS: frozenset[str] = field(
        default_factory=lambda: frozenset({"list", "show", "find", "give", "display"}),
        init=False,
    )
    _AVAILABILITY_WORDS: frozenset[str] = field(
        default_factory=lambda: frozenset({"available", "availability", "leaving", "departing", "arriving"}),
        init=False,
    )

    def _extra_routes(self, *, tokens: list[str], canonical_tokens: list[str], ent_types: list[str]) -> list[str]:
        raw_set = {t.lower() for t in tokens}
        routes: list[str] = []
        if raw_set.intersection(self._TRANSPORT_WORDS):
            routes.append("ground_transport")
        if raw_set.intersection(self._AIRFARE_WORDS):
            routes.append("airfare")
        if raw_set.intersection(self._AIRLINE_WORDS):
            routes.append("airline")
        if raw_set.intersection(self._DAY_NAME_WORDS) and {"flight", "flights", "fly"}.intersection(raw_set):
            routes.append("day_name")
        if raw_set.intersection(self._TIME_WORDS) and {"flight", "flights"}.intersection(raw_set):
            routes.append("flight_time")
        return routes

    def _extra_concepts(self, *, tokens: list[str], canonical_tokens: list[str], ent_types: list[str], routes: list[str]) -> list[dict[str, object]]:
        concepts: list[dict[str, object]] = []
        canon_lower = [token.lower() for token in canonical_tokens]
        raw_set = {token.lower() for token in tokens}
        ent_set = set(ent_types)
        if "GPE" in ent_set or "LOC" in ent_set:
            concepts.append({"concept_id": "concept::location_constraint", "concept_kind": "slot_concept", "evidence": ("location",)})
            if "from" in canon_lower:
                concepts.append({"concept_id": "concept::from_city", "concept_kind": "slot_role_concept", "evidence": ("from",)})
            if "to" in canon_lower or "for" in canon_lower:
                concepts.append({"concept_id": "concept::to_city", "concept_kind": "slot_role_concept", "evidence": ("to",)})
        if raw_set.intersection(self._TRANSPORT_WORDS):
            concepts.append({"concept_id": "concept::ground_transport_query", "concept_kind": "route_concept", "evidence": tuple(sorted(raw_set.intersection(self._TRANSPORT_WORDS)))})
        if raw_set.intersection(self._AIRLINE_WORDS):
            concepts.append({"concept_id": "concept::airline_query", "concept_kind": "route_concept", "evidence": tuple(sorted(raw_set.intersection(self._AIRLINE_WORDS)))})
        if raw_set.intersection(self._AIRFARE_WORDS):
            concepts.append({"concept_id": "concept::airfare_query", "concept_kind": "route_concept", "evidence": tuple(sorted(raw_set.intersection(self._AIRFARE_WORDS)))})
        if raw_set.intersection(self._DAY_NAME_WORDS) and {"flight", "flights", "fly"}.intersection(raw_set):
            concepts.append({"concept_id": "concept::day_name_query", "concept_kind": "route_concept", "evidence": tuple(sorted(raw_set.intersection(self._DAY_NAME_WORDS)))})
        if raw_set.intersection(self._TIME_WORDS) and {"flight", "flights"}.intersection(raw_set):
            concepts.append({"concept_id": "concept::flight_time_query", "concept_kind": "route_concept", "evidence": tuple(sorted(raw_set.intersection(self._TIME_WORDS)))})
        return concepts

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        tokens = list(packet.context.get("tokens", []))
        canonical_tokens = list(packet.context.get("canonical_tokens", []))
        ent_types = list(packet.context.get("ent_types", []))
        content_words = list(packet.context.get("content_words", []))
        skeleton = list(packet.context.get("skeleton", []))
        ngrams = list(packet.context.get("skeleton_ngrams", []))

        anchors, concepts, view_anchor_map = build_shared_anchor_payload(
            tokens=tokens,
            canonical_tokens=canonical_tokens,
            ent_types=ent_types,
            content_words=content_words,
            skeleton=skeleton,
            ngrams=ngrams,
            profile=self._PROFILE,
            extra_route_detector=self._extra_routes,
            extra_concept_builder=self._extra_concepts,
        )

        packet.context["atis_bridge_anchors"] = anchors
        packet.context["atis_concepts"] = concepts
        packet.context["atis_view_anchor_map"] = view_anchor_map
        return packet
