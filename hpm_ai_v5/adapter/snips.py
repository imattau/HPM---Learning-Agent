"""SNIPS dataset loader and bridge-aware adapters for HPM v5."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

from .base import Adapter
from .intent_concepts import IntentProfile, build_shared_anchor_payload
from .packet import AdapterPacket


def load_snips() -> tuple[list[dict], list[dict]]:
    """Return (train, test) as lists of {text, intent} dicts from local CSVs."""

    def _read_csv(file_path: str) -> list[dict]:
        data: list[dict] = []
        if not os.path.exists(file_path):
            return data
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({"text": row["text"], "intent": row["intent"]})
        return data

    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train = _read_csv(os.path.join(_repo, "data", "snips", "snips_train.csv"))
    test = _read_csv(os.path.join(_repo, "data", "snips", "snips_test.csv"))
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
class SNIPSCanonicalIntentAdapter(Adapter):
    """Dataset-scoped canonical enrichment for SNIPS intent families."""

    name: str = "snips_canonical_intent"
    requires: list[str] = field(default_factory=lambda: ["canonical_phraser"])
    provides: list[str] = field(default_factory=lambda: ["canonical_tokens"])

    _MAP: dict[str, str] = field(default_factory=lambda: {
        "screening": "SCREENING",
        "showing": "SCREENING",
        "showings": "SCREENING",
        "theater": "SCREENING",
        "theatre": "SCREENING",
        "cinema": "SCREENING",
        "movie": "SCREENING",
        "movies": "SCREENING",
        "film": "SCREENING",
        "films": "SCREENING",
        "restaurant": "RESTAURANT",
        "restaurants": "RESTAURANT",
        "table": "RESTAURANT",
        "tables": "RESTAURANT",
        "dining": "RESTAURANT",
        "playlist": "MUSIC",
        "playlists": "MUSIC",
        "music": "MUSIC",
        "song": "MUSIC",
        "songs": "MUSIC",
        "album": "MUSIC",
        "albums": "MUSIC",
        "artist": "MUSIC",
        "artists": "MUSIC",
        "jazz": "MUSIC",
        "pop": "MUSIC",
        "play": "PLAY",
        "start": "PLAY",
        "listen": "PLAY",
        "novel": "CREATIVE_WORK",
        "book": "CREATIVE_WORK",
        "books": "CREATIVE_WORK",
        "work": "CREATIVE_WORK",
        "works": "CREATIVE_WORK",
        "author": "CREATIVE_WORK",
        "pixar": "CREATIVE_WORK",
    }, init=False)

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        lemmas = list(packet.context.get("lemmas", ()))
        canonical_tokens = list(packet.context.get("canonical_tokens", ()))
        if not canonical_tokens or not lemmas:
            return packet
        adjusted = []
        for index, token in enumerate(canonical_tokens):
            lemma = lemmas[index] if index < len(lemmas) else ""
            adjusted.append(self._MAP.get(lemma, token))
        packet.context["canonical_tokens"] = adjusted
        return packet


@dataclass(slots=True)
class SNIPSBridgeAnchorAdapter(Adapter):
    """Emit bridge anchors and concepts for SNIPS intent utterances."""

    name: str = "snips_bridge_anchor"
    requires: list[str] = field(default_factory=lambda: [
        "ner_canonicaliser",
        "content_word_extractor",
        "skeleton_extractor",
        "skeleton_ngram",
    ])
    provides: list[str] = field(default_factory=lambda: ["atis_bridge_anchors", "atis_view_anchor_map", "atis_concepts"])

    _PROFILE: IntentProfile = field(default_factory=lambda: IntentProfile(
        route_keywords={
            "add_to_playlist": frozenset({"PLAYLIST", "TRACK", "SONG", "ALBUM"}),
            "book_restaurant": frozenset({"RESTAURANT", "TABLE", "RESERVE", "BOOK"}),
            "get_weather": frozenset({"WEATHER", "FORECAST", "TEMPERATURE", "RAIN", "SUNNY"}),
            "play_music": frozenset({"MUSIC", "PLAY", "SONG", "ARTIST", "ALBUM"}),
            "rate_book": frozenset({"RATE", "BOOK", "NOVEL", "AUDIOBOOK", "STARS"}),
            "search_creative_work": frozenset({"SEARCH", "FIND", "BOOK", "MOVIE", "WORK"}),
            "search_screening_event": frozenset({"SCREENING", "MOVIE", "FILM", "THEATER", "SHOWING"}),
        },
        route_concepts={
            "add_to_playlist": "concept::playlist_query",
            "book_restaurant": "concept::restaurant_query",
            "get_weather": "concept::weather_query",
            "play_music": "concept::music_query",
            "rate_book": "concept::rating_query",
            "search_creative_work": "concept::creative_work_query",
            "search_screening_event": "concept::screening_query",
        },
        query_terms={
            "concept::listing_request": frozenset({"show", "find", "search", "list"}),
            "concept::availability_request": frozenset({"available", "availability", "open"}),
        },
        entity_concepts={
            "TIME": "concept::time_constraint",
            "DATE": "concept::date_constraint",
            "GPE": "concept::location_constraint",
            "LOC": "concept::location_constraint",
            "ORG": "concept::entity_constraint",
        },
        broad_concepts=frozenset({"concept::listing_request", "concept::availability_request", "concept::entity_constraint"}),
    ), init=False)

    def _extra_routes(self, *, tokens: list[str], canonical_tokens: list[str], ent_types: list[str]) -> list[str]:
        token_set = {token.lower() for token in tokens}
        canon_set = set(canonical_tokens)
        routes: list[str] = []
        if {"playlist", "song", "songs", "track"}.intersection(token_set) and {"add", "put", "save"}.intersection(token_set):
            routes.append("add_to_playlist")
        if {"restaurant", "table", "sushi", "italian", "mexican"}.intersection(token_set) and {"book", "reserve", "find"}.intersection(token_set):
            routes.append("book_restaurant")
        if {"weather", "forecast", "rain", "sunny", "temperature"}.intersection(token_set):
            routes.append("get_weather")
        if {"play", "songs", "music", "album", "artist", "playlist", "jazz", "pop"}.intersection(token_set):
            routes.append("play_music")
        if {"screening", "showing", "movie", "movies", "film", "theater", "cinema"}.intersection(token_set):
            routes.append("search_screening_event")
        if {"book", "books", "novel", "work", "author", "pixar"}.intersection(token_set) and {"find", "search", "look", "lookup"}.intersection(token_set):
            routes.append("search_creative_work")
        if {"rate", "rating", "stars", "star"}.intersection(token_set):
            routes.append("rate_book")
        if "SCREENING" in canon_set:
            routes.append("search_screening_event")
        if "RESTAURANT" in canon_set:
            routes.append("book_restaurant")
        if "MUSIC" in canon_set and "PLAY" in canon_set:
            routes.append("play_music")
        return routes

    def _extra_concepts(self, *, tokens: list[str], canonical_tokens: list[str], ent_types: list[str], routes: list[str]) -> list[dict[str, object]]:
        token_set = {token.lower() for token in tokens}
        canon_set = set(canonical_tokens)
        concepts: list[dict[str, object]] = []
        if any(token in token_set for token in {"today", "tomorrow", "tonight", "morning", "afternoon"}):
            concepts.append({"concept_id": "concept::time_constraint", "concept_kind": "slot_concept", "evidence": ("time",)})
        if any(token in token_set for token in {"near", "downtown", "nearby"}):
            concepts.append({"concept_id": "concept::location_constraint", "concept_kind": "slot_concept", "evidence": ("location",)})
        if {"screening", "showing", "movie", "movies", "film", "theater", "cinema"}.intersection(token_set) or "SCREENING" in canon_set:
            concepts.append({"concept_id": "concept::screening_query", "concept_kind": "route_concept", "evidence": ("screening",)})
        if {"restaurant", "table", "sushi", "italian", "mexican"}.intersection(token_set) or "RESTAURANT" in canon_set:
            concepts.append({"concept_id": "concept::restaurant_query", "concept_kind": "route_concept", "evidence": ("restaurant",)})
        if {"weather", "forecast", "rain", "sunny", "temperature"}.intersection(token_set):
            concepts.append({"concept_id": "concept::weather_query", "concept_kind": "route_concept", "evidence": ("weather",)})
        if {"play", "songs", "music", "album", "artist", "playlist"}.intersection(token_set) or {"MUSIC", "PLAY"}.issubset(canon_set):
            concepts.append({"concept_id": "concept::music_query", "concept_kind": "route_concept", "evidence": ("music",)})
        if {"book", "books", "novel", "work", "author", "pixar"}.intersection(token_set) or "CREATIVE_WORK" in canon_set:
            concepts.append({"concept_id": "concept::creative_work_query", "concept_kind": "route_concept", "evidence": ("creative_work",)})
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
