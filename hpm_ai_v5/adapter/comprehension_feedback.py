"""Comprehension feedback adapters for intent pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import Adapter
from .packet import AdapterPacket


@dataclass(slots=True)
class ComprehensionFeedbackAdapter(Adapter):
    """Inject previous-turn ambiguity signals back into preprocessing state."""

    name: str = "comprehension_feedback"
    requires: list[str] = field(default_factory=lambda: ["canonical_phraser"])
    provides: list[str] = field(default_factory=lambda: ["canonical_tokens", "feedback"])

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        ctx = packet.context
        canonical_tokens = list(ctx.get("canonical_tokens", ()))
        markers: list[str] = []

        if bool(ctx.get("intent_feedback_conflict", False)):
            markers.append("FB_CONFLICT")
        if float(ctx.get("intent_feedback_margin", 1.0)) < 0.2:
            markers.append("FB_LOW_MARGIN")

        top_source = ctx.get("intent_feedback_top_source")
        if top_source:
            markers.append(f"FB_SOURCE_{str(top_source).upper()}")

        focus_intent = ctx.get("intent_feedback_focus_intent")
        if focus_intent:
            markers.append(f"FB_INTENT_{str(focus_intent).upper()}")

        route_hint = ctx.get("intent_feedback_route_hint")
        if route_hint:
            markers.append(f"FB_ROUTE_{str(route_hint).upper()}")

        if markers:
            ctx["feedback_markers"] = tuple(markers)
            ctx["canonical_tokens"] = canonical_tokens + markers
        return packet


@dataclass(slots=True)
class ComprehensionCarryBridgeAdapter(Adapter):
    """Merge prior-turn comprehension carry into shared bridge anchors/concepts."""

    name: str = "comprehension_carry_bridge"
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=lambda: ["atis_view_anchor_map", "atis_concepts", "dialogue_carry_bridge"])

    @staticmethod
    def _unique(items: list[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item for item in items if item))

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        ctx = packet.context
        view_anchor_map = {
            name: {
                "leaf_keys": tuple(metadata.get("leaf_keys", ())),
                "anchor_ids": tuple(metadata.get("anchor_ids", ())),
                "concept_ids": tuple(metadata.get("concept_ids", ())),
            }
            for name, metadata in dict(ctx.get("atis_view_anchor_map", {})).items()
        }
        if not view_anchor_map:
            return packet

        anchor_ids: list[str] = []
        concept_ids: list[str] = []
        concept_records = {
            str(entry.get("concept_id")): dict(entry)
            for entry in list(ctx.get("atis_concepts", ()))
            if entry.get("concept_id")
        }

        focus_intent = ctx.get("intent_feedback_focus_intent")
        if focus_intent:
            focus_value = str(focus_intent)
            anchor_ids.append(f"dialogue::focus::{focus_value}")
            concept_id = f"concept::dialogue_focus::{focus_value}"
            concept_ids.append(concept_id)
            concept_records.setdefault(concept_id, {
                "concept_id": concept_id,
                "concept_kind": "dialogue_concept",
                "evidence": ("focus_intent",),
            })

        route_hint = ctx.get("intent_feedback_route_hint")
        if route_hint:
            route_value = str(route_hint)
            anchor_ids.append(f"dialogue::route::{route_value}")
            concept_id = f"concept::dialogue_route::{route_value}"
            concept_ids.append(concept_id)
            concept_records.setdefault(concept_id, {
                "concept_id": concept_id,
                "concept_kind": "dialogue_route_concept",
                "evidence": ("route_hint",),
            })

        if bool(ctx.get("kb_error_supported", False)):
            anchor_ids.append("dialogue::kb_contradiction")
            for expected in ctx.get("kb_expected_intents", ()):
                expected_value = str(expected)
                anchor_ids.append(f"dialogue::kb_expected::{expected_value}")
                concept_id = f"concept::dialogue_kb_expected::{expected_value}"
                concept_ids.append(concept_id)
                concept_records.setdefault(concept_id, {
                    "concept_id": concept_id,
                    "concept_kind": "dialogue_kb_concept",
                    "evidence": ("kb_expected_intent",),
                })

        if float(ctx.get("intent_feedback_margin", 1.0)) < 0.2:
            anchor_ids.append("dialogue::low_margin")
        if bool(ctx.get("intent_feedback_conflict", False)):
            anchor_ids.append("dialogue::source_conflict")

        if not anchor_ids and not concept_ids:
            return packet

        merged_anchor_ids = self._unique(anchor_ids)
        merged_concept_ids = self._unique(concept_ids)
        for metadata in view_anchor_map.values():
            metadata["anchor_ids"] = self._unique(list(metadata["anchor_ids"]) + list(merged_anchor_ids))
            metadata["concept_ids"] = self._unique(list(metadata["concept_ids"]) + list(merged_concept_ids))

        ctx["atis_view_anchor_map"] = view_anchor_map
        ctx["atis_concepts"] = [concept_records[key] for key in sorted(concept_records)]
        ctx["dialogue_carry_bridge"] = {
            "anchor_ids": merged_anchor_ids,
            "concept_ids": merged_concept_ids,
        }
        return packet


class ComprehensionFeedbackPostprocessor:
    """Derive carry_context for the next comprehension step from vote traces."""

    name: str = "comprehension_feedback_postprocessor"
    requires: list[str] = []
    provides: list[str] = ["feedback"]
    _ATIS_CONCEPT_INTENTS: dict[str, tuple[str, ...]] = {
        "concept::airfare_query": ("airfare",),
        "concept::airline_query": ("airline",),
        "concept::day_name_query": ("day_name",),
        "concept::flight_time_query": ("flight_time",),
        "concept::ground_transport_query": ("ground_service",),
    }

    @staticmethod
    def _route_hint(text: str, predicted_intent: str | None) -> str | None:
        lowered = text.lower()
        if predicted_intent:
            return predicted_intent
        if " from " in lowered and " to " in lowered:
            return "flight"
        if any(word in lowered for word in ("fare", "airfare", "price", "cost")):
            return "airfare"
        if any(word in lowered for word in ("restaurant", "table", "reserve")):
            return "book_restaurant"
        if any(word in lowered for word in ("movie", "film", "screening", "showing")):
            return "search_screening_event"
        return None

    @classmethod
    def _kb_contradiction(
        cls,
        concepts: list[dict[str, object]],
        predicted_intent: str | None,
    ) -> tuple[bool, float, tuple[str, ...], str | None]:
        expected: list[str] = []
        for concept in concepts:
            concept_id = str(concept.get("concept_id", ""))
            expected.extend(cls._ATIS_CONCEPT_INTENTS.get(concept_id, ()))
        expected_intents = tuple(dict.fromkeys(expected))
        if not expected_intents:
            return False, 0.0, (), None
        if predicted_intent in expected_intents:
            return False, 0.0, expected_intents, None
        if predicted_intent is None:
            return True, 0.7, expected_intents, "kb_missing_supported_intent"
        return True, 1.0, expected_intents, "kb_route_intent_conflict"

    def run(self, packet: AdapterPacket) -> AdapterPacket:
        ctx = packet.context
        votes = dict(ctx.get("intent_votes", {}))
        vote_trace = list(ctx.get("intent_vote_trace", ()))
        predicted_intent = ctx.get("predicted_intent")
        concepts = list(ctx.get("atis_concepts", ()))

        ordered_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        if len(ordered_votes) >= 2:
            margin = float(ordered_votes[0][1] - ordered_votes[1][1])
            focus_intent = ordered_votes[1][0] if margin < 0.2 else ordered_votes[0][0]
        elif len(ordered_votes) == 1:
            margin = float(ordered_votes[0][1])
            focus_intent = ordered_votes[0][0]
        else:
            margin = 0.0
            focus_intent = None

        source_intents: dict[str, str] = {}
        source_scores: dict[str, float] = {}
        for row in vote_trace:
            source = str(row.get("source"))
            source_scores[source] = source_scores.get(source, 0.0) + float(row.get("contribution", 0.0))
            source_intents.setdefault(source, str(row.get("intent")))
        top_source = max(source_scores, key=source_scores.__getitem__) if source_scores else None
        conflict = len({intent for intent in source_intents.values() if intent}) >= 2
        source_penalty = 1.0
        if top_source is not None and margin < 0.2:
            source_penalty = 0.75 if conflict else 0.85
        route_boost = 1.15 if margin < 0.2 else 1.0

        ctx["carry_intent_feedback_margin"] = margin
        ctx["carry_intent_feedback_conflict"] = conflict
        if top_source is not None:
            ctx["carry_intent_feedback_top_source"] = top_source
        if focus_intent is not None:
            ctx["carry_intent_feedback_focus_intent"] = focus_intent
        route_hint = self._route_hint(str(packet.raw), predicted_intent)
        if route_hint is not None:
            ctx["carry_intent_feedback_route_hint"] = route_hint
        ctx["carry_intent_feedback_top_source_penalty"] = source_penalty
        ctx["carry_intent_feedback_route_boost"] = route_boost

        kb_contradiction, kb_error_score, kb_expected_intents, kb_error_reason = self._kb_contradiction(
            concepts,
            predicted_intent,
        )
        ctx["carry_kb_error_supported"] = kb_contradiction
        ctx["carry_kb_error_score"] = kb_error_score
        if kb_expected_intents:
            ctx["carry_kb_expected_intents"] = kb_expected_intents
        if kb_error_reason is not None:
            ctx["carry_kb_error_reason"] = kb_error_reason

        packet.log(
            self.name,
            {
                "margin": margin,
                "conflict": conflict,
                "top_source": top_source,
                "focus_intent": focus_intent,
                "route_hint": route_hint,
                "top_source_penalty": source_penalty,
                "route_boost": route_boost,
                "kb_error_supported": kb_contradiction,
                "kb_error_score": kb_error_score,
                "kb_expected_intents": kb_expected_intents,
                "kb_error_reason": kb_error_reason,
            },
            role="adapter",
        )
        return packet
