from hpm_ai_v5.agents.atis import AuditableIntentInferenceAgent
from hpm_ai_v5.adapter.comprehension_feedback import (
    ComprehensionCarryBridgeAdapter,
    ComprehensionFeedbackAdapter,
    ComprehensionFeedbackPostprocessor,
)
from hpm_ai_v5.core import PatternEngine
from hpm_ai_v5.core.config import CoreConfig
from hpm_ai_v5.pipeline import HPMPipeline
from hpm_ai_v5.adapter.validation_only import ValidationOnlyAdapter
from hpm_ai_v5.polygraphs.base import PolygraphView
from hpm_ai_v5.core.state import State
from hpm_ai_v5.agents.packet import AgentPacket
from hpm_ai_v5.adapter.packet import AdapterPacket


def test_auditable_intent_agent_emits_feedback_carry():
    class StubPreprocessor:
        name = "stub_preprocessor"
        requires = []
        provides = ["state"]

        def run(self, packet):
            packet.context.setdefault("tokens", ["book", "flight"])
            packet.context.setdefault("canonical_tokens", ["BOOK", "FLIGHT"])
            packet.states.append(State(value=(101.0,), context=packet.context))
            return packet

    class StubGenerator:
        def generate(self, raw, *, context=None):
            return [PolygraphView(name="canonical_view", state=State(value=(11.0,), context=context or {}))]

    engine = PatternEngine(config=CoreConfig(exact_threshold=0.1, near_threshold=1.0))
    primary = engine.store.learn((101.0,), name="primary_flight")
    primary.utility = 1.0
    pipeline = HPMPipeline(
        preprocessor=StubPreprocessor(),
        engine=engine,
        postprocessor=ValidationOnlyAdapter(),
        polygraph_generator=StubGenerator(),
        polygraph_confidence_skip=1.1,
    )
    pipeline.register_preprocessor(ComprehensionFeedbackAdapter())
    view_engine = PatternEngine(config=CoreConfig(exact_threshold=0.1, near_threshold=1.0))
    view_pattern = view_engine.store.learn((11.0,), name="view_airfare")
    view_pattern.utility = 1.05

    agent = AuditableIntentInferenceAgent(
        view_engines={"canonical_view": view_engine},
        pattern_intent={"primary_flight": "flight", "view_airfare": "airfare"},
        pipeline=pipeline,
    )
    packet = AgentPacket(raw="book flight")
    result = agent.step_packet(packet)

    assert result.context["intent_feedback_margin"] < 0.2
    assert "intent_feedback_top_source" in result.context
    assert "intent_feedback_focus_intent" in result.context
    assert agent.carry_context["intent_feedback_margin"] < 0.2
    assert agent.carry_context["intent_feedback_top_source_penalty"] < 1.0
    assert agent.carry_context["intent_feedback_route_boost"] >= 1.0


def test_feedback_adapter_injects_markers_on_next_turn():
    class StubPreprocessor:
        name = "stub_preprocessor"
        requires = []
        provides = ["state"]

        def run(self, packet):
            packet.context.setdefault("tokens", ["book", "flight"])
            packet.context.setdefault("canonical_tokens", ["BOOK", "FLIGHT"])
            packet.states.append(State(value=(101.0,), context=packet.context))
            return packet

    class StubGenerator:
        def generate(self, raw, *, context=None):
            return [PolygraphView(name="canonical_view", state=State(value=(11.0,), context=context or {}))]

    engine = PatternEngine(config=CoreConfig(exact_threshold=0.1, near_threshold=1.0))
    primary = engine.store.learn((101.0,), name="primary_flight")
    primary.utility = 1.0
    pipeline = HPMPipeline(
        preprocessor=StubPreprocessor(),
        engine=engine,
        postprocessor=ValidationOnlyAdapter(),
        polygraph_generator=StubGenerator(),
        polygraph_confidence_skip=1.1,
    )
    pipeline.register_preprocessor(ComprehensionFeedbackAdapter())
    view_engine = PatternEngine(config=CoreConfig(exact_threshold=0.1, near_threshold=1.0))
    view_pattern = view_engine.store.learn((11.0,), name="view_airfare")
    view_pattern.utility = 1.05

    agent = AuditableIntentInferenceAgent(
        view_engines={"canonical_view": view_engine},
        pattern_intent={"primary_flight": "flight", "view_airfare": "airfare"},
        pipeline=pipeline,
    )
    agent.step_packet(AgentPacket(raw="book flight"))
    next_packet = AgentPacket(raw="book flight")
    result = agent.step_packet(next_packet)

    assert "feedback_markers" in result.context
    assert any(marker.startswith("FB_") for marker in result.context["feedback_markers"])


def test_feedback_postprocessor_marks_kb_contradiction_for_atis_route_concept():
    postprocessor = ComprehensionFeedbackPostprocessor()
    packet = AdapterPacket(
        raw="what is the airfare from denver to pittsburgh",
        context={
            "predicted_intent": "flight",
            "intent_votes": {"flight": 1.0, "airfare": 0.9},
            "intent_vote_trace": [
                {"source": "primary", "intent": "flight", "contribution": 1.0},
                {"source": "content_view", "intent": "airfare", "contribution": 0.9},
            ],
            "atis_concepts": [
                {"concept_id": "concept::airfare_query", "concept_kind": "route_concept"},
                {"concept_id": "concept::from_city", "concept_kind": "slot_role_concept"},
            ],
        },
        states=[],
    )

    result = postprocessor.run(packet)

    assert result.context["carry_kb_error_supported"] is True
    assert result.context["carry_kb_error_score"] == 1.0
    assert result.context["carry_kb_expected_intents"] == ("airfare",)
    assert result.context["carry_kb_error_reason"] == "kb_route_intent_conflict"


def test_carry_bridge_adapter_merges_dialogue_anchors_into_view_map():
    adapter = ComprehensionCarryBridgeAdapter()
    packet = AdapterPacket(
        raw="what is the airfare",
        context={
            "atis_view_anchor_map": {
                "canonical_view": {
                    "leaf_keys": ("canon:what", "canon:AIRFARE"),
                    "anchor_ids": ("intent::utterance", "route::airfare"),
                    "concept_ids": ("concept::airfare_query",),
                },
            },
            "atis_concepts": [
                {"concept_id": "concept::airfare_query", "concept_kind": "route_concept", "evidence": ("airfare",)},
            ],
            "intent_feedback_focus_intent": "airfare",
            "intent_feedback_route_hint": "airfare",
            "intent_feedback_margin": 0.1,
            "intent_feedback_conflict": True,
            "kb_error_supported": True,
            "kb_expected_intents": ("airfare",),
        },
        states=[],
    )

    result = adapter.run(packet)
    metadata = result.context["atis_view_anchor_map"]["canonical_view"]

    assert "dialogue::focus::airfare" in metadata["anchor_ids"]
    assert "dialogue::route::airfare" in metadata["anchor_ids"]
    assert "dialogue::kb_contradiction" in metadata["anchor_ids"]
    assert "dialogue::kb_expected::airfare" in metadata["anchor_ids"]
    assert "dialogue::low_margin" in metadata["anchor_ids"]
    assert "dialogue::source_conflict" in metadata["anchor_ids"]
    assert "concept::dialogue_route::airfare" in metadata["concept_ids"]
    assert "concept::dialogue_kb_expected::airfare" in metadata["concept_ids"]
    assert result.context["dialogue_carry_bridge"]["anchor_ids"]
