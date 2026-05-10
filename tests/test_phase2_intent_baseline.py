from hpm_ai_v5.agents.atis import (
    AnchorIntentStrategy,
    AuditableIntentInferenceAgent,
    FlatIntentStrategy,
    InterconnectedATISInferenceAgent,
)
from hpm_ai_v5.core.config import CoreConfig
from hpm_ai_v5.core import PatternEngine, PatternStoreProjector, PolygraphPatternRetriever
from hpm_ai_v5.pipeline import HPMPipeline
from hpm_ai_v5.adapter.validation_only import ValidationOnlyAdapter
from hpm_ai_v5.polygraphs.base import PolygraphView
from hpm_ai_v5.core.state import State
from hpm_ai_v5.agents.packet import AgentPacket


def test_flat_intent_strategy_emits_vote_evidence():
    config = CoreConfig(exact_threshold=0.1, near_threshold=1.0)
    engine = PatternEngine(config=config)
    p = engine.store.learn((1.0,), name="flight_pattern")
    p.utility = 2.0
    match = engine.store.match((1.0,))

    prediction, votes, evidence = FlatIntentStrategy(
        {"flight_pattern": "flight"}
    ).predict([("primary", match)])

    assert prediction == "flight"
    assert votes["flight"] > 0.0
    assert evidence[0]["source"] == "primary"
    assert evidence[0]["intent"] == "flight"
    assert evidence[0]["contribution"] > 0.0


def test_flat_intent_strategy_defaults_to_no_feedback_bias():
    config = CoreConfig(exact_threshold=0.1, near_threshold=1.0)
    primary_engine = PatternEngine(config=config)
    primary_pattern = primary_engine.store.learn((1.0,), name="flight_pattern")
    primary_pattern.utility = 1.0
    primary_match = primary_engine.store.match((1.0,))

    view_engine = PatternEngine(config=config)
    view_pattern = view_engine.store.learn((2.0,), name="airfare_pattern")
    view_pattern.utility = 1.0
    view_match = view_engine.store.match((2.0,))

    prediction, votes, _ = FlatIntentStrategy(
        {"flight_pattern": "flight", "airfare_pattern": "airfare"},
        feedback_context={
            "intent_feedback_top_source": "primary",
            "intent_feedback_top_source_penalty": 0.1,
            "intent_feedback_route_hint": "airfare",
            "intent_feedback_route_boost": 10.0,
            "kb_error_supported": True,
            "kb_error_score": 1.0,
            "kb_expected_intents": ("airfare",),
        },
    ).predict([("primary", primary_match), ("canonical_view", view_match)])

    assert prediction == "flight"
    assert votes["flight"] == votes["airfare"]


def test_flat_intent_strategy_applies_feedback_reweighting():
    config = CoreConfig(exact_threshold=0.1, near_threshold=1.0)
    primary_engine = PatternEngine(config=config)
    primary_pattern = primary_engine.store.learn((1.0,), name="flight_pattern")
    primary_pattern.utility = 1.0
    primary_match = primary_engine.store.match((1.0,))

    view_engine = PatternEngine(config=config)
    view_pattern = view_engine.store.learn((2.0,), name="airfare_pattern")
    view_pattern.utility = 1.0
    view_match = view_engine.store.match((2.0,))

    prediction, votes, evidence = FlatIntentStrategy(
        {"flight_pattern": "flight", "airfare_pattern": "airfare"},
        feedback_context={
            "intent_feedback_top_source": "primary",
            "intent_feedback_top_source_penalty": 0.5,
            "intent_feedback_route_hint": "airfare",
            "intent_feedback_route_boost": 1.2,
        },
        use_feedback_reweighting=True,
    ).predict([("primary", primary_match), ("canonical_view", view_match)])

    assert prediction == "airfare"
    assert votes["airfare"] > votes["flight"]
    assert any(row["route_aligned"] for row in evidence if row["intent"] == "airfare")


def test_flat_intent_strategy_applies_kb_error_bias():
    config = CoreConfig(exact_threshold=0.1, near_threshold=1.0)
    primary_engine = PatternEngine(config=config)
    primary_pattern = primary_engine.store.learn((1.0,), name="flight_pattern")
    primary_pattern.utility = 1.0
    primary_match = primary_engine.store.match((1.0,))

    view_engine = PatternEngine(config=config)
    view_pattern = view_engine.store.learn((2.0,), name="airfare_pattern")
    view_pattern.utility = 1.0
    view_match = view_engine.store.match((2.0,))

    prediction, votes, evidence = FlatIntentStrategy(
        {"flight_pattern": "flight", "airfare_pattern": "airfare"},
        feedback_context={
            "kb_error_supported": True,
            "kb_error_score": 1.0,
            "kb_expected_intents": ("airfare",),
        },
        use_kb_bias=True,
    ).predict([("primary", primary_match), ("canonical_view", view_match)])

    assert prediction == "airfare"
    assert votes["airfare"] > votes["flight"]
    assert any(row["kb_expected"] for row in evidence if row["intent"] == "airfare")


def test_auditable_intent_inference_agent_writes_vote_trace():
    class StubPreprocessor:
        name = "stub_preprocessor"
        requires = []
        provides = ["state"]

        def run(self, packet):
            packet.context.setdefault("tokens", ["book", "flight"])
            packet.states.append(State(value=(101.0,), context=packet.context))
            return packet

    class StubGenerator:
        def generate(self, raw, *, context=None):
            return [PolygraphView(name="canonical_view", state=State(value=(11.0,), context=context or {}))]

    engine = PatternEngine(config=CoreConfig(exact_threshold=0.1, near_threshold=1.0))
    primary = engine.store.learn((101.0,), name="primary_flight")
    primary.utility = 2.0
    pipeline = HPMPipeline(
        preprocessor=StubPreprocessor(),
        engine=engine,
        postprocessor=ValidationOnlyAdapter(),
        polygraph_generator=StubGenerator(),
        polygraph_confidence_skip=1.1,
    )
    view_engine = PatternEngine(config=CoreConfig(exact_threshold=0.1, near_threshold=1.0))
    view_pattern = view_engine.store.learn((11.0,), name="canon_flight")
    view_pattern.utility = 3.0

    agent = AuditableIntentInferenceAgent(
        view_engines={"canonical_view": view_engine},
        pattern_intent={"primary_flight": "flight", "canon_flight": "flight"},
        pipeline=pipeline,
    )
    packet = AgentPacket(raw="book flight")
    result = agent.step_packet(packet)

    assert result.final_output == "flight"
    assert result.context["intent_votes"]["flight"] > 0.0
    assert len(result.context["intent_vote_trace"]) >= 1


def test_anchor_intent_strategy_uses_dialogue_kb_anchor_bias():
    config = CoreConfig(exact_threshold=0.1, near_threshold=1.0)
    engine = PatternEngine(config=config)
    primary_pattern = engine.store.learn((10.0,), name="primary_flight")
    primary_pattern.utility = 1.0
    view_pattern = engine.store.learn((11.0,), name="view_airfare")
    view_pattern.utility = 1.0

    primary_match = engine.store.match((10.0,))
    views = [
        PolygraphView(
            name="canonical_view",
            state=State(value=(11.0,), context={}),
            anchor_ids=(
                "intent::utterance",
                "dialogue::kb_contradiction",
                "dialogue::kb_expected::airfare",
            ),
        ),
    ]

    prediction, votes, anchor_support, _ = AnchorIntentStrategy(
        {"primary_flight": "flight", "view_airfare": "airfare"},
        feedback_context={
            "kb_error_supported": True,
            "kb_error_score": 1.0,
            "kb_expected_intents": ("airfare",),
        },
        use_dialogue_anchor_scoring=True,
    ).predict(primary_match, views, engine)

    assert prediction == "airfare"
    assert votes["airfare"] > votes["flight"]
    assert anchor_support["dialogue::kb_expected::airfare"]["airfare"] > 0.0


def test_interconnected_inference_does_not_override_without_opt_in():
    class StubPreprocessor:
        name = "stub_preprocessor"
        requires = []
        provides = ["state"]

        def run(self, packet):
            packet.context.setdefault("tokens", ["book", "flight"])
            packet.states.append(State(value=(101.0, 202.0), context=packet.context))
            return packet

    class StubGenerator:
        def generate(self, raw, *, context=None):
            return [
                PolygraphView(
                    name="canonical_view",
                    state=State(value=(11.0,), context=context or {}),
                    anchor_ids=("intent::utterance", "route::flight"),
                ),
            ]

    config = CoreConfig(exact_threshold=0.1, near_threshold=1.0)
    engine = PatternEngine(config=config)
    engine.store.learn((101.0, 202.0), name="primary_flight")
    p1 = engine.store.learn((11.0,), name="canon_flight")
    p1.utility = 2.0
    projector = PatternStoreProjector()
    views = StubGenerator().generate("book flight", context={})
    projector.observe(
        views,
        {"canonical_view": type("M", (), {"pattern": p1})()},
    )

    pipeline = HPMPipeline(
        preprocessor=StubPreprocessor(),
        engine=engine,
        postprocessor=ValidationOnlyAdapter(),
        polygraph_generator=StubGenerator(),
        polygraph_confidence_skip=1.1,
    )
    agent = InterconnectedATISInferenceAgent(
        pattern_intent={"primary_flight": "wrong_intent", "canon_flight": "flight"},
        pipeline=pipeline,
        retriever=PolygraphPatternRetriever(engine.store, projector),
        allow_retriever_override=False,
    )
    packet = AgentPacket(raw="book flight")
    result = agent.step_packet(packet, use_retriever=True)

    assert result.context["retriever_enabled"] is True
    assert result.context["retriever_override_enabled"] is False
    assert result.context["retriever_changed_prediction"] is False


def test_polygraph_retriever_prefers_dialogue_kb_expected_intent():
    config = CoreConfig(exact_threshold=0.1, near_threshold=1.0)
    engine = PatternEngine(config=config)
    flight_pattern = engine.store.learn((1.0,), name="flight_pattern")
    airfare_pattern = engine.store.learn((2.0,), name="airfare_pattern")
    flight_pattern.utility = 1.0
    airfare_pattern.utility = 1.0

    projector = PatternStoreProjector()
    dialogue_view = PolygraphView(
        name="canonical_view",
        state=State(value=(2.0,), context={}),
        anchor_ids=("intent::utterance", "dialogue::kb_expected::airfare"),
        concept_ids=(),
    )
    projector.observe_candidates(dialogue_view, [(flight_pattern, 1.0), (airfare_pattern, 1.0)], pattern_intent={
        "flight_pattern": "flight",
        "airfare_pattern": "airfare",
    })

    retriever = PolygraphPatternRetriever(
        engine.store,
        projector,
        pattern_intent={"flight_pattern": "flight", "airfare_pattern": "airfare"},
        use_dialogue_priors=True,
    )
    candidates = retriever.retrieve([dialogue_view], top_k=2)

    assert candidates[0].pattern_name == "airfare_pattern"
    assert candidates[0].total_score > candidates[1].total_score


def test_polygraph_retriever_injects_expected_intent_patterns_from_projection():
    config = CoreConfig(exact_threshold=0.1, near_threshold=1.0)
    engine = PatternEngine(config=config)
    flight_pattern = engine.store.learn((1.0,), name="flight_pattern")
    airfare_pattern = engine.store.learn((2.0,), name="airfare_pattern")
    flight_pattern.utility = 1.0
    airfare_pattern.utility = 1.0

    projector = PatternStoreProjector()
    training_view = PolygraphView(
        name="skeleton_view",
        state=State(value=(2.0,), context={}),
        anchor_ids=("intent::utterance", "route::airfare"),
        concept_ids=(),
    )
    projector.observe_candidates(training_view, [(airfare_pattern, 2.0)], pattern_intent={
        "airfare_pattern": "airfare",
    })

    current_view = PolygraphView(
        name="canonical_view",
        state=State(value=(1.0,), context={}),
        anchor_ids=("intent::utterance", "dialogue::kb_expected::airfare"),
        concept_ids=(),
    )
    projector.observe_candidates(current_view, [(flight_pattern, 1.0)], pattern_intent={
        "flight_pattern": "flight",
    })

    retriever = PolygraphPatternRetriever(
        engine.store,
        projector,
        pattern_intent={"flight_pattern": "flight", "airfare_pattern": "airfare"},
        use_dialogue_priors=True,
        use_expected_intent_injection=True,
    )
    candidates = retriever.retrieve([current_view], top_k=3)
    candidate_names = [candidate.pattern_name for candidate in candidates]

    assert "airfare_pattern" in candidate_names
