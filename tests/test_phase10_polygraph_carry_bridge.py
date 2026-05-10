from hpm_ai_v5.adapter.atis import IntentLabelAdapter, ATISBridgeAnchorAdapter
from hpm_ai_v5.adapter.packet import AdapterPacket
from hpm_ai_v5.core import PatternEngine
from hpm_ai_v5.core.config import CoreConfig
from hpm_ai_v5.experiments.intent_shared import build_intent_pipeline


def test_interconnected_pipeline_emits_dialogue_carry_anchors_on_views():
    engine = PatternEngine(config=CoreConfig(exact_threshold=0.1, near_threshold=1.0))
    pipeline, _ = build_intent_pipeline(
        engine,
        label_adapter=IntentLabelAdapter(),
        bridge_adapter=ATISBridgeAnchorAdapter(),
        interconnected=True,
        include_bridge_anchors=True,
    )

    packet = pipeline.preprocessing_pipeline.run(
        AdapterPacket(
            raw="what is the airfare from denver to pittsburgh",
            context={
                "intent_feedback_focus_intent": "airfare",
                "intent_feedback_route_hint": "airfare",
                "intent_feedback_margin": 0.1,
                "kb_error_supported": True,
                "kb_expected_intents": ("airfare",),
            },
            states=[],
        ),
        target_outputs=list(pipeline.preprocessing_pipeline.adapters.keys()),
    )
    views = pipeline.polygraph_generator.generate(packet.raw, context=packet.context)
    canonical_view = next(view for view in views if view.name == "canonical_view")

    assert "dialogue::focus::airfare" in canonical_view.anchor_ids
    assert "dialogue::route::airfare" in canonical_view.anchor_ids
    assert "dialogue::kb_contradiction" in canonical_view.anchor_ids
    assert "dialogue::kb_expected::airfare" in canonical_view.anchor_ids
    assert "concept::dialogue_route::airfare" in canonical_view.concept_ids
