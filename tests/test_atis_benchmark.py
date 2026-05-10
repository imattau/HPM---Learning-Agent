# tests/test_atis_benchmark.py
def test_atis_loads_train_split():
    from hpm_ai_v5.adapter.atis import load_atis
    train, test = load_atis()
    assert len(train) > 1000
    assert len(test) > 100
    assert "text" in train[0]
    assert "intent" in train[0]

def test_intent_label_adapter_injects_label():
    from hpm_ai_v5.adapter.atis import IntentLabelAdapter
    from hpm_ai_v5.adapter.packet import AdapterPacket
    adapter = IntentLabelAdapter(label="flight")
    packet = AdapterPacket(raw="book a flight", context={}, states=[])
    result = adapter.run(packet)
    assert result.context.get("intent_label") == "flight"

def test_intent_label_adapter_withheld_in_inference():
    from hpm_ai_v5.adapter.atis import IntentLabelAdapter
    from hpm_ai_v5.adapter.packet import AdapterPacket
    adapter = IntentLabelAdapter(label=None)
    packet = AdapterPacket(raw="book a flight", context={}, states=[])
    result = adapter.run(packet)
    assert "intent_label" not in result.context


def test_atis_bridge_anchor_adapter_emits_view_metadata():
    from hpm_ai_v5.adapter.atis import ATISBridgeAnchorAdapter
    from hpm_ai_v5.adapter.packet import AdapterPacket

    adapter = ATISBridgeAnchorAdapter()
    packet = AdapterPacket(
        raw="flight from boston to denver tomorrow",
        context={
            "tokens": ["flight", "from", "GPE", "to", "GPE", "DATE"],
            "canonical_tokens": ["FLIGHT", "from", "GPE", "to", "GPE", "DATE"],
            "ent_types": ["", "", "GPE", "", "GPE", "DATE"],
            "content_words": ["flight"],
            "skeleton": ["N", "R", "N", "R", "N", "N"],
            "skeleton_ngrams": ["N_R", "R_N"],
        },
        states=[],
    )
    result = adapter.run(packet)
    anchors = result.context["atis_bridge_anchors"]
    view_map = result.context["atis_view_anchor_map"]
    concepts = result.context["atis_concepts"]
    assert any(anchor["anchor_id"] == "intent::utterance" for anchor in anchors)
    assert "canonical_view" in view_map
    assert "route::flight" in view_map["canonical_view"]["anchor_ids"]
    assert "concept::flight_query" in view_map["canonical_view"]["concept_ids"]
    assert any(concept["concept_id"] == "concept::date_constraint" for concept in concepts)
    assert any(anchor_id.startswith("slot::") for anchor_id in view_map["token_view"]["anchor_ids"])


def test_interconnected_nlp_polygraph_generator_attaches_anchor_metadata():
    from hpm_ai_v5.polygraphs.nlp import InterconnectedNLPPolygraphGenerator

    generator = InterconnectedNLPPolygraphGenerator()
    views = generator.generate(
        "book a flight",
        context={
            "tokens": ["book", "flight"],
            "canonical_tokens": ["BOOK", "FLIGHT"],
            "content_vector": (0.1, 0.2, 0.3),
            "skeleton": ["V", "N"],
            "skeleton_ngrams": ["V_N"],
            "atis_view_anchor_map": {
                "canonical_view": {
                    "leaf_keys": ("canon:BOOK", "canon:FLIGHT"),
                    "anchor_ids": ("intent::utterance", "route::flight"),
                    "concept_ids": ("concept::flight_query", "concept::listing_request"),
                }
            },
        },
    )
    canonical = next(view for view in views if view.name == "canonical_view")
    assert canonical.leaf_keys == ("canon:BOOK", "canon:FLIGHT")
    assert canonical.anchor_ids == ("intent::utterance", "route::flight")
    assert canonical.concept_ids == ("concept::flight_query", "concept::listing_request")


def test_interconnected_atis_inference_agent_prefers_multi_view_anchor_support():
    from hpm_ai_v5.agents.atis import InterconnectedATISInferenceAgent
    from hpm_ai_v5.core import PatternEngine, PatternStoreProjector, PolygraphPatternRetriever
    from hpm_ai_v5.core.config import CoreConfig
    from hpm_ai_v5.pipeline import HPMPipeline
    from hpm_ai_v5.adapter.validation_only import ValidationOnlyAdapter
    from hpm_ai_v5.polygraphs.base import PolygraphView
    from hpm_ai_v5.core.state import State
    from hpm_ai_v5.agents.packet import AgentPacket

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
                    leaf_keys=("canon:FLIGHT",),
                    anchor_ids=("intent::utterance", "route::flight"),
                ),
                PolygraphView(
                    name="skeleton_view",
                    state=State(value=(22.0,), context=context or {}),
                    leaf_keys=("sk:V", "sk:N"),
                    anchor_ids=("intent::utterance", "route::flight"),
                ),
            ]

    config = CoreConfig(exact_threshold=0.1, near_threshold=1.0)
    engine = PatternEngine(config=config)
    engine.store.learn((101.0, 202.0), name="primary_flight")
    p1 = engine.store.learn((11.0,), name="canon_flight")
    p1.utility = 2.0
    p2 = engine.store.learn((22.0,), name="skeleton_flight")
    p2.utility = 2.0
    projector = PatternStoreProjector()
    views = StubGenerator().generate("book flight", context={})
    projector.observe(
        views,
        {
            "canonical_view": type("M", (), {"pattern": p1})(),
            "skeleton_view": type("M", (), {"pattern": p2})(),
        },
    )

    pipeline = HPMPipeline(
        preprocessor=StubPreprocessor(),
        engine=engine,
        postprocessor=ValidationOnlyAdapter(),
        polygraph_generator=StubGenerator(),
        polygraph_confidence_skip=1.1,
    )
    agent = InterconnectedATISInferenceAgent(
        pattern_intent={
            "primary_flight": "flight",
            "canon_flight": "flight",
            "skeleton_flight": "flight",
        },
        pipeline=pipeline,
        retriever=PolygraphPatternRetriever(engine.store, projector),
    )
    packet = AgentPacket(raw="book flight")
    result = agent.step_packet(packet)
    assert result.final_output == "flight"
    assert result.context["multi_view_anchor_hit"] is True
    assert "route::flight" in result.context["anchor_support"]
    assert result.context["polygraph_candidates"][0]["pattern_name"] in {"canon_flight", "skeleton_flight"}
    assert result.context["retriever_enabled"] is True

    anchor_only_packet = AgentPacket(raw="book flight")
    anchor_only_result = agent.step_packet(anchor_only_packet, use_retriever=False)
    assert anchor_only_result.context["retriever_enabled"] is False
    assert anchor_only_result.context["polygraph_candidates"] == []
    assert anchor_only_result.context["anchor_only_predicted_intent"] == anchor_only_result.final_output


def test_pattern_store_projector_builds_map_and_retriever_ranks_candidates():
    from hpm_ai_v5.core import PatternStore, PatternStoreProjector, PolygraphPatternRetriever
    from hpm_ai_v5.core.pattern import Pattern
    from hpm_ai_v5.polygraphs.base import PolygraphView
    from hpm_ai_v5.core.state import State

    store = PatternStore()
    p1 = store.add(Pattern(name="canon_flight", template=(11.0,), utility=1.0))
    p2 = store.add(Pattern(name="skeleton_flight", template=(22.0,), utility=1.0))
    projector = PatternStoreProjector()
    views = [
        PolygraphView(
            name="canonical_view",
            state=State(value=(11.0,), context={}),
            anchor_ids=("intent::utterance", "route::flight"),
            leaf_keys=("canon:FLIGHT",),
            concept_ids=("concept::flight_query", "concept::listing_request"),
        ),
        PolygraphView(
            name="skeleton_view",
            state=State(value=(22.0,), context={}),
            anchor_ids=("intent::utterance", "route::flight"),
            leaf_keys=("sk:V", "sk:N"),
            concept_ids=("concept::flight_query",),
        ),
    ]
    projector.observe_candidates(
        views[0],
        [(p1, 1.0), (p2, 0.4)],
        pattern_intent={"canon_flight": "flight", "skeleton_flight": "flight"},
    )
    projector.observe_candidates(
        views[1],
        [(p2, 1.0), (p1, 0.4)],
        pattern_intent={"canon_flight": "flight", "skeleton_flight": "flight"},
    )
    store_map = projector.build_map()
    assert "route::flight" in store_map.bridge_hubs
    assert "concept::flight_query" in store_map.concept_hubs
    assert len(store_map.projected_patterns) == 2
    assert store_map.avg_patterns_per_anchor >= 2.0
    assert store_map.avg_patterns_per_concept >= 2.0
    assert "concept::listing_request" in projector.projected_patterns["canon_flight"].concept_ids
    assert projector.projected_patterns["canon_flight"].total_support > 1.0

    retriever = PolygraphPatternRetriever(store, projector)
    candidates = retriever.retrieve(views, top_k=2)
    assert len(candidates) == 2
    assert candidates[0].agreement_count >= 1
    assert candidates[0].total_score >= candidates[1].total_score


def test_top_k_projection_candidates_filters_weak_non_winners():
    from hpm_ai_v5.experiments.atis_shared import top_k_projection_candidates
    from hpm_ai_v5.core import PatternEngine
    from hpm_ai_v5.core.config import CoreConfig
    from hpm_ai_v5.polygraphs.base import PolygraphView
    from hpm_ai_v5.core.state import State

    config = CoreConfig(exact_threshold=0.1, near_threshold=1.0)
    engine = PatternEngine(config=config)
    p_exact = engine.store.learn((0.0,), name="exact")
    p_near = engine.store.learn((0.2,), name="near")
    engine.store.learn((0.95,), name="weak_near")
    engine.store.learn((2.0,), name="far")

    view = PolygraphView(name="canonical_view", state=State(value=(0.0,), context={}))
    candidates = top_k_projection_candidates(engine, view, top_k=4)

    assert [pattern.name for pattern, _ in candidates] == ["exact", "near"]
    assert candidates[0][1] == 1.0
    assert 0.0 < candidates[1][1] < 0.2
    assert all(pattern.name not in {"weak_near", "far"} for pattern, _ in candidates)


def test_interconnected_benchmark_retriever_comparison_metrics():
    from hpm_ai_v5.experiments.run_atis_interconnected_benchmark import InterconnectedATISBenchmark

    bench = InterconnectedATISBenchmark()
    responses = {
        ("u1", False): ("wrong", {}),
        ("u1", True): ("flight", {"retriever_changed_prediction": True}),
        ("u2", False): ("airfare", {}),
        ("u2", True): ("airfare", {"retriever_changed_prediction": False}),
        ("u3", False): ("flight", {}),
        ("u3", True): ("ground_transport", {"retriever_changed_prediction": True}),
    }

    def _predict(text: str, *, use_retriever: bool = True):
        pred, extra = responses[(text, use_retriever)]
        ctx = {"predicted_intent": pred, **extra}
        return pred, ctx

    bench._predict = _predict
    metrics = bench.run_b_retriever_comparison([
        {"text": "u1", "intent": "flight"},
        {"text": "u2", "intent": "airfare"},
        {"text": "u3", "intent": "flight"},
    ])
    assert metrics["anchor_only_accuracy"] == 2 / 3
    assert metrics["anchor_plus_retriever_accuracy"] == 2 / 3
    assert metrics["changed_prediction_rate"] == 2 / 3
    assert metrics["improved_rate"] == 1 / 3
    assert metrics["degraded_rate"] == 1 / 3
