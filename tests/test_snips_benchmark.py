def test_snips_loads_splits():
    from hpm_ai_v5.adapter.snips import load_snips

    train, test = load_snips()
    assert len(train) >= 20
    assert len(test) >= 10
    assert "text" in train[0]
    assert "intent" in train[0]


def test_snips_bridge_adapter_emits_concepts():
    from hpm_ai_v5.adapter.packet import AdapterPacket
    from hpm_ai_v5.adapter.snips import SNIPSBridgeAnchorAdapter

    adapter = SNIPSBridgeAnchorAdapter()
    packet = AdapterPacket(
        raw="what is the weather in london tomorrow",
        context={
            "tokens": ["what", "is", "the", "weather", "in", "GPE", "DATE"],
            "canonical_tokens": ["what", "is", "the", "WEATHER", "in", "GPE", "DATE"],
            "ent_types": ["", "", "", "", "", "GPE", "DATE"],
            "content_words": ["weather"],
            "skeleton": ["R", "V", "D", "N", "R", "N", "N"],
            "skeleton_ngrams": ["R_V", "V_D"],
        },
        states=[],
    )
    result = adapter.run(packet)
    concepts = result.context["atis_concepts"]
    view_map = result.context["atis_view_anchor_map"]
    assert any(concept["concept_id"] == "concept::weather_query" for concept in concepts)
    assert "concept::weather_query" in view_map["canonical_view"]["concept_ids"]
    assert "route::get_weather" in view_map["canonical_view"]["anchor_ids"]


def test_interconnected_snips_benchmark_store_map_shape():
    from hpm_ai_v5.experiments.run_snips_interconnected_benchmark import InterconnectedSNIPSBenchmark

    bench = InterconnectedSNIPSBenchmark()
    bench.store_map = None
    summary = bench.run_store_map()
    assert "concept_hubs" in summary
    assert "avg_patterns_per_concept" in summary


def test_snips_retriever_comparison_reports_concept_metrics():
    from hpm_ai_v5.experiments.run_snips_interconnected_benchmark import InterconnectedSNIPSBenchmark

    bench = InterconnectedSNIPSBenchmark()
    responses = {
        ("u1", False, False): ("wrong", {}),
        ("u1", True, False): ("music", {"retriever_changed_prediction": True}),
        ("u1", True, True): ("music", {"retriever_changed_prediction": True}),
        ("u2", False, False): ("weather", {}),
        ("u2", True, False): ("weather", {"retriever_changed_prediction": False}),
        ("u2", True, True): ("weather", {"retriever_changed_prediction": False}),
    }

    def _predict(text: str, *, use_retriever: bool = True, use_concept_scoring: bool = False):
        return responses[(text, use_retriever, use_concept_scoring)]

    bench._predict = _predict
    metrics = bench.run_b_retriever_comparison([
        {"text": "u1", "intent": "music"},
        {"text": "u2", "intent": "weather"},
    ])
    assert metrics["anchor_plus_concept_accuracy"] == 1.0
    assert "concept_changed_prediction_rate" in metrics
    assert "concept_improved_rate" in metrics
    assert "concept_degraded_rate" in metrics
    assert "concept_supported_candidate_count" in metrics
    assert "avg_concept_agreements_per_candidate" in metrics
    assert "concept_family_breakdown" in metrics


def test_snips_benchmark_baseline_smoke():
    from hpm_ai_v5.experiments.run_snips_benchmark import SNIPSBenchmark

    bench = SNIPSBenchmark()
    train = [
        {"text": "play jazz music", "intent": "play_music"},
        {"text": "book a restaurant table", "intent": "book_restaurant"},
    ]
    test = [
        {"text": "play songs now", "intent": "play_music"},
        {"text": "reserve a restaurant", "intent": "book_restaurant"},
    ]
    acc = bench.run_b1(train, test)
    assert 0.0 <= acc <= 1.0
