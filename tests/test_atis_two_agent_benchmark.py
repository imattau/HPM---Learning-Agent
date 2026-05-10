from hpm_ai_v5.experiments.run_atis_two_agent_benchmark import TwoAgentATISBenchmark


def test_two_agent_benchmark_builds_interconnected_inference_agent():
    bench = TwoAgentATISBenchmark()
    bench.pattern_intent = {"canon_flight": "flight"}

    agent = bench._build_inference_agent()

    assert agent.name == "atis_interconnected_inference"
    assert agent.retriever is not None
    assert bench.intent_pipeline.polygraph_generator.name == "interconnected_nlp_polygraph"


def test_two_agent_benchmark_predict_routes_through_interconnected_inference():
    bench = TwoAgentATISBenchmark()
    bench.pattern_intent = {"p": "flight"}
    seen = {}

    class StubInferenceAgent:
        name = "atis_interconnected_inference"

        def step_packet(self, packet, *, use_retriever=True, use_concept_scoring=False):
            seen["route"] = packet.context.get("atis_route")
            packet.final_output = "flight"
            return packet

    bench._build_inference_agent = lambda: StubInferenceAgent()

    prediction = bench._predict("show me flights from boston")

    assert prediction == "flight"
    assert seen["route"] == "flight_info"
