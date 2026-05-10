from collections import defaultdict

from hpm_ai_v5.experiments.intent_shared import IntentBenchmarkSupport
from hpm_ai_v5.core import PatternEngine, PatternManager
from hpm_ai_v5.core.config import CoreConfig
from hpm_ai_v5.pipeline import HPMPipeline
from hpm_ai_v5.adapter.validation_only import ValidationOnlyAdapter
from hpm_ai_v5.core.state import State


def test_intent_support_records_weighted_votes_for_primary_and_views():
    class StubLabelAdapter:
        label = None

    class StubPreprocessor:
        name = "stub_preprocessor"
        requires = []
        provides = ["state"]

        def run(self, packet):
            packet.states.append(State(value=(1.0,), context=packet.context))
            return packet

    engine = PatternEngine(config=CoreConfig(exact_threshold=0.1, near_threshold=1.0))
    primary = engine.store.learn((1.0,), name="primary_pattern")
    primary.utility = 2.0
    pipeline = HPMPipeline(
        preprocessor=StubPreprocessor(),
        engine=engine,
        postprocessor=ValidationOnlyAdapter(),
    )
    support = IntentBenchmarkSupport(
        engine=engine,
        manager=PatternManager(),
        pipeline=pipeline,
        label_adapter=StubLabelAdapter(),
    )
    support._intent_votes = defaultdict(lambda: defaultdict(float))

    primary_match = engine.store.match((1.0,))
    view_pattern = engine.store.learn((2.0,), name="view_pattern")
    view_pattern.utility = 3.0
    view_match = engine.store.match((2.0,))
    engine.last_match = primary_match
    pipeline.view_matches = {"content_view": view_match}

    support.record_votes("flight")

    assert support._intent_votes["primary_pattern"]["flight"] == 2.0
    assert support._intent_votes["view_pattern"]["flight"] == 3.0


def test_content_view_calibration_updates_thresholds():
    class StubLabelAdapter:
        label = None

    class StubTokenizer:
        name = "nlp_tokenizer"
        requires = []
        provides = ["tokens"]

        def run(self, packet):
            packet.context["tokens"] = [packet.raw]
            return packet

    class StubContentAdapter:
        name = "content_word_extractor"
        requires = ["nlp_tokenizer"]
        provides = ["content_vector"]

        def run(self, packet):
            mapping = {
                "flight_a": (0.0, 0.0),
                "flight_b": (0.02, 0.01),
                "weather_a": (1.0, 1.0),
                "weather_b": (1.02, 0.98),
            }
            packet.context["content_vector"] = mapping[packet.raw]
            return packet

    engine = PatternEngine(config=CoreConfig())
    pipeline = HPMPipeline(
        preprocessor=StubTokenizer(),
        engine=engine,
        postprocessor=ValidationOnlyAdapter(),
        view_configs={"content_view": {"near_threshold": 0.08, "exact_threshold": 0.02}},
    )
    pipeline.register_preprocessor(StubContentAdapter())
    support = IntentBenchmarkSupport(
        engine=engine,
        manager=PatternManager(),
        pipeline=pipeline,
        label_adapter=StubLabelAdapter(),
    )

    support.calibrate_content_view([
        {"text": "flight_a", "intent": "flight"},
        {"text": "flight_b", "intent": "flight"},
        {"text": "weather_a", "intent": "weather"},
        {"text": "weather_b", "intent": "weather"},
    ])

    assert "content_view" in pipeline.view_configs
    config = pipeline.view_configs["content_view"]
    assert config["exact_threshold"] >= 0.015
    assert config["near_threshold"] > config["exact_threshold"]
    assert config["near_threshold"] <= 0.35
