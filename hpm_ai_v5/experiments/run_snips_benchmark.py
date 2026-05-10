"""SNIPS Intent Recognition Benchmark for HPM v5."""

from __future__ import annotations

import random

from hpm_ai_v5.adapter.snips import IntentLabelAdapter, SNIPSCanonicalIntentAdapter, load_snips
from hpm_ai_v5.agents.atis import AuditableIntentInferenceAgent
from hpm_ai_v5.agents.packet import AgentPacket
from hpm_ai_v5.core import PatternEngine, PatternManager
from hpm_ai_v5.core.config import CoreConfig
from hpm_ai_v5.experiments.intent_error_analysis import analyze_intent_errors
from hpm_ai_v5.experiments.intent_shared import IntentBenchmarkSupport, build_intent_pipeline


class SNIPSBenchmark:
    def __init__(self, consolidation: bool = True):
        self.config = CoreConfig(
            max_patterns=8192,
            max_sequences=1024,
            history_limit=100,
            near_threshold=1.5,
            consolidation_threshold=0.8,
            consolidation_distance=0.8,
            density_decay=0.0,
            utility_decay=0.0,
        )
        self.engine = PatternEngine(config=self.config)
        self.manager = PatternManager(promotion_threshold=0.01)
        self.pipeline, self.intent_adapter = build_intent_pipeline(
            self.engine,
            label_adapter=IntentLabelAdapter(),
            canonical_override_adapter=SNIPSCanonicalIntentAdapter(),
            view_configs={"content_view": {"near_threshold": 0.08, "exact_threshold": 0.02}},
        )
        self.support = IntentBenchmarkSupport(
            engine=self.engine,
            manager=self.manager,
            pipeline=self.pipeline,
            label_adapter=self.intent_adapter,
            consolidation=consolidation,
        )
        self.pattern_intent: dict[str, str] = self.support.pattern_intent
        self._inference_agent: AuditableIntentInferenceAgent | None = None

    def train(self, train: list[dict]) -> None:
        self.support.run_training(train, shuffle=True)
        self.pattern_intent = dict(self.support.pattern_intent)
        self._inference_agent = AuditableIntentInferenceAgent(
            self.pipeline.view_engines,
            self.pattern_intent,
            self.pipeline,
        )

    def _predict(self, text: str) -> str | None:
        if self._inference_agent is None:
            raise RuntimeError("train() must be called before predict()")
        packet = AgentPacket(raw=text)
        self._inference_agent.step_packet(packet)
        return packet.final_output

    def _predict_with_context(self, text: str) -> tuple[str | None, dict]:
        if self._inference_agent is None:
            raise RuntimeError("train() must be called before predict()")
        packet = AgentPacket(raw=text)
        self._inference_agent.step_packet(packet)
        return packet.final_output, dict(packet.context)

    def run_b1(self, train: list[dict], test: list[dict]) -> float:
        print("\nB1: SNIPS Intent Recognition...")
        self.train(train)
        acc = self.support.evaluate_predictions(test, self._predict)
        print(f"  Accuracy: {acc:.2%} ({round(acc * len(test))}/{len(test)})")
        return acc

    def run_error_analysis(self, test: list[dict]) -> dict[str, object]:
        print("\nSNIPS Error Analysis...")
        analysis = analyze_intent_errors(test, self._predict_with_context)
        print(f"  Errors: {analysis['n_errors']}")
        print(f"  Categories: {analysis['category_counts']}")
        print(f"  Top confusions: {analysis['top_confusions'][:5]}")
        return analysis

    def run_all(self) -> None:
        print("Loading SNIPS...")
        train, test = load_snips()
        print(f"  Train: {len(train)}, Test: {len(test)}")
        b1 = self.run_b1(train, test)
        print("\n" + "=" * 40)
        print("SNIPS RESULTS")
        print("=" * 40)
        print(f"B1 Intent Accuracy: {b1:.2%}")
        print("=" * 40)


if __name__ == "__main__":
    random.seed(42)
    SNIPSBenchmark().run_all()
