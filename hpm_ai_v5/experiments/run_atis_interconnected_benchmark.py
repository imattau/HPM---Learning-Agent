"""Experimental interconnected-polygraph ATIS benchmark for HPM v5."""

from __future__ import annotations

import random

from hpm_ai_v5.adapter.atis import load_atis
from hpm_ai_v5.core import PatternEngine, PatternManager, PatternStoreMap, PatternStoreProjector, PolygraphPatternRetriever
from hpm_ai_v5.core.config import CoreConfig
from hpm_ai_v5.agents.atis import InterconnectedATISInferenceAgent
from hpm_ai_v5.agents.packet import AgentPacket
from hpm_ai_v5.experiments.atis_shared import ATISBenchmarkSupport, build_atis_pipeline


class InterconnectedATISBenchmark:
    """ATIS benchmark with bridge-aware multi-view aggregation."""

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
        self.consolidation = consolidation
        self.pipeline, self.intent_adapter = build_atis_pipeline(
            self.engine,
            interconnected=True,
            include_bridge_anchors=True,
            view_configs={
                "content_view": {"near_threshold": 0.08, "exact_threshold": 0.02},
            },
        )
        self.support = ATISBenchmarkSupport(
            engine=self.engine,
            manager=self.manager,
            pipeline=self.pipeline,
            intent_adapter=self.intent_adapter,
            consolidation=consolidation,
            projector=PatternStoreProjector(),
        )
        self.pattern_intent: dict[str, str] = self.support.pattern_intent
        self._intent_votes = self.support._intent_votes
        self._inference_agent: InterconnectedATISInferenceAgent | None = None
        self.store_map: PatternStoreMap | None = None

    def _reset(self):
        self.support.reset()
        self.pattern_intent = self.support.pattern_intent
        self._intent_votes = self.support._intent_votes
        self._inference_agent = None

    def _train_utterance(self, text: str, intent: str):
        self.support.train_utterance(text, intent)

    def train(self, train: list[dict]) -> None:
        self.support.run_training(train, shuffle=True)
        self.pattern_intent = dict(self.support.pattern_intent)
        self.store_map = self.support.projector.build_map() if self.support.projector is not None else None
        self._inference_agent = InterconnectedATISInferenceAgent(
            pattern_intent=self.pattern_intent,
            pipeline=self.pipeline,
            retriever=PolygraphPatternRetriever(self.engine.store, self.support.projector) if self.support.projector is not None else None,
        )

    def _predict(self, text: str, *, use_retriever: bool = True) -> tuple[str | None, dict]:
        if self._inference_agent is None:
            raise RuntimeError("train() must be called before predict()")
        packet = AgentPacket(raw=text)
        self._inference_agent.step_packet(packet, use_retriever=use_retriever)
        return packet.final_output, packet.context

    def run_b1(self, train: list[dict], test: list[dict]) -> float:
        print("\nB1x: Interconnected Intent Recognition...")
        self.train(train)
        correct = 0
        for item in test:
            pred, _ = self._predict(item["text"])
            if pred == item["intent"]:
                correct += 1
        acc = correct / max(len(test), 1)
        print(f"  Accuracy: {acc:.2%} ({correct}/{len(test)})")
        return acc

    def run_b_anchor_metrics(self, test: list[dict]) -> dict[str, float]:
        print("\nB2x: Interconnected Anchor Support...")
        multi_view_correct = 0
        total_correct = 0
        anchor_count = 0
        distinct_anchor_views = 0
        candidate_count = 0
        total = 0

        for item in test:
            pred, context = self._predict(item["text"])
            total += 1
            support = context.get("anchor_support", {})
            anchor_count += len(support)
            distinct_anchor_views += sum(1 for scores in support.values() if scores)
            candidate_count += len(context.get("polygraph_candidates", ()))
            if pred == item["intent"]:
                total_correct += 1
                if context.get("multi_view_anchor_hit"):
                    multi_view_correct += 1

        metrics = {
            "multi_view_correct_rate": multi_view_correct / max(total_correct, 1),
            "avg_anchors_per_utterance": anchor_count / max(total, 1),
            "avg_supported_anchors_per_utterance": distinct_anchor_views / max(total, 1),
            "avg_polygraph_candidates": candidate_count / max(total, 1),
        }
        print(f"  Multi-view support on correct predictions: {metrics['multi_view_correct_rate']:.2%}")
        print(f"  Avg anchors per utterance: {metrics['avg_anchors_per_utterance']:.2f}")
        print(f"  Avg supported anchors per utterance: {metrics['avg_supported_anchors_per_utterance']:.2f}")
        print(f"  Avg polygraph candidates per utterance: {metrics['avg_polygraph_candidates']:.2f}")
        return metrics

    def run_b_retriever_comparison(self, test: list[dict]) -> dict[str, float]:
        print("\nB4x: Retriever Comparison...")
        anchor_only_correct = 0
        anchor_plus_retriever_correct = 0
        changed_prediction_count = 0
        improved_count = 0
        degraded_count = 0

        for item in test:
            _, anchor_ctx = self._predict(item["text"], use_retriever=False)
            retriever_pred, retriever_ctx = self._predict(item["text"], use_retriever=True)
            anchor_pred = anchor_ctx.get("predicted_intent")
            if anchor_pred == item["intent"]:
                anchor_only_correct += 1
            if retriever_pred == item["intent"]:
                anchor_plus_retriever_correct += 1
            if retriever_ctx.get("retriever_changed_prediction"):
                changed_prediction_count += 1
                if anchor_pred != item["intent"] and retriever_pred == item["intent"]:
                    improved_count += 1
                elif anchor_pred == item["intent"] and retriever_pred != item["intent"]:
                    degraded_count += 1

        total = max(len(test), 1)
        metrics = {
            "anchor_only_accuracy": anchor_only_correct / total,
            "anchor_plus_retriever_accuracy": anchor_plus_retriever_correct / total,
            "changed_prediction_rate": changed_prediction_count / total,
            "improved_rate": improved_count / total,
            "degraded_rate": degraded_count / total,
        }
        print(f"  Anchor-only accuracy:        {metrics['anchor_only_accuracy']:.2%}")
        print(f"  Anchor+retriever accuracy:   {metrics['anchor_plus_retriever_accuracy']:.2%}")
        print(f"  Changed prediction rate:     {metrics['changed_prediction_rate']:.2%}")
        print(f"  Improved prediction rate:    {metrics['improved_rate']:.2%}")
        print(f"  Degraded prediction rate:    {metrics['degraded_rate']:.2%}")
        return metrics

    def run_store_map(self) -> dict[str, int]:
        print("\nB3x: Pattern Store Map...")
        if self.store_map is None:
            return {
                "projected_patterns": 0,
                "bridge_hubs": 0,
                "concept_hubs": 0,
                "isolated_patterns": 0,
                "avg_patterns_per_anchor": 0.0,
                "avg_patterns_per_concept": 0.0,
            }
        summary = {
            "projected_patterns": len(self.store_map.projected_patterns),
            "bridge_hubs": len(self.store_map.bridge_hubs),
            "concept_hubs": len(self.store_map.concept_hubs),
            "isolated_patterns": len(self.store_map.isolated_patterns),
            "avg_patterns_per_anchor": self.store_map.avg_patterns_per_anchor,
            "avg_patterns_per_concept": self.store_map.avg_patterns_per_concept,
        }
        print(f"  Projected patterns: {summary['projected_patterns']}")
        print(f"  Bridge hubs:        {summary['bridge_hubs']}")
        print(f"  Concept hubs:       {summary['concept_hubs']}")
        print(f"  Isolated patterns:  {summary['isolated_patterns']}")
        print(f"  Avg patterns/anchor:{summary['avg_patterns_per_anchor']:.2f}")
        print(f"  Avg patterns/concept:{summary['avg_patterns_per_concept']:.2f}")
        return summary

    def run_all(self):
        print("Loading ATIS...")
        train, test = load_atis()
        print(f"  Train: {len(train)}, Test: {len(test)}")
        b1 = self.run_b1(train, test)
        b2 = self.run_b_anchor_metrics(test)
        b3 = self.run_store_map()
        b4 = self.run_b_retriever_comparison(test)
        print("\n" + "=" * 40)
        print("INTERCONNECTED ATIS RESULTS")
        print("=" * 40)
        print(f"B1x Intent Accuracy: {b1:.2%}")
        print(f"B2x Multi-view Correct Rate: {b2['multi_view_correct_rate']:.2%}")
        print(f"B2x Avg Anchors/Utt: {b2['avg_anchors_per_utterance']:.2f}")
        print(f"B3x Projected Patterns: {b3['projected_patterns']}")
        print(f"B3x Avg Patterns/Anchor: {b3['avg_patterns_per_anchor']:.2f}")
        print(f"B3x Avg Patterns/Concept: {b3['avg_patterns_per_concept']:.2f}")
        print(f"B4x Anchor-only Accuracy: {b4['anchor_only_accuracy']:.2%}")
        print(f"B4x Anchor+Retriever Accuracy: {b4['anchor_plus_retriever_accuracy']:.2%}")
        print("=" * 40)


if __name__ == "__main__":
    random.seed(42)
    InterconnectedATISBenchmark().run_all()
