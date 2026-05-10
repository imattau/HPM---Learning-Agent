"""ATIS Intent Recognition Benchmark (B1-B4) for HPM v5."""
from __future__ import annotations
import random
import pickle
from collections import defaultdict
from pathlib import Path

from hpm_ai_v5.adapter.atis import load_atis
from hpm_ai_v5.core import PatternEngine, PatternManager
from hpm_ai_v5.core.config import CoreConfig
from hpm_ai_v5.experiments.atis_shared import (
    ATISBenchmarkSupport,
    build_atis_pipeline,
    evaluate_novel_entity_accuracy,
)
from hpm_ai_v5.experiments.intent_error_analysis import analyze_intent_errors

from hpm_ai_v5.agents.atis import AuditableIntentInferenceAgent
from hpm_ai_v5.agents.packet import AgentPacket

class ATISBenchmark:
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
        )
        self.pipeline.polygraph_confidence_skip = 1.1
        self.pipeline.view_configs = {
            # content_view uses 96-dim unit vectors; MAE distances are 0.02-0.20
            # near_threshold must be tuned to this range, not the token-ID default of 1.5
            "content_view": {"near_threshold": 0.08, "exact_threshold": 0.02},
        }
        self.pattern_intent: dict[str, str] = {}        # pattern_name -> winning intent
        self._intent_votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._inference_agent: AuditableIntentInferenceAgent | None = None

    def _reset(self):
        self.support.reset()
        self.pattern_intent = self.support.pattern_intent
        self._intent_votes = self.support._intent_votes
        self._inference_agent = None

    def _train_utterance(self, text: str, intent: str):
        self.support.train_utterance(text, intent)

    def _predict_intent(self) -> str | None:
        """Vote across primary + all view matches using O(1) pattern_intent lookup."""
        intent_votes: dict[str, float] = defaultdict(float)
        
        matches = [self.engine.last_match] + list(self.pipeline.view_matches.values())
        
        for match in matches:
            if not match or not match.pattern:
                continue
            weight = 1.0 if match.status == "exact" else 0.5 if match.status == "near" else 0.0
            if weight == 0:
                continue
            intent = self.pattern_intent.get(match.pattern.name)
            if intent:
                intent_votes[intent] += weight * match.pattern.utility
        return max(intent_votes, key=intent_votes.__getitem__) if intent_votes else None

    _CHECKPOINT = "checkpoints/atis_bundle.pkl"
    _INTENT_CHECKPOINT = "checkpoints/atis_intent.json"

    def _save_checkpoint_bundle(self) -> None:
        path = Path(self._CHECKPOINT)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "manager": self.manager,
            "engine_store": self.engine.store,
            "view_engines": {
                name: {"config": engine.config, "store": engine.store}
                for name, engine in self.pipeline.view_engines.items()
            },
            "pattern_intent": dict(self.pattern_intent),
        }
        with path.open("wb") as handle:
            pickle.dump(bundle, handle)

    def _load_checkpoint_bundle(self) -> bool:
        path = Path(self._CHECKPOINT)
        if not path.exists():
            return False
        with path.open("rb") as handle:
            bundle = pickle.load(handle)

        self.manager = bundle["manager"]
        self.engine.store = bundle["engine_store"]
        self.pattern_intent = dict(bundle["pattern_intent"])
        self.pipeline.view_engines.clear()
        for name, state in bundle.get("view_engines", {}).items():
            self.pipeline.view_engines[name] = PatternEngine(
                config=state["config"],
                store=state["store"],
            )
        return True

    def run_b1(self, train: list[dict], test: list[dict]) -> float:
        print("\nB1: Intent Recognition...")
        self._reset()
        if self._load_checkpoint_bundle():
            self._inference_agent = AuditableIntentInferenceAgent(
                self.pipeline.view_engines, self.pattern_intent, self.pipeline
            )
            print(f"  Loaded from checkpoint ({len(self.pattern_intent)} patterns mapped)")
        else:
            self.support.run_training(train, shuffle=True)
            self.pattern_intent = dict(self.support.pattern_intent)
            self._save_checkpoint_bundle()
            print(f"  Trained and saved checkpoint ({len(self.pattern_intent)} patterns)")

        # Build inference agent from frozen view stores
        self._inference_agent = AuditableIntentInferenceAgent(
            self.pipeline.view_engines, self.pattern_intent, self.pipeline
        )

        correct = 0
        total = len(test)
        for item in test:
            if self._run_and_predict(item["text"]) == item["intent"]:
                correct += 1
        
        acc = correct / total
        print(f"  Accuracy: {acc:.2%} ({correct}/{total})")
        return acc

    def _run_and_predict(self, text: str) -> str | None:
        if self._inference_agent is not None:
            packet = AgentPacket(raw=text)
            self._inference_agent.step_packet(packet)
            return packet.final_output
        # Fallback: full pipeline step (used during B3/B4 which run without inference agent)
        self.engine.current_state = None
        self.engine.history = []
        for adapter in self.pipeline.preprocessing_pipeline.adapters.values():
            if hasattr(adapter, "reset"):
                adapter.reset()
        self.intent_adapter.label = None
        self.pipeline.step(text)
        return self._predict_intent()

    def _run_and_predict_with_context(self, text: str) -> tuple[str | None, dict]:
        if self._inference_agent is not None:
            packet = AgentPacket(raw=text)
            self._inference_agent.step_packet(packet)
            return packet.final_output, dict(packet.context)
        prediction = self._run_and_predict(text)
        return prediction, {}

    def run_b2(self, train: list[dict], test: list[dict]) -> float:
        """B2: generalise to utterances with novel named entities (GPE, ORG, LOC).
        Uses spaCy NER to find test utterances containing entity types not seen
        in training — the meaningful novelty criterion from SNLP T3 lessons.
        """
        print("\nB2: Slot Generalisation (novel named entities)...")
        nlp = self.pipeline.preprocessing_pipeline.adapters["nlp_tokenizer"].nlp
        acc, novel_count = evaluate_novel_entity_accuracy(
            train,
            test,
            nlp=nlp,
            predictor=self._run_and_predict,
            entity_types={"GPE", "ORG", "LOC", "FAC"},
            train_limit=1000,
        )
        if novel_count == 0:
            print("  No novel-entity items — skipping")
            return 0.0
        correct = round(acc * novel_count)
        print(f"  Accuracy: {acc:.2%} ({correct}/{novel_count} novel-entity items)")
        return acc

    def run_b3(self, train: list[dict]) -> dict:
        print("\nB3: Consolidation Effectiveness...")
        subset = train[:1000]
        
        # Without consolidation — same config as B1
        b_no = ATISBenchmark(consolidation=False)
        b_no._reset()
        for item in subset:
            b_no._train_utterance(item["text"], item["intent"])
        size_no = len(b_no.engine.store.patterns)

        # With consolidation — same config as B1
        b_yes = ATISBenchmark(consolidation=True)
        b_yes._reset()
        b_yes.manager.start_episode(b_yes.engine)
        for item in subset:
            b_yes._train_utterance(item["text"], item["intent"])
        b_yes.manager.end_episode(b_yes.engine)
        size_yes = len(b_yes.engine.store.patterns)
        variants = len(b_yes.engine.store.variants)
        reduction = (size_no - size_yes) / max(size_no, 1)
        
        print(f"  Without consolidation: {size_no} patterns")
        print(f"  With consolidation:    {size_yes} patterns, {variants} variants")
        print(f"  Reduction: {reduction:.1%}")
        return {"size_without": size_no, "size_with": size_yes, "variants": variants, "reduction": reduction}

    def run_b4(self, test: list[dict]) -> float:
        print("\nB4: Variant Match Rate...")
        variant_hits = 0
        concrete_hits = 0
        for item in test:
            self.engine.current_state = None
            self.engine.history = []
            self.pipeline.step(item["text"])
            pred = self._predict_intent()
            if pred == item["intent"]:
                match = self.engine.last_match
                if match and match.status == "variant":
                    variant_hits += 1
                else:
                    concrete_hits += 1
        
        total = variant_hits + concrete_hits
        rate = variant_hits / max(total, 1)
        print(f"  Concrete correct: {concrete_hits}, Variant correct: {variant_hits}")
        print(f"  Variant contribution: {rate:.2%}")
        return rate

    def run_error_analysis(self, test: list[dict]) -> dict[str, object]:
        print("\nATIS Error Analysis...")
        analysis = analyze_intent_errors(test, self._run_and_predict_with_context)
        print(f"  Errors: {analysis['n_errors']}")
        print(f"  Categories: {analysis['category_counts']}")
        print(f"  Top confusions: {analysis['top_confusions'][:5]}")
        return analysis

    def run_all(self):
        print("Loading ATIS...")
        train, test = load_atis()
        print(f"  Train: {len(train)}, Test: {len(test)}")
        
        b1 = self.run_b1(train, test)
        b2 = self.run_b2(train, test)
        b3 = self.run_b3(train)
        b4 = self.run_b4(test)
        
        print("\n" + "=" * 40)
        print("ATIS RESULTS")
        print("=" * 40)
        print(f"B1 Intent Accuracy:      {b1:.2%}  (target >60%)")
        print(f"B2 Slot Generalisation:  {b2:.2%}  (target >70%)")
        print(f"B3 Store Reduction:      {b3['reduction']:.1%}  (target >30%)")
        print(f"B4 Variant Rate:         {b4:.2%}  (target >0%)")
        print("=" * 40)


if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    ATISBenchmark().run_all()
