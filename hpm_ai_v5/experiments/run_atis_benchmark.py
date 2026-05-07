"""ATIS Intent Recognition Benchmark (B1-B4) for HPM v5."""
from __future__ import annotations
import random
from collections import defaultdict

from hpm_ai_v5.adapter.nlp import (
    NLPTokenizer, CanonicalPhraser, NamedEntityCanonicaliser, SkeletonExtractor,
    SkeletonNgramAdapter, KnowledgeBaseLookup, StartOfEpisodeAdapter,
)
from hpm_ai_v5.adapter.atis import load_atis, IntentLabelAdapter
from hpm_ai_v5.adapter.validation_only import ValidationOnlyAdapter
from hpm_ai_v5.core import PatternEngine, PatternManager, PatternStore
from hpm_ai_v5.core.config import CoreConfig
from hpm_ai_v5.pipeline import HPMPipeline
from hpm_ai_v5.polygraphs.nlp import NLPPolygraphGenerator


class ATISBenchmark:
    def __init__(self, consolidation: bool = True):
        self.config = CoreConfig(
            max_patterns=2048,
            max_sequences=512,
            history_limit=100,
            near_threshold=1.5,
            consolidation_threshold=0.8,
        )
        self.engine = PatternEngine(config=self.config)
        self.manager = PatternManager(promotion_threshold=0.01)
        self.consolidation = consolidation
        self.pipeline = HPMPipeline(
            preprocessor=NLPTokenizer(),
            engine=self.engine,
            postprocessor=ValidationOnlyAdapter(),
            polygraph_generator=NLPPolygraphGenerator(),
            polygraph_confidence_skip=1.1,
        )
        self.pipeline.register_preprocessor(StartOfEpisodeAdapter())
        self.pipeline.register_preprocessor(CanonicalPhraser())
        self.pipeline.register_preprocessor(NamedEntityCanonicaliser())
        self.pipeline.register_preprocessor(SkeletonExtractor())
        self.pipeline.register_preprocessor(SkeletonNgramAdapter())
        self.pipeline.register_preprocessor(KnowledgeBaseLookup())
        self.intent_adapter = IntentLabelAdapter()
        self.pipeline.register_preprocessor(self.intent_adapter)
        self.intent_patterns: dict[str, list[str]] = defaultdict(list)

    def _reset(self):
        self.engine.current_state = None
        self.engine.history = []
        self.pipeline.view_engines.clear()
        self.engine.store = PatternStore(config=self.config)
        self.intent_patterns.clear()
        for adapter in self.pipeline.preprocessing_pipeline.adapters.values():
            if hasattr(adapter, "reset"):
                adapter.reset()

    def _train_utterance(self, text: str, intent: str):
        self.engine.current_state = None
        self.engine.history = []
        self.intent_adapter.label = intent
        self.pipeline.step(text)
        # Track primary engine match
        if self.engine.last_match and self.engine.last_match.pattern:
            self.intent_patterns[intent].append(self.engine.last_match.pattern.name)
        # Track all view engine matches so they can contribute votes at inference
        for view_engine in self.pipeline.view_engines.values():
            if view_engine.last_match and view_engine.last_match.pattern:
                pname = view_engine.last_match.pattern.name
                if pname not in self.intent_patterns.get(intent, []):
                    self.intent_patterns[intent].append(pname)

    def _predict_intent(self) -> str | None:
        """Vote across primary + all view engines using intent_patterns lookup."""
        all_engines = [self.engine] + list(self.pipeline.view_engines.values())
        intent_votes: dict[str, float] = defaultdict(float)

        for engine in all_engines:
            match = engine.last_match
            if not match or not match.pattern:
                continue
            weight = 1.0 if match.status == "exact" else 0.5 if match.status == "near" else 0.0
            if weight == 0:
                continue
            for intent, names in self.intent_patterns.items():
                if match.pattern.name in names:
                    intent_votes[intent] += weight * match.pattern.utility
                    break

        return max(intent_votes, key=intent_votes.__getitem__) if intent_votes else None

    def run_b1(self, train: list[dict], test: list[dict]) -> float:
        print("\nB1: Intent Recognition...")
        self._reset()
        random.shuffle(train)
        # Scale training down to 1000 for development speed
        train_set = train[:1000]
        self.manager.start_episode(self.engine)
        for item in train_set:
            self._train_utterance(item["text"], item["intent"])
        if self.consolidation:
            self.manager.end_episode(self.engine)
        
        correct = 0
        total = len(test)
        for item in test:
            if self._run_and_predict(item["text"]) == item["intent"]:
                correct += 1
        
        acc = correct / total
        print(f"  Accuracy: {acc:.2%} ({correct}/{total})")
        return acc

    def _run_and_predict(self, text: str) -> str | None:
        self.engine.current_state = None
        self.engine.history = []
        self.intent_adapter.label = None
        self.pipeline.step(text)
        return self._predict_intent()

    def run_b2(self, train: list[dict], test: list[dict]) -> float:
        print("\nB2: Slot Generalisation...")
        train_tokens: set[str] = set()
        for item in train:
            train_tokens.update(item["text"].lower().split())
        
        novel = [i for i in test if any(w not in train_tokens for w in i["text"].lower().split())]
        if not novel:
            print("  No novel-token items — skipping")
            return 0.0

        correct = sum(1 for item in novel if self._run_and_predict(item["text"]) == item["intent"])
        acc = correct / len(novel)
        print(f"  Accuracy: {acc:.2%} ({correct}/{len(novel)} novel-token items)")
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
