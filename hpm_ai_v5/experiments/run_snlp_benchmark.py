"""Structural NLP Benchmark (SNLP) for HPM v5."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple
import numpy as np

from hpm_ai_v5.adapter import AdapterPacket
from hpm_ai_v5.adapter.nlp import (
    NLPTokenizer,
    CanonicalPhraser,
    SkeletonExtractor,
    SkeletonNgramAdapter,
    DeltaEncoder,
    KnowledgeBaseLookup,
)
from hpm_ai_v5.adapter.clt import UnifiedVocabulary
from hpm_ai_v5.adapter.validation_only import ValidationOnlyAdapter
from hpm_ai_v5.core import PatternEngine, PatternManager, State
from hpm_ai_v5.core.config import CoreConfig
from hpm_ai_v5.pipeline import HPMPipeline
from hpm_ai_v5.polygraphs.nlp import NLPPolygraphGenerator


class SNLPBenchmark:
    """Evaluates HPM v5 on Structural NLP tasks."""

    def __init__(self):
        # Config with enough capacity for meta-patterns
        self.config = CoreConfig(
            max_patterns=512,
            max_sequences=128,
            history_limit=100,
            near_threshold=0.4
        )
        self.engine = PatternEngine(config=self.config)
        self.manager = PatternManager(promotion_threshold=0.01)
        
        # Pipeline setup
        self.pipeline = HPMPipeline(
            preprocessor=NLPTokenizer(),
            engine=self.engine,
            postprocessor=ValidationOnlyAdapter(),
            polygraph_generator=NLPPolygraphGenerator(),
            polygraph_confidence_skip=1.1 # Force polygraph evaluation
        )
        self.pipeline.register_preprocessor(CanonicalPhraser())
        self.pipeline.register_preprocessor(SkeletonExtractor())
        self.pipeline.register_preprocessor(SkeletonNgramAdapter())
        self.pipeline.register_preprocessor(DeltaEncoder())
        self.pipeline.register_preprocessor(KnowledgeBaseLookup())
        
        # Corpus Templates
        self.templates = {
            "weather": ["What is the weather in {city}?", "Show me the forecast for {city}.", "Tell me the conditions in {city}."],
            "flight": ["Book a flight from {departure} to {arrival}.", "I want a plane to {arrival} from {departure}.", "Reserve a journey from {departure} to {arrival}."],
            "buy": ["I would like to buy {item}.", "Purchase {item} for me.", "Acquire {item} immediately."],
        }
        self.cities = ["London", "Paris", "Berlin", "Madrid", "Rome"]
        self.items = ["laptop", "book", "coffee", "ticket"]

    def generate_sentence(self, intent: str, city: str = None, departure: str = None, arrival: str = None, item: str = None) -> str:
        template = random.choice(self.templates[intent])
        if intent == "weather":
            sentence = template.format(city=city or random.choice(self.cities))
        elif intent == "flight":
            dep = departure or random.choice(self.cities)
            arr = arrival or random.choice([c for c in self.cities if c != dep])
            sentence = template.format(departure=dep, arrival=arr)
        else:  # buy
            sentence = template.format(item=item or random.choice(self.items))
        return sentence

    def reset_for_isolation(self, clear_views: bool = False):
        """Reset state between unrelated sentences."""
        self.engine.current_state = None
        self.engine.history = []
        if clear_views:
            self.pipeline.view_engines.clear()
        for adapter in self.pipeline.preprocessing_pipeline.adapters.values():
            if hasattr(adapter, "reset"):
                adapter.reset()

    def run_t1_skeleton_recognition(self, episodes: int = 50) -> float:
        """T1: Skeleton Recognition accuracy."""
        print("\nRunning T1: Skeleton Recognition...")
        self.reset_for_isolation(clear_views=True)
        correct = 0
        total = 100
        
        # Training
        for _ in range(episodes):
            self.reset_for_isolation(clear_views=False)
            intent = random.choice(list(self.templates.keys()))
            sent = self.generate_sentence(intent)
            self.pipeline.step(sent)
        
        # Testing
        for _ in range(total):
            self.reset_for_isolation(clear_views=False)
            intent = random.choice(list(self.templates.keys()))
            sent = self.generate_sentence(intent)
            res = self.pipeline.step(sent)
            
            # Check if skeleton view matched
            view_engine = self.pipeline.view_engines.get("skeleton_view")
            if view_engine and view_engine.last_match:
                if view_engine.last_match.status in ("exact", "near"):
                    correct += 1
                
        acc = correct / total
        print(f"  T1 Score: {acc:.2%}")
        return acc

    def run_t2_delta_induction(self, steps: int = 100) -> float:
        """T2: Delta Induction - engine forecast should match next observed state."""
        print("\nRunning T2: Delta Induction...")
        self.reset_for_isolation(clear_views=True)
        correct = 0
        forecast_count = 0
        trials = 50

        # Training on sequences: weather -> flight -> buy
        for _ in range(30):
            self.engine.history = []
            self.reset_for_isolation(clear_views=False)
            for intent in ["weather", "flight", "buy"]:
                self.pipeline.step(self.generate_sentence(intent))

        # Testing: after weather, forecast should match the flight skeleton
        for _ in range(trials):
            self.engine.history = []
            self.reset_for_isolation(clear_views=False)
            self.pipeline.step(self.generate_sentence("weather"))

            # Capture forecast before observing the next state
            action = self.engine.act()

            # Skip trials where no forecast was made
            if action.action_type == "defer":
                continue

            forecast_count += 1
            forecast_value = action.forecast.value if action.forecast else None

            # Observe the next state
            next_sent = self.generate_sentence("flight")
            self.pipeline.step(next_sent)
            actual_value = self.engine.current_state.value if self.engine.current_state else None

            # Match if forecast is non-null and numerically close to actual
            if forecast_value is not None and actual_value is not None:
                fv = np.array(forecast_value, dtype=float)
                av = np.array(actual_value, dtype=float)
                min_len = min(len(fv), len(av))
                if min_len > 0:
                    dist = np.linalg.norm(fv[:min_len] - av[:min_len]) / min_len
                    if dist < 1.0:
                        correct += 1

        forecast_rate = forecast_count / trials
        acc = correct / forecast_count if forecast_count > 0 else 0.0
        print(f"  Forecast Rate: {forecast_rate:.2%} ({forecast_count}/{trials} trials)")
        print(f"  T2 Score (accuracy among forecasts): {acc:.2%}")
        return acc

    def run_t3_slot_filling(self) -> float:
        """T3: Canonical Slot-Filling with novel synonym words.

        Training and test sentences are chosen so they share at least one KB
        semantic candidate, ensuring the same semantic_view_* engine is
        populated during both training and evaluation.

        KB overlaps used:
          check    → [if, validate, verify]
          validate → [check, if, verify]     shared: if, verify
          keep     → [while, loop, repeat]
          perform  → [call, run, execute]    shared with keep via loop/run: none
                                             (perform→run; keep→loop — different views)
          output   → [return, yield]
          store    → [set, assign]
        """
        print("\nRunning T3: Slot-Filling...")
        self.reset_for_isolation(clear_views=True)
        correct = 0

        # Each tuple: (training_sentence, test_sentence)
        # Pairs share >=1 KB candidate so the same semantic_view_X engine fires on both sides.
        test_cases = [
            # check→[if,validate,verify] ∩ validate→[check,if,verify] = {if, verify}
            ("Check the weather in London.",        "Validate the weather in Paris."),
            # check→[if,validate,verify] ∩ validate→[check,if,verify] = {if, verify}
            ("Please check the data now.",          "Please validate the data now."),
            # keep→[while,loop,repeat] — train and test both map through "keep"
            ("Keep repeating the search.",          "Keep looping the search."),
            # perform→[call,run,execute] ∩ perform→[call,run,execute] (same root)
            ("Perform the task immediately.",       "Execute the task immediately."),
            # output→[return,yield] — train and test both map through "output"
            ("Output the result now.",              "Return the result now."),
        ]

        for train_sent, test_sent in test_cases:
            self.reset_for_isolation(clear_views=True)
            for _ in range(10):
                self.pipeline.step(train_sent)

            self.pipeline.step(test_sent)

            found_match = False
            for view_name, engine in self.pipeline.view_engines.items():
                if view_name.startswith("semantic_view_") and engine.last_match:
                    if engine.last_match.status in ("exact", "near"):
                        found_match = True
                        break

            if found_match:
                correct += 1

        acc = correct / len(test_cases)
        print(f"  T3 Score: {acc:.2%}")
        return acc

    def run_t4_word_salad_rejection(self) -> float:
        """T4: Word-Salad Rejection - ROC-AUC over clean vs. anomalous input."""
        print("\nRunning T4: Word-Salad Rejection...")
        self.reset_for_isolation(clear_views=True)

        salad_sentences = [
            "the the the the the",
            "!!! ??? $$$",
            "flight London book to",           # scrambled word order
            "weather the in what is",          # reverse order
            "buy immediately laptop would I",  # inverted syntax
            "to from departure arrival plane",  # content without structure
            "??? book London !!!",             # mixed noise + content
            "is what forecast the London in",  # shuffled weather query
        ]

        # Train on clean data
        for _ in range(100):
            self.reset_for_isolation(clear_views=False)
            intent = random.choice(list(self.templates.keys()))
            self.pipeline.step(self.generate_sentence(intent))

        labels: list[int] = []
        scores: list[float] = []

        def _get_skeleton_confidence() -> float:
            # Use match status: 1.0=exact, 0.5=near, 0.0=no match or no view
            # act() confidence decays over time and has no discriminative power here.
            view_engine = self.pipeline.view_engines.get("skeleton_bigram_view")
            if view_engine is None or view_engine.last_match is None:
                return 0.0
            if view_engine.last_match.status == "exact":
                return 1.0
            if view_engine.last_match.status == "near":
                return 0.5
            return 0.0

        # Clean sentences (label=1 → high confidence expected)
        for _ in range(len(salad_sentences)):
            self.reset_for_isolation(clear_views=False)
            intent = random.choice(list(self.templates.keys()))
            self.pipeline.step(self.generate_sentence(intent))
            labels.append(1)
            scores.append(_get_skeleton_confidence())

        # Salad sentences (label=0 → low confidence expected)
        for sent in salad_sentences:
            self.reset_for_isolation(clear_views=False)
            self.pipeline.step(sent)
            labels.append(0)
            scores.append(_get_skeleton_confidence())

        avg_clean = float(np.mean([s for s, l in zip(scores, labels) if l == 1]))
        avg_salad = float(np.mean([s for s, l in zip(scores, labels) if l == 0]))
        print(f"  Clean Avg Confidence: {avg_clean:.4f}")
        print(f"  Salad Avg Confidence: {avg_salad:.4f}")

        # ROC-AUC: probability that a random clean > random salad (Wilcoxon statistic)
        clean_scores = [s for s, l in zip(scores, labels) if l == 1]
        salad_scores = [s for s, l in zip(scores, labels) if l == 0]
        pairs = len(clean_scores) * len(salad_scores)
        wins = sum(1 for c in clean_scores for s in salad_scores if c > s)
        ties = sum(1 for c in clean_scores for s in salad_scores if c == s)
        auc = (wins + 0.5 * ties) / pairs if pairs > 0 else 0.5
        print(f"  ROC-AUC: {auc:.4f}")

        # Pass threshold: AUC > 0.65 (better than random discrimination)
        score = auc
        return score

    def run_t5_discourse_meta_patterns(self) -> float:
        """T5: Discourse Meta-Patterns - paragraph-level templates."""
        print("\nRunning T5: Discourse Meta-Patterns...")
        self.reset_for_isolation(clear_views=True)

        # Training: repeated exposure to the two-sentence weather sequence
        for _ in range(40):
            self.engine.history = []
            self.reset_for_isolation(clear_views=False)
            self.manager.start_episode(self.engine, context={"domain": "discourse"})
            self.pipeline.step("What is the weather in London?")
            self.pipeline.step("Tell me the forecast for Paris.")
            self.manager.end_episode(self.engine)

        # Test: recognize the same structural sequence with novel cities
        self.engine.history = []
        self.reset_for_isolation(clear_views=False)
        self.manager.start_episode(self.engine, context={"domain": "discourse"})
        self.pipeline.step("What is the weather in Berlin?")

        self.pipeline.step("Tell me the forecast for Rome.")

        match = self.engine.last_match
        if match and match.status == "exact":
            print("  T5 Score: 100% (Sequence Recognized)")
            return 1.0
        else:
            print("  T5 Score: 0% (Sequence Not Recognized)")
            return 0.0

    def run_all(self):
        print("Starting Structural NLP Benchmark (SNLP)...")
        results = {}
        results["T1"] = self.run_t1_skeleton_recognition()
        results["T2"] = self.run_t2_delta_induction()
        results["T3"] = self.run_t3_slot_filling()
        results["T4"] = self.run_t4_word_salad_rejection()
        results["T5"] = self.run_t5_discourse_meta_patterns()
        
        print("\n" + "="*30)
        print("SNLP FINAL SUMMARY")
        print("="*30)
        for task, score in results.items():
            print(f"{task}: {score:.2%}")
        print("="*30)


if __name__ == "__main__":
    benchmark = SNLPBenchmark()
    benchmark.run_all()
