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
    DeltaEncoder,
    KnowledgeBaseLookup
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
        """T2: Delta Induction - predicting the next structural change."""
        print("\nRunning T2: Delta Induction...")
        self.reset_for_isolation(clear_views=True)
        correct = 0
        
        # Training on sequences
        for _ in range(30):
            self.engine.history = []
            self.reset_for_isolation(clear_views=False)
            intents = ["weather", "flight", "buy"]
            for intent in intents:
                sent = self.generate_sentence(intent)
                self.pipeline.step(sent)
                
        # Testing prediction
        for _ in range(50):
            self.engine.history = []
            self.reset_for_isolation(clear_views=False)
            self.pipeline.step(self.generate_sentence("weather"))
            
            # Predict next
            action = self.engine.act()
            next_sent = self.generate_sentence("flight")
            res = self.pipeline.step(next_sent)
            
            if self.engine.last_match and self.engine.last_match.status == "exact":
                correct += 1
                
        acc = correct / 50
        print(f"  T2 Score: {acc:.2%}")
        return acc

    def run_t3_slot_filling(self) -> float:
        """T3: Canonical Slot-Filling with novel words."""
        print("\nRunning T3: Slot-Filling...")
        self.reset_for_isolation(clear_views=True)
        correct = 0
        # 'Validate' maps to [check, if, verify]
        # We'll train on 'check'
        test_cases = [
            ("Validate the weather.", "WEATHER"),
        ]
        
        for query, expected_concept in test_cases:
            self.reset_for_isolation(clear_views=True)
            # Train on 'check'
            for _ in range(10):
                self.pipeline.step("check")
                
            res = self.pipeline.step(query)
            
            # Check if ANY of the semantic views matched a pattern
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
        """T4: Word-Salad Rejection - confidence should drop for anomalous input."""
        print("\nRunning T4: Word-Salad Rejection...")
        self.reset_for_isolation(clear_views=True)
        
        # Train on clean data
        for _ in range(50):
            self.reset_for_isolation(clear_views=False)
            intent = random.choice(list(self.templates.keys()))
            self.pipeline.step(self.generate_sentence(intent))
            
        # Test clean
        clean_confs = []
        for _ in range(20):
            self.reset_for_isolation(clear_views=False)
            intent = random.choice(list(self.templates.keys()))
            res = self.pipeline.step(self.generate_sentence(intent))
            view_engine = self.pipeline.view_engines.get("skeleton_view")
            if view_engine:
                action = view_engine.act()
                clean_confs.append(action.confidence)
            
        # Test salad - more chaotic
        salad_sentences = [
            "the the the the the",
            "!!! ??? $$$",
        ]
        salad_confs = []
        for sent in salad_sentences:
            self.reset_for_isolation(clear_views=False)
            res = self.pipeline.step(sent)
            view_engine = self.pipeline.view_engines.get("skeleton_view")
            if view_engine:
                action = view_engine.act()
                salad_confs.append(action.confidence)
            
        avg_clean = np.mean(clean_confs) if clean_confs else 0.0
        avg_salad = np.mean(salad_confs) if salad_confs else 0.0
        
        print(f"  Clean Avg Confidence: {avg_clean:.4f}")
        print(f"  Salad Avg Confidence: {avg_salad:.4f}")
        
        # Pass if avg_salad is lower
        return 1.0 if avg_salad < avg_clean * 0.9 else 0.0

    def run_t5_discourse_meta_patterns(self) -> float:
        """T5: Discourse Meta-Patterns - paragraph-level templates."""
        print("\nRunning T5: Discourse Meta-Patterns...")
        self.reset_for_isolation(clear_views=True)
        
        # Training
        for _ in range(40):
            self.engine.history = []
            self.reset_for_isolation(clear_views=False)
            self.pipeline.step("What is the weather in London?")
            self.pipeline.step("Tell me the forecast for Paris.")
            self.manager.end_episode(self.engine)
            
        # Test: Recognize the sequence
        self.engine.history = []
        self.reset_for_isolation(clear_views=False)
        self.pipeline.step("What is the weather in Berlin?")
        
        res = self.pipeline.step("Tell me the forecast for Rome.")
        
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
