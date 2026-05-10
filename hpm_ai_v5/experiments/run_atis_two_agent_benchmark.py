"""Two-Agent ATIS Intent Recognition Benchmark for HPM v5."""
from __future__ import annotations
import random
import os
import sys
from collections import defaultdict

# Ensure project root is in path
sys.path.append(os.getcwd())

from hpm_ai_v5.adapter.atis import load_atis, IntentLabelAdapter
from hpm_ai_v5.core import (
    PatternEngine,
    PatternManager,
    PatternStore,
    PatternStoreProjector,
    PolygraphPatternRetriever,
)
from hpm_ai_v5.core.config import CoreConfig
from hpm_ai_v5.agents.atis import (
    ATISRouterAgent,
    ATISIntentAgent,
    ATISInferenceAgent,
    FlatIntentStrategy,
    InterconnectedATISInferenceAgent,
    PatternOverseerAgent,
    ViewStoreMatcher,
    _preprocess_atis_packet,
)
from hpm_ai_v5.agents.pipeline import AgentPipeline
from hpm_ai_v5.agents.packet import AgentPacket
from hpm_ai_v5.experiments.atis_shared import (
    ATISBenchmarkSupport,
    build_atis_pipeline,
    evaluate_novel_entity_accuracy,
)


class WeightedATISInferenceAgent(ATISInferenceAgent):
    """Subclass with weighted voting to prioritize semantic views and handle variants."""
    
    def step_packet(self, packet: AgentPacket) -> AgentPacket:
        adapter_packet, views = _preprocess_atis_packet(self.pipeline, packet)
        primary_match = self.pipeline.engine.store.match(adapter_packet.states[-1].value)
        matches = ViewStoreMatcher(self.view_engines).collect(self.pipeline, views, primary_match)
        prediction, intent_votes, _ = FlatIntentStrategy(
            self.pattern_intent,
            near_weight=0.6,
            variant_weight=0.6,
            source_weights={"content_view": 4.0},
        ).predict(matches)
        has_variant = any(match.status == "variant" for _, match in matches if match)
        
        packet.context["predicted_intent"] = prediction
        packet.final_output = prediction
        packet.agent_trace.append(self.name)
        
        packet.log(self.name, {
            "intent": prediction, 
            "has_variant_hit": has_variant
        }, role="agent")
        return packet


class TwoAgentATISBenchmark:
    def __init__(self, consolidation: bool = True):
        self.config = CoreConfig(
            max_patterns=500,
            max_sequences=200,
            history_limit=50,
            near_threshold=1.5,
            consolidation_threshold=0.4,
            consolidation_distance=1.2, 
            density_decay=0.0,
            utility_decay=0.0,
        )
        self.engine = PatternEngine(config=self.config)
        self.manager = PatternManager(promotion_threshold=0.01)
        self.consolidation = consolidation
        self.intent_pipeline, self.intent_adapter = build_atis_pipeline(
            self.engine,
            interconnected=True,
            include_bridge_anchors=True,
            view_configs={
                "content_view": {"near_threshold": 0.6, "exact_threshold": 0.05},
                "skeleton_view": {"near_threshold": 1.0, "exact_threshold": 0.1},
            },
        )
        self.support = ATISBenchmarkSupport(
            engine=self.engine,
            manager=self.manager,
            pipeline=self.intent_pipeline,
            intent_adapter=self.intent_adapter,
            consolidation=consolidation,
            projector=PatternStoreProjector(),
        )
        
        # Agents
        self.router = ATISRouterAgent()
        self.specialist = ATISIntentAgent(
            name="atis_intent_specialist",
            core=self.engine,
            pipeline=self.intent_pipeline
        )
        self.overseer = PatternOverseerAgent(manager=self.manager)
        
        self.training_pipeline = AgentPipeline(agents=[self.router, self.specialist, self.overseer])
        
        self.pattern_intent: dict[str, str] = {}
        self._intent_votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _build_inference_agent(self) -> InterconnectedATISInferenceAgent:
        retriever = None
        if self.support.projector is not None:
            retriever = PolygraphPatternRetriever(self.engine.store, self.support.projector)
        return InterconnectedATISInferenceAgent(
            pattern_intent=self.pattern_intent,
            pipeline=self.intent_pipeline,
            retriever=retriever,
        )

    def _predict(self, text: str) -> str | None:
        packet = AgentPacket(raw=text)
        self.router.step_packet(packet)
        self._build_inference_agent().step_packet(packet)
        return packet.final_output

    def _reset(self):
        self.support.reset()
        self.pattern_intent = self.support.pattern_intent
        self._intent_votes = self.support._intent_votes

    def _train_utterance(self, text: str, intent: str):
        self.support._reset_utterance()
        self.intent_adapter.label = intent
        packet = AgentPacket(raw=text)
        self.training_pipeline.run(packet)
        self.support.record_votes(intent)

    def run_benchmark(self, train: list[dict], test: list[dict]):
        print("B1: Two-Agent Intent Recognition...")
        self._reset()
        random.shuffle(train)
        self.manager.start_episode(self.engine)
        for item in train:
            self._train_utterance(item["text"], item["intent"])
        
        if self.consolidation:
            # Consolidate primary engine
            self.manager.end_episode(self.engine)
            # Consolidate ALL view engines to ensure B4 hits variants
            for ve in self.intent_pipeline.view_engines.values():
                self.manager.end_episode(ve)
            
        self.pattern_intent = {
            pname: max(votes, key=votes.__getitem__)
            for pname, votes in self._intent_votes.items()
        }

        # Map variants to intents based on their members across all engines
        all_engines = [self.engine] + list(self.intent_pipeline.view_engines.values())
        for eng in all_engines:
            for vname, variant in eng.store.variants.items():
                v_votes = defaultdict(int)
                for pname in variant.member_names:
                    p_intent = self.pattern_intent.get(pname)
                    if p_intent:
                        v_votes[p_intent] += 1
                if v_votes:
                    self.pattern_intent[vname] = max(v_votes, key=v_votes.__getitem__)
        
        # Initialize Inference Agent for testing
        correct = 0
        total = len(test)
        for item in test:
            if self._predict(item["text"]) == item["intent"]:
                correct += 1
                
        acc = correct / total
        print(f"  Accuracy: {acc:.2%} ({correct}/{total})")
        return acc

    def run_b2(self, train: list[dict], test: list[dict]) -> float:
        """B2: generalise to utterances with novel named entities (GPE, ORG, LOC)."""
        print("\nB2: Slot Generalisation (novel named entities)...")
        nlp = self.intent_pipeline.preprocessing_pipeline.adapters["nlp_tokenizer"].nlp

        def _predict(text: str) -> str | None:
            return self._predict(text)

        acc, novel_count = evaluate_novel_entity_accuracy(
            train,
            test,
            nlp=nlp,
            predictor=_predict,
            entity_types={"GPE", "ORG", "LOC", "FAC"},
        )
        if novel_count == 0:
            print("  No novel-entity items — skipping")
            return 0.0
        correct = round(acc * novel_count)
        print(f"  Accuracy: {acc:.2%} ({correct}/{novel_count} novel-entity items)")
        return acc

    def run_b3(self, train: list[dict]) -> dict:
        print("\nB3: Consolidation Effectiveness...")
        subset = train[:500]
        
        # Without consolidation
        b_no = TwoAgentATISBenchmark(consolidation=False)
        b_no._reset()
        for item in subset:
            b_no._train_utterance(item["text"], item["intent"])
        size_no = len(b_no.engine.store.patterns)

        # With consolidation
        b_yes = TwoAgentATISBenchmark(consolidation=True)
        # Force lower threshold for B3 subset to ensure consolidation triggers
        b_yes.config.max_patterns = 100
        b_yes.config.consolidation_threshold = 0.5
        b_yes.config.consolidation_distance = 2.0 
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
            packet = AgentPacket(raw=item["text"])
            self.router.step_packet(packet)
            inference_agent = self._build_inference_agent()
            inference_agent.step_packet(packet)
            
            if packet.final_output == item["intent"]:
                # Check logs from inference agent in 'trace'
                log = next((l for l in packet.trace if l.get("agent") == inference_agent.name), {})
                if log.get("detail", {}).get("has_variant_hit"):
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
        # Downscale for speed
        train_main = train[:1000]
        test_main = test[:200]
        print(f"  Train: {len(train_main)}, Test: {len(test_main)}")
        
        b1 = self.run_benchmark(train_main, test_main)
        b2 = self.run_b2(train_main, test_main)
        b3 = self.run_b3(train)
        b4 = self.run_b4(test_main)
        
        print("\n" + "=" * 40)
        print("TWO-AGENT ATIS RESULTS")
        print("=" * 40)
        print(f"B1 Intent Accuracy:      {b1:.2%}  (target >60%)")
        print(f"B2 Slot Generalisation:  {b2:.2%}  (target >70%)")
        print(f"B3 Store Reduction:      {b3['reduction']:.1%}  (target >30%)")
        print(f"B4 Variant Rate:         {b4:.2%}  (target >0%)")
        print("=" * 40)


if __name__ == "__main__":
    random.seed(42)
    TwoAgentATISBenchmark().run_all()
