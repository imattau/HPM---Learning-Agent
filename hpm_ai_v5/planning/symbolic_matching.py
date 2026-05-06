"""Symbolic Pattern Matching Benchmark for HPM v5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..adapter import AdapterPacket
from ..adapter.nlp import CanonicalPhraser, NLPTokenizer, ToolSchemaEncoder
from ..adapter.validation_only import ValidationOnlyAdapter
from ..core import PatternEngine, PatternManager, State
from ..core.config import CoreConfig
from ..pipeline import HPMPipeline
from ..polygraphs.nlp import NLPPolygraphGenerator


@dataclass(frozen=True, slots=True)
class SymbolicMatchingResult:
    tool_accuracy: float
    parameter_f1: float
    distractor_rejection: float
    status: str


class SymbolicPatternMatchingBenchmark:
    """Benchmark for natural language to tool invocation matching."""

    def __init__(self, tool_corpus: list[dict[str, Any]]) -> None:
        self.tool_corpus = tool_corpus
        self.engine = PatternEngine(config=CoreConfig(
            max_patterns=512, 
            history_limit=10,
            density_decay=0.001
        ))
        self.manager = PatternManager(promotion_threshold=0.01, min_support=1)
        
        self.synonyms = {
            "temperature": "weather",
            "forecast": "weather",
            "conditions": "weather",
            "send": "email",
            "mail": "email",
            "compute": "calculate",
            "add": "calculate",
        }
        
        self.tokenizer = NLPTokenizer()
        self.phraser = CanonicalPhraser(synonyms=self.synonyms)
        
        self.pipeline = HPMPipeline(
            preprocessor=self.tokenizer,
            engine=self.engine,
            postprocessor=ValidationOnlyAdapter(),
            polygraph_generator=NLPPolygraphGenerator()
        )
        self.pipeline.register_preprocessor(self.phraser)
        
        # Mapping from pattern name to (tool_name, is_parameter)
        self.pattern_metadata: dict[str, tuple[str | None, bool]] = {}

    def _get_canonical_sequence(self, query: str, placeholders: dict[str, str]) -> list[str]:
        self.phraser.placeholders = placeholders
        packet = AdapterPacket(raw=query)
        packet = self.pipeline.preprocessing_pipeline.run(packet, target_outputs=["canonical_phraser"])
        return packet.context.get("canonical_tokens", [])

    def train(self, training_data: list[dict[str, Any]], epochs: int = 15) -> None:
        """Train the engine on labeled natural language queries."""
        print(f"Training on {len(training_data)} queries over {epochs} epochs...")
        start_state = State(value=(0.0,))
        
        for epoch in range(epochs):
            for item in training_data:
                seq = self._get_canonical_sequence(item["query"], item["placeholders"])
                
                for token in seq:
                    self.engine.current_state = start_state
                    self.pipeline.step(token, goal={"utility": 1.0}, context={"tool": item["tool"]})
                    
                    match = self.engine.last_match
                    if match and match.pattern:
                        is_param = token.startswith("PARAM_")
                        self.pattern_metadata[match.pattern.name] = (item["tool"], is_param)
                        match.pattern.reward(1.0)
            
            self.manager.end_episode(self.engine)

    def run_test(self, test_data: list[dict[str, Any]], distractor_data: list[str]) -> SymbolicMatchingResult:
        """Evaluate on unseen queries and distractors."""
        print(f"Testing on {len(test_data)} queries...")
        start_state = State(value=(0.0,))
        
        tool_matches = 0
        param_tp = 0
        param_fp = 0
        param_fn = 0
        
        for item in test_data:
            seq = self._get_canonical_sequence(item["query"], item["placeholders"])
            
            votes: dict[str, float] = {}
            max_conf = 0.0
            detected_params = set()
            expected_params = {p for p in seq if p.startswith("PARAM_")}
            
            for token in seq:
                if len(token) <= 1:
                    continue
                self.engine.current_state = start_state
                self.pipeline.step(token, context={"test": True})
                
                match = self.engine.last_match
                if match and match.pattern:
                    p = match.pattern
                    meta = self.pattern_metadata.get(p.name)
                    if meta:
                        tool, is_param = meta
                        if tool:
                            votes[tool] = votes.get(tool, 0.0) + (1.0 / (1.0 + match.distance))
                        if is_param:
                            detected_params.add(token)
            
            best_tool = max(votes.items(), key=lambda x: x[1])[0] if votes else None
            is_correct = best_tool == item["tool"]
            
            if is_correct:
                tool_matches += 1
            
            # Parameter F1 Calculation
            param_tp += len(detected_params.intersection(expected_params))
            param_fp += len(detected_params - expected_params)
            param_fn += len(expected_params - detected_params)
            
        distractor_hits = 0
        for query in distractor_data:
            seq = self._get_canonical_sequence(query, {})
            max_match_score = 0.0
            for token in seq:
                if len(token) <= 1:
                    continue
                self.engine.current_state = start_state
                self.pipeline.step(token, context={"test": True})
                match = self.engine.last_match
                if match and match.status == "exact" and match.pattern and match.pattern.name in self.pattern_metadata:
                    meta = self.pattern_metadata[match.pattern.name]
                    if meta[0] is not None or meta[1]:
                        max_match_score = 1.0
            
            if max_match_score > 0.5:
                distractor_hits += 1
                
        total_test = len(test_data)
        total_dist = len(distractor_data)
        
        precision = param_tp / (param_tp + param_fp) if (param_tp + param_fp) > 0 else 1.0
        recall = param_tp / (param_tp + param_fn) if (param_tp + param_fn) > 0 else 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return SymbolicMatchingResult(
            tool_accuracy=tool_matches / total_test if total_test > 0 else 1.0,
            parameter_f1=f1,
            distractor_rejection=(total_dist - distractor_hits) / total_dist if total_dist > 0 else 1.0,
            status="complete"
        )
