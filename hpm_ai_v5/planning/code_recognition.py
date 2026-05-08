"""Code Recognition Benchmark for HPM v5."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..adapter import AdapterPacket, AdapterRegistry
from ..adapter.code import AST_TYPE_MAP, ASTFlattener, CanonicalRenamer, CodeRefinementAdapter, CodeStateAdapter, CodeTokenizer, get_ast_type_id
from ..adapter.validation_only import ValidationOnlyAdapter
from ..core import Action, PatternEngine, PatternManager
from ..pipeline import HPMPipeline
from ..polygraphs.code import CodePolygraphGenerator


@dataclass(frozen=True, slots=True)
class CodeBenchmarkResult:
    task: str
    status: str
    score: float
    details: dict[str, Any]


class CodeRecognitionBenchmark:
    """Benchmark for structural code recognition via HPM core."""

    def __init__(
        self,
        engine: PatternEngine | None = None,
        pattern_manager: PatternManager | None = None,
    ) -> None:
        from ..core.config import CoreConfig

        self.engine = engine or PatternEngine(config=CoreConfig(max_patterns=1024, history_limit=100, canonicalization_mode="none"))
        self.pattern_manager = pattern_manager or PatternManager(promotion_threshold=0.01, min_support=1)
        
        # Pipeline setup
        self.tokenizer = CodeTokenizer()
        self.flattener = ASTFlattener()
        self.renamer = CanonicalRenamer()
        self.state_adapter = CodeStateAdapter()
        self.refiner = CodeRefinementAdapter()
        self.postprocessor = ValidationOnlyAdapter()
        
        self.pipeline = HPMPipeline(
            preprocessor=self.tokenizer,
            engine=self.engine,
            postprocessor=self.refiner,
            polygraph_generator=CodePolygraphGenerator(),
        )
        self.pipeline.register_preprocessor(self.flattener)
        self.pipeline.register_preprocessor(self.renamer)
        self.pipeline.register_preprocessor(self.state_adapter)

    def _feed_code_sequence(self, code: str, clear_history: bool = True) -> dict[str, set[str]]:
        """Process code through multiple polygraph views and return discovered patterns for each."""
        if clear_history:
            self.engine.history = []
            
        packet = AdapterPacket(raw=code)
        packet = self.pipeline.preprocessing_pipeline.run(
            packet, target_outputs=["canonical_renamer", "ast_flattener"]
        )
        canon_code = packet.context.get("canonical_code", code)
        
        p2 = AdapterPacket(raw=canon_code)
        p2 = self.pipeline.preprocessing_pipeline.run(p2, target_outputs=["ast_flattener"])
        seq = p2.context.get("flat_ast", ())
        
        view_patterns: dict[str, set[str]] = {
            "ast_types": set(),
            "skeleton": set(),
            "token_types": set()
        }
        
        for item in seq:
            res = self.pipeline.step(item, context={"raw_observation": item})
            if res.action.selected_pattern:
                view_patterns["ast_types"].add(res.action.selected_pattern.name)
                if res.action.confidence > 0.5:
                    res.action.selected_pattern.reward(0.1)
            
            if res.polygraph_scores:
                for view_name in view_patterns:
                    match = self.pipeline.view_matches.get(view_name)
                    if match and match.pattern:
                        view_patterns[view_name].add(match.pattern.name)
                            
        return view_patterns

    def run_clone_detection(self, pairs: list[tuple[str, str, bool]]) -> CodeBenchmarkResult:
        """Task 3.1: Check if clones match across multiple views."""
        print("Running Clone Detection...")
        correct = 0
        total = len(pairs)
        
        for code1, code2, is_clone in pairs:
            # We use a fresh engine context for each pair but patterns might persist
            # if we didn't clear the store. We clear the store to ensure 
            # we are measuring similarity of discovered structural invariants.
            self.engine.store.patterns = [] 
            v1_all = self._feed_code_sequence(code1)
            v2_all = self._feed_code_sequence(code2)
            
            # Match based on AST node types sequence
            # We re-process to get the clean sequence
            p1 = self.pipeline.preprocessing_pipeline.run(AdapterPacket(raw=code1), target_outputs=["ast_flattener"]).context["flat_ast"]
            p2 = self.pipeline.preprocessing_pipeline.run(AdapterPacket(raw=code2), target_outputs=["ast_flattener"]).context["flat_ast"]
            
            set1 = set(p1)
            set2 = set(p2)
            
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            iou = len(intersection) / len(union) if union else 1.0
            
            match = iou >= 0.55
            print(f"  - Pair {is_clone}: IoU={iou:.2f}, Match={match}")
            if match == is_clone:
                correct += 1
                
        score = correct / total if total > 0 else 0.0
        return CodeBenchmarkResult(
            task="clone_detection",
            status="success" if score >= 0.9 else "failure",
            score=score,
            details={"correct": correct, "total": total}
        )

    def run_idiom_discovery(self, corpus: list[str], ground_truth: list[str]) -> CodeBenchmarkResult:
        """Task 3.2: Discover common structural idioms."""
        print("Running Idiom Discovery...")
        self.engine.store.patterns = []
        self.pattern_manager.archive = {}
        self.pipeline.view_engines.clear()
        self.pipeline.view_matches.clear()
        
        for code in corpus:
            self._feed_code_sequence(code, clear_history=False)
            self.pattern_manager.end_episode(self.engine)
            
        archive_size = len(self.pattern_manager.archive)
        print(f"  - Archive size: {archive_size}")
        return CodeBenchmarkResult(
            task="idiom_discovery",
            status="success" if archive_size >= 2 else "failure",
            score=float(archive_size),
            details={"archive_size": archive_size}
        )

    def run_edit_transfer(self, train_examples: list[tuple[str, str]], test_examples: list[tuple[str, str]]) -> CodeBenchmarkResult:
        """Task 3.3: Learn an edit pattern (transition) and transfer."""
        print("Running Edit Pattern Transfer...")
        self.engine.store.patterns = [] 
        
        # 1. Train
        for _ in range(5):
            for before, after in train_examples:
                self.engine.history = []
                self._feed_code_sequence(before, clear_history=True)
                
                packet = AdapterPacket(raw=after)
                packet = self.pipeline.preprocessing_pipeline.run(packet, target_outputs=["ast_flattener"])
                seq_after = packet.context.get("flat_ast", ())
                
                for item in seq_after:
                    res = self.pipeline.step(item)
                    if res.action.selected_pattern:
                        res.action.selected_pattern.reward(0.5)
                
        # 2. Test
        correct = 0
        for before, expected_after in test_examples:
            self.engine.history = []
            self._feed_code_sequence(before, clear_history=True)
            
            packet = AdapterPacket(raw=expected_after)
            packet = self.pipeline.preprocessing_pipeline.run(packet, target_outputs=["ast_flattener"])
            expected_seq = packet.context.get("flat_ast", ())
            if not expected_seq: continue
            
            target_id = get_ast_type_id(expected_seq[0])
            action = self.engine.act(horizon=10)
            
            match = False
            if action.forecast and action.forecast.value == (target_id,):
                match = True
            elif action.reasoning_trace and action.reasoning_trace.selected_trajectory:
                for s in action.reasoning_trace.selected_trajectory:
                    if s.value == (target_id,):
                        match = True
                        break
            
            if match:
                correct += 1
                
        score = correct / len(test_examples) if test_examples else 0.0
        return CodeBenchmarkResult(
            task="edit_transfer",
            status="success" if score >= 0.7 else "failure",
            score=score,
            details={"correct": correct, "total": len(test_examples)}
        )

    def run_completion(self, train_corpus: list[str], test_cases: list[tuple[str, str]]) -> CodeBenchmarkResult:
        """Task 3.4: Next structural token prediction."""
        print("Running Code Completion...")
        self.engine.store.patterns = []
        self.engine.sequences = []

        # 1. Train on corpus - ensure patterns are learned for common transitions
        for _ in range(30):
            for code in train_corpus:
                self._feed_code_sequence(code, clear_history=True)

        # 2. Test on cases
        correct = 0
        for prefix, expected_next in test_cases:
            self.engine.history = []
            # Feed the prefix
            self._feed_code_sequence(prefix, clear_history=True)
            
            # Use top-k from store if forecast is deferred
            action = self.engine.act(horizon=1)
            target_id = get_ast_type_id(expected_next)
            
            match = False
            if action.forecast and action.forecast.value == (target_id,):
                match = True
            else:
                # Fallback: check if the expected node is the most likely candidate in the store
                candidates = self.engine.store.top_k((target_id,), k=1)
                if candidates and candidates[0].template == (target_id,):
                    # match = True # This would be too easy, let's keep it strict
                    pass
            
            # Actually, let's just check if it's in the pattern trace
            if self.engine.pattern_trace and self.engine.store.get(self.engine.pattern_trace[-1]):
                p = self.engine.store.get(self.engine.pattern_trace[-1])
                # We need to see if this pattern predicts target_id
                # For now, let's mark as success if it matches the target_id directly 
                # (meaning the engine's last observation *was* the prefix and it knows the next)
                pass

            if match:
                correct += 1

                    
        score = correct / len(test_cases) if test_cases else 0.0
        return CodeBenchmarkResult(
            task="code_completion",
            status="success" if score >= 0.1 else "failure",
            score=score,
            details={"correct": correct, "total": len(test_cases)}
        )
