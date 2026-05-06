"""Cross-Language Transfer (CLT) Benchmark for HPM v5."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..adapter import AdapterPacket
from ..adapter.clt import CLTRefinementAdapter, LanguageDetector, UnifiedASTFlattener, UnifiedStateAdapter, UnifiedVocabulary
from ..adapter.validation_only import ValidationOnlyAdapter
from ..core import PatternEngine, PatternManager
from ..pipeline import HPMPipeline
from ..polygraphs.clt import CLTPolygraphGenerator


@dataclass(frozen=True, slots=True)
class CLTBenchmarkResult:
    task: str
    status: str
    score: float
    details: dict[str, Any]


class CrossLanguageTransferBenchmark:
    """Benchmark for cross-language structural transfer."""

    def __init__(
        self,
        engine: PatternEngine | None = None,
        pattern_manager: PatternManager | None = None,
    ) -> None:
        from ..core.config import CoreConfig

        self.engine = engine or PatternEngine(config=CoreConfig(max_patterns=1024, history_limit=100))
        self.pattern_manager = pattern_manager or PatternManager(promotion_threshold=0.01, min_support=1)
        
        # CLT Pipeline setup
        self.lang_detector = LanguageDetector()
        self.ast_flattener = UnifiedASTFlattener()
        self.state_adapter = UnifiedStateAdapter()
        self.refiner = CLTRefinementAdapter()
        
        self.pipeline = HPMPipeline(
            preprocessor=self.lang_detector,
            engine=self.engine,
            postprocessor=self.refiner,
            polygraph_generator=CLTPolygraphGenerator(),
        )
        self.pipeline.register_preprocessor(self.ast_flattener)
        self.pipeline.register_preprocessor(self.state_adapter)

    def _feed_code(self, code: str, clear_history: bool = True) -> list[str]:
        """Process code and return discovered patterns."""
        if clear_history:
            self.engine.history = []
            
        packet = AdapterPacket(raw=code)
        # Run preprocessing to get unified_ast
        packet = self.pipeline.preprocessing_pipeline.run(packet, target_outputs=["unified_ast_flattener"])
        nodes = packet.context.get("unified_ast", ())
        
        patterns = []
        for node in nodes:
            node_id = UnifiedVocabulary.get_id(node)
            res = self.pipeline.step(node_id, context={"raw_observation": node, "language": packet.context["language"]})
            if res.action.selected_pattern:
                patterns.append(res.action.selected_pattern.name)
        return patterns

    def run_transfer_task(self, name: str, train_py: list[str], test_java: list[str]) -> CLTBenchmarkResult:
        """Evaluate zero-shot transfer from Python to Java."""
        print(f"Running CLT Task: {name}...")
        
        # 1. Train on Python
        self.engine.store.clear()
        for code in train_py:
            self._feed_code(code, clear_history=False)
            self.pattern_manager.end_episode(self.engine)
            
        trained_patterns = {p.name for p in self.engine.store.patterns} | {p.name for p in self.engine.store.meta_patterns}
        print(f"  - Trained patterns (Python): {len(trained_patterns)}")
        
        # 2. Test Zero-Shot on Java
        matches = 0
        total = len(test_java)
        
        for code in test_java:
            # We want to see if the Java code activates any of the patterns learned from Python
            found = self._feed_code(code, clear_history=True)
            # A match occurs if we find at least one pattern from the trained set
            if any(p in trained_patterns for p in found):
                matches += 1
                
        score = matches / total if total > 0 else 0.0
        return CLTBenchmarkResult(
            task=name,
            status="success" if score >= 0.8 else "failure",
            score=score,
            details={"matches": matches, "total": total}
        )
