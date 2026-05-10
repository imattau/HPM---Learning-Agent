"""Two-Agent ATIS Verification Script (Subset)"""
from __future__ import annotations
import random
from hpm_ai_v5.experiments.run_atis_two_agent_benchmark import TwoAgentATISBenchmark

class TwoAgentATISSubset(TwoAgentATISBenchmark):
    def __init__(self, consolidation: bool = True):
        super().__init__(consolidation=consolidation)
        # Low threshold to trigger consolidation quickly on small subsets
        self.config.max_patterns = 200
        self.engine.store.max_patterns = 200
        self.config.consolidation_threshold = 0.5 # Trigger at 100 patterns

    def run_benchmark(self, train: list[dict], test: list[dict]):
        print("Running B1/B4 subset...")
        return super().run_benchmark(train[:100], test[:20])

    def run_b2(self, train: list[dict], test: list[dict]) -> float:
        print("Running B2 subset...")
        return super().run_b2(train[:100], test[:50])

    def run_b3(self, train: list[dict]) -> dict:
        print("Running B3 subset...")
        return super().run_b3(train[:100])

    def run_b4(self, test: list[dict]) -> float:
        print("Running B4 subset...")
        return super().run_b4(test[:20])

if __name__ == "__main__":
    random.seed(42)
    benchmark = TwoAgentATISSubset()
    benchmark.run_all()
