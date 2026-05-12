#!/usr/bin/env python3
"""
Example runner for the V6 dataset training agent.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hpm_ai_v6.agents.dataset_training_agent import DatasetTrainingAgent
from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader


def main():
    reader = MultiAgentReader("hpm_ai_v6/data/corpus/alice_mini.txt", warm_start=True)
    reader.train(
        episodes=1,
        max_chunks=2,
        max_words_per_chunk=32,
        enable_pruning=False,
        enable_causal=False,
    )

    dataset_agent = DatasetTrainingAgent(
        reader,
        corpus_path="hpm_ai_v6/data/corpus/alice_mini.txt",
    )

    added = dataset_agent.add_from_gutenberg(
        book_ids=DatasetTrainingAgent.curated_gutenberg_book_ids(),
        top_k=12,
        min_score=0.05,
    )
    print(f"Added {added} examples.")
    print(reader.generate("Alice was", max_length=20))


if __name__ == "__main__":
    main()
