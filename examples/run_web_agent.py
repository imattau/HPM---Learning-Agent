#!/usr/bin/env python3
"""
Demonstrate the V6 WebAgent using explicit URLs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader
from hpm_ai_v6.agents.web_agent import WebAgent


def main():
    corpus_path = "hpm_ai_v6/data/corpus/alice_mini.txt"
    reader = MultiAgentReader(corpus_path=corpus_path, warm_start=True)
    reader.train(
        episodes=1,
        max_chunks=2,
        max_words_per_chunk=32,
        enable_pruning=False,
        enable_causal=False,
    )

    web = WebAgent(reader, corpus_path=corpus_path, min_sentence_len=20)
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
    ]
    added = web.add_informative_sentences(urls, top_k=20, min_score=0.5)
    print(f"Added {added} new sentences.")
    print(reader.generate("Artificial intelligence is", max_length=30))


if __name__ == "__main__":
    main()
